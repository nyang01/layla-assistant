"""Long-term memory for Layla — persists user info, contacts, and facts across sessions."""

import json
import re
from collections import Counter
from pathlib import Path

from auth import get_credentials
from googleapiclient.discovery import build

MEMORY_FILE = Path(__file__).parent / "memory.json"

# Patterns for automated/business emails to filter out (checked against full email)
JUNK_EMAIL_PATTERNS = [
    "noreply", "no-reply", "no.reply", "donotreply", "do-not-reply",
    "notification", "alert", "support", "newsletter",
    "mailer-daemon", "postmaster", "billing", "invoice",
    "unsubscribe", "promo", "marketing", "updates@",
    "info@", "admin@", "system@", "service@",
    "hello@", "hi@", "help@", "team@", "contact@",
    "offers@", "news@", "mail@", "customerservice",
    "statements@", "welcome@", "feedback@",
    "accounts@", "it@", "communications@", "csuk@",
    "c_tax@", "bydapp@", "premium", "tax@",
]

# Domain patterns that indicate mass/business email (checked against domain only)
JUNK_DOMAIN_PATTERNS = [
    "mail.", "mkt.", "e.", "email", "notify",
    "messages.", "travel.", "info.", "send.",
    "calendar.", "luma-mail", "welcome.",
    "advice.", "campaign", "hubspot",
    "mailchimp", "sendgrid", "marketo",
    ".gov.uk", ".auto",
]


def load_memory(user_id: str | None = None) -> dict:
    """Load memory from database (multi-user) or disk (legacy).

    When user_id is provided, loads from database.
    When None, falls back to file-based storage.
    """
    if user_id:
        from database import get_user_memory
        return get_user_memory(user_id)

    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {"user_email": "", "user_name": "", "contacts": {}, "facts": [], "last_session_timestamp": None}


def save_memory(memory: dict, user_id: str | None = None) -> None:
    """Save memory to database (multi-user) or disk (legacy)."""
    if user_id:
        from database import save_user_memory
        save_user_memory(user_id, memory)
        return

    MEMORY_FILE.write_text(json.dumps(memory, indent=2, ensure_ascii=False))


def save_fact(fact: str, user_id: str | None = None) -> dict:
    """Save a user fact/preference to long-term memory."""
    memory = load_memory(user_id)
    # Avoid duplicates
    if fact not in memory["facts"]:
        memory["facts"].append(fact)
        save_memory(memory, user_id)
    return {"status": "success", "fact": fact}


def delete_fact(fact_keyword: str, user_id: str | None = None) -> dict:
    """Delete a fact from long-term memory that matches the keyword."""
    memory = load_memory(user_id)
    keyword_lower = fact_keyword.lower()
    removed = []
    remaining = []
    for fact in memory["facts"]:
        if keyword_lower in fact.lower():
            removed.append(fact)
        else:
            remaining.append(fact)
    memory["facts"] = remaining
    save_memory(memory, user_id)
    return {"status": "success", "removed": removed, "remaining_count": len(remaining)}


def _parse_email_address(from_header: str) -> tuple[str, str]:
    """Extract name and email from a From header like 'John Doe <john@example.com>'."""
    match = re.match(r"^(.+?)\s*<(.+?)>$", from_header.strip())
    if match:
        name = match.group(1).strip().strip('"').strip("'")
        email = match.group(2).strip()
        return name, email
    # Just an email address with no name
    email = from_header.strip().strip("<>")
    name = email.split("@")[0]
    return name, email


def _is_junk_email(email: str) -> bool:
    """Check if an email address is automated/business junk."""
    email_lower = email.lower()
    # Check local part and full address against junk patterns
    if any(pattern in email_lower for pattern in JUNK_EMAIL_PATTERNS):
        return True
    # Check domain for mass-email infrastructure patterns
    domain = email_lower.split("@")[-1] if "@" in email_lower else ""
    if any(pattern in domain for pattern in JUNK_DOMAIN_PATTERNS):
        return True
    # Personal emails typically use common providers or short domains
    return False


def update_contacts_from_headers(from_headers: list[str], user_id: str | None = None) -> None:
    """Update contacts in memory from a list of From headers."""
    memory = load_memory(user_id)
    for header in from_headers:
        name, email = _parse_email_address(header)
        if not _is_junk_email(email) and name and email:
            memory["contacts"][name] = email
    save_memory(memory, user_id)


def detect_user_email(user_id: str | None = None, credentials=None) -> str:
    """Fetch the authenticated user's Gmail address."""
    memory = load_memory(user_id)
    if memory["user_email"]:
        return memory["user_email"]

    creds = credentials or get_credentials()
    service = build("gmail", "v1", credentials=creds)
    profile = service.users().getProfile(userId="me").execute()
    email = profile.get("emailAddress", "")

    memory["user_email"] = email
    # Try to extract name from email
    if not memory["user_name"]:
        memory["user_name"] = email.split("@")[0].replace(".", " ").title()
    save_memory(memory, user_id)
    return email


def bootstrap_contacts(user_id: str | None = None, credentials=None) -> dict:
    """Scan last 3 months of emails and build a frequent contacts list.

    Fetches headers only (lightweight), counts sender frequency,
    filters out junk, keeps top 20 personal contacts.
    """
    memory = load_memory(user_id)

    creds = credentials or get_credentials()
    service = build("gmail", "v1", credentials=creds)

    # Search for emails from the last 3 months
    results = service.users().messages().list(
        userId="me",
        q="newer_than:3m",
        maxResults=200,
        labelIds=["INBOX"],
    ).execute()

    messages = results.get("messages", [])
    if not messages:
        return {"status": "success", "contacts_found": 0}

    # Fetch From headers for all messages (batch-friendly: metadata only)
    sender_counter = Counter()
    sender_emails = {}

    for msg_meta in messages:
        try:
            msg = service.users().messages().get(
                userId="me",
                id=msg_meta["id"],
                format="metadata",
                metadataHeaders=["From"],
            ).execute()
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            from_header = headers.get("From", "")
            if from_header:
                name, email = _parse_email_address(from_header)
                if not _is_junk_email(email) and name and email:
                    sender_counter[name] += 1
                    sender_emails[name] = email
        except Exception:
            continue

    # Keep top 20 most frequent contacts
    top_contacts = sender_counter.most_common(20)
    for name, _count in top_contacts:
        memory["contacts"][name] = sender_emails[name]

    save_memory(memory, user_id)
    return {"status": "success", "contacts_found": len(top_contacts)}
