"""Gmail API tools for reading and sending emails."""

import base64
import re
from email.mime.text import MIMEText

from googleapiclient.discovery import build

from auth import get_credentials
from memory import update_contacts_from_headers

MAX_BODY_LENGTH = 500  # Truncate long emails for voice readability


_gmail_service = None


def _get_gmail_service(credentials=None):
    """Build Gmail service. Uses provided credentials (multi-user) or legacy singleton."""
    if credentials:
        return build("gmail", "v1", credentials=credentials)
    global _gmail_service
    if _gmail_service is None:
        _gmail_service = build("gmail", "v1", credentials=get_credentials())
    return _gmail_service


def _extract_body(payload: dict) -> str:
    """Extract plain text body from a Gmail message payload."""
    # Simple single-part message
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    # Multipart message — look for text/plain first, then text/html
    parts = payload.get("parts", [])
    plain_text = ""
    html_text = ""

    for part in parts:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data", "")

        if mime == "text/plain" and data:
            plain_text = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        elif mime == "text/html" and data:
            html_text = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        elif mime.startswith("multipart/"):
            # Recurse into nested multipart
            nested = _extract_body(part)
            if nested:
                return nested

    if plain_text:
        return plain_text

    # Fall back to HTML with tags stripped
    if html_text:
        clean = re.sub(r"<[^>]+>", " ", html_text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    return ""


def read_emails(max_results: int = 10, credentials=None, user_id=None) -> dict:
    """Read recent emails from Gmail inbox (primary category only).

    Filters to primary inbox, excluding Promotions, Social, and Updates tabs.

    Args:
        max_results: Maximum number of emails to fetch. Default 10.

    Returns:
        dict with status and list of email summaries including full body text.
    """
    service = _get_gmail_service(credentials)

    results = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        labelIds=["INBOX"],
        q="category:primary",
    ).execute()

    msg_ids = [m["id"] for m in results.get("messages", [])]
    if not msg_ids:
        return {"status": "success", "email_count": 0, "emails": []}

    # Fetch all messages in parallel using batch API
    raw_msgs = [None] * len(msg_ids)

    def _on_message(request_id, response, exception):
        if exception is None:
            idx = int(request_id)
            raw_msgs[idx] = response

    batch = service.new_batch_http_request(callback=_on_message)
    for i, msg_id in enumerate(msg_ids):
        batch.add(
            service.users().messages().get(userId="me", id=msg_id, format="full"),
            request_id=str(i),
        )
    batch.execute()
    raw_msgs = [m for m in raw_msgs if m is not None]

    messages = []
    from_headers = []

    for msg in raw_msgs:
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        from_header = headers.get("From", "Unknown")
        from_headers.append(from_header)

        body = _extract_body(msg["payload"])
        if len(body) > MAX_BODY_LENGTH:
            body = body[:MAX_BODY_LENGTH] + "..."

        messages.append({
            "id": msg["id"],
            "from": from_header,
            "to": headers.get("To", ""),
            "subject": headers.get("Subject", "No subject"),
            "date": headers.get("Date", ""),
            "body": body,
            "snippet": msg.get("snippet", ""),
            "is_unread": "UNREAD" in msg.get("labelIds", []),
        })

    # Mark unread emails as read using batch API
    unread_ids = [m["id"] for m in messages if m["is_unread"]]
    if unread_ids:
        try:
            batch = service.new_batch_http_request()
            for msg_id in unread_ids:
                batch.add(service.users().messages().modify(
                    userId="me", id=msg_id,
                    body={"removeLabelIds": ["UNREAD"]},
                ))
            batch.execute()
        except Exception:
            pass  # Don't fail if marking read fails

    # Update contacts in long-term memory
    try:
        update_contacts_from_headers(from_headers, user_id=user_id)
    except Exception:
        pass  # Don't fail email reading if memory update fails

    return {"status": "success", "email_count": len(messages), "emails": messages}


def send_email(
    to: str,
    subject: str,
    body: str,
    reply_to_message_id: str | None = None,
    credentials=None,
) -> dict:
    """Send an email via Gmail.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
        reply_to_message_id: If replying, the original Gmail message ID.

    Returns:
        dict with status and sent message ID.
    """
    service = _get_gmail_service(credentials)

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    send_body = {}

    if reply_to_message_id:
        # Fetch original message headers for threading
        original = service.users().messages().get(
            userId="me",
            id=reply_to_message_id,
            format="metadata",
            metadataHeaders=["Message-ID", "Subject", "References"],
        ).execute()

        orig_headers = {
            h["name"]: h["value"] for h in original["payload"]["headers"]
        }
        original_message_id = orig_headers.get("Message-ID", "")

        if original_message_id:
            message["In-Reply-To"] = original_message_id
            # Build References chain
            existing_refs = orig_headers.get("References", "")
            if existing_refs:
                message["References"] = f"{existing_refs} {original_message_id}"
            else:
                message["References"] = original_message_id

        # Thread the reply with the original message's thread
        send_body["threadId"] = original.get("threadId", "")

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    send_body["raw"] = raw

    sent = service.users().messages().send(
        userId="me",
        body=send_body,
    ).execute()

    return {"status": "success", "message_id": sent["id"]}


def search_emails(query: str, max_results: int = 10, credentials=None) -> dict:
    """Search Gmail using Gmail search syntax.

    Args:
        query: Gmail search query (e.g. "from:sarah contract", "subject:invoice").
        max_results: Maximum number of results to return. Default 10.

    Returns:
        dict with status and list of matching email summaries.
    """
    service = _get_gmail_service(credentials)

    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results,
    ).execute()

    msg_ids = [m["id"] for m in results.get("messages", [])]
    if not msg_ids:
        return {"status": "success", "result_count": 0, "emails": []}

    raw_msgs = [None] * len(msg_ids)

    def _on_message(request_id, response, exception):
        if exception is None:
            idx = int(request_id)
            raw_msgs[idx] = response

    batch = service.new_batch_http_request(callback=_on_message)
    for i, msg_id in enumerate(msg_ids):
        batch.add(
            service.users().messages().get(userId="me", id=msg_id, format="full"),
            request_id=str(i),
        )
    batch.execute()
    raw_msgs = [m for m in raw_msgs if m is not None]

    messages = []
    for msg in raw_msgs:
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}

        body = _extract_body(msg["payload"])
        if len(body) > MAX_BODY_LENGTH:
            body = body[:MAX_BODY_LENGTH] + "..."

        messages.append({
            "id": msg["id"],
            "from": headers.get("From", "Unknown"),
            "to": headers.get("To", ""),
            "subject": headers.get("Subject", "No subject"),
            "date": headers.get("Date", ""),
            "body": body,
            "snippet": msg.get("snippet", ""),
            "is_unread": "UNREAD" in msg.get("labelIds", []),
        })

    return {"status": "success", "result_count": len(messages), "emails": messages}


def get_full_email(message_id: str, credentials=None) -> dict:
    """Read the complete untruncated body of a specific email.

    Args:
        message_id: The Gmail message ID to read in full.

    Returns:
        dict with status and full email content (no body truncation).
    """
    service = _get_gmail_service(credentials)

    msg = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full",
    ).execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    body = _extract_body(msg["payload"])  # No truncation

    return {
        "status": "success",
        "message_id": message_id,
        "from": headers.get("From", "Unknown"),
        "to": headers.get("To", ""),
        "subject": headers.get("Subject", "No subject"),
        "date": headers.get("Date", ""),
        "body": body,
    }


def archive_email(message_id: str, credentials=None) -> dict:
    """Archive an email by removing it from the inbox.

    Args:
        message_id: The Gmail message ID to archive.

    Returns:
        dict with status and details of the archived email.
    """
    service = _get_gmail_service(credentials)

    # Fetch subject for confirmation before archiving
    msg = service.users().messages().get(
        userId="me",
        id=message_id,
        format="metadata",
        metadataHeaders=["Subject", "From"],
    ).execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}

    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"removeLabelIds": ["INBOX"]},
    ).execute()

    return {
        "status": "success",
        "archived_message_id": message_id,
        "subject": headers.get("Subject", "No subject"),
        "from": headers.get("From", "Unknown"),
    }
