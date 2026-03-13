"""Google OAuth2 credential management for Gmail and Calendar APIs."""

import os
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]

TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")

_cached_creds: Credentials | None = None


def get_credentials() -> Credentials:
    """Load and return valid Google OAuth2 credentials (legacy single-user mode).

    Reads from token.json and auto-refreshes if expired.
    Raises an error if no valid credentials are available.
    """
    global _cached_creds

    if _cached_creds and _cached_creds.valid:
        return _cached_creds

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save refreshed token
            with open(TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
        else:
            raise RuntimeError(
                "No valid credentials found. Run generate_token.py first."
            )

    _cached_creds = creds
    return creds


# --- Multi-user credentials ---

_user_creds_cache: dict[str, Credentials] = {}


def get_credentials_for_user(user_id: str) -> Credentials:
    """Build Google OAuth2 Credentials from database-stored tokens for a specific user.

    Auto-refreshes if expired and saves the new access token back to the database.
    """
    # Check cache first
    cached = _user_creds_cache.get(user_id)
    if cached and cached.valid:
        return cached

    from database import get_user_credentials_data, update_user_tokens

    token_data = get_user_credentials_data(user_id)
    if not token_data:
        raise RuntimeError(f"No credentials found for user {user_id}")

    creds = Credentials(
        token=token_data["access_token"],
        refresh_token=token_data["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=SCOPES,
    )

    # Auto-refresh if expired
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        expiry = creds.expiry.isoformat() if creds.expiry else ""
        update_user_tokens(user_id, creds.token, expiry)

    _user_creds_cache[user_id] = creds
    return creds
