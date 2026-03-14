"""Layla AI Personal Assistant — FastAPI server with multi-user OAuth."""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import urlencode

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from agent import process_message
from memory import bootstrap_contacts, detect_user_email, load_memory, save_memory
from session import SessionManager

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Add it to .env file.")
genai.configure(api_key=api_key)

# OAuth config (multi-user)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")
GOOGLE_SCOPES = "openid email profile https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/calendar"


# --- Auth middleware ---

async def get_current_user(authorization: str = Header(None)) -> dict:
    """Validate Bearer token and return user dict.

    Falls back to legacy single-user mode if no auth header and token.json exists.
    """
    if authorization and authorization.startswith("Bearer "):
        api_token = authorization.removeprefix("Bearer ").strip()
        try:
            from database import get_user_by_api_token
            user = get_user_by_api_token(api_token)
        except Exception:
            raise HTTPException(status_code=503, detail="Database not configured. Visit /login to set up multi-user mode.")
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API token")
        return user

    # Legacy fallback: if token.json exists, run in single-user mode
    token_path = os.path.join(os.path.dirname(__file__), "token.json")
    if os.path.exists(token_path):
        return {"id": None, "email": "", "name": ""}

    raise HTTPException(
        status_code=401,
        detail="Missing Authorization header. Visit /login to get your API token.",
    )


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks."""
    # Initialize database
    if GOOGLE_CLIENT_ID:
        from database import init_db
        init_db()

    # Legacy single-user bootstrap (when token.json exists)
    token_path = os.path.join(os.path.dirname(__file__), "token.json")
    if os.path.exists(token_path):
        print("[Startup] Legacy mode: detecting user email...")
        try:
            email = detect_user_email()
            print(f"[Startup] User email: {email}")
        except Exception as e:
            print(f"[Startup] Could not detect email: {e}")

        memory = load_memory()
        if not memory.get("contacts"):
            print("[Startup] Bootstrapping contacts from last 3 months...")
            try:
                result = bootstrap_contacts()
                print(f"[Startup] Found {result['contacts_found']} frequent contacts")
            except Exception as e:
                print(f"[Startup] Contact bootstrap failed: {e}")
        else:
            print(f"[Startup] {len(memory['contacts'])} contacts already in memory")
    else:
        print("[Startup] Multi-user mode — no legacy token.json")

    yield  # Server runs


app = FastAPI(title="Layla AI Personal Assistant", lifespan=lifespan)
session_manager = SessionManager()


# --- Models ---

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    action: str = "continue"


# --- Chat endpoint ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    """Process a voice command from the iOS Shortcut."""
    user_id = user["id"]  # None for legacy mode

    if not request.message.strip():
        return ChatResponse(reply="I didn't catch that. Could you say it again?")

    # Handle goodbye on the server side for clean shortcut exit
    words = request.message.lower().split()
    if "goodbye" in words or "bye" in words or "bye-bye" in words:
        session_key = user_id or request.user_id
        print(f"\n[User: {session_key}] {request.message}")
        print("[Layla] Goodbye! (session ending)")
        # Save last session timestamp before clearing
        memory = load_memory(user_id)
        memory["last_session_timestamp"] = datetime.now().isoformat()
        save_memory(memory, user_id)
        session_manager._sessions.pop(session_key, None)
        return ChatResponse(reply="Goodbye! Talk to you later.", action="stop")

    try:
        session_key = user_id or request.user_id
        history = session_manager.get_or_create(session_key)
        print(f"\n[User: {session_key}] {request.message}")

        reply = process_message(request.message, history, user_id=user_id)

        print(f"[Layla] {reply}")
        return ChatResponse(reply=reply)

    except Exception as e:
        print(f"[Error] {e}")
        raise HTTPException(
            status_code=500,
            detail="Sorry, something went wrong. Please try again.",
        )


# --- Health check ---

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "agent": "layla"}


# --- OAuth routes ---

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve the login page with Sign in with Google button."""
    if not GOOGLE_CLIENT_ID:
        return HTMLResponse(
            "<h2>OAuth not configured</h2><p>Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env</p>",
            status_code=503,
        )
    return HTMLResponse(_LOGIN_HTML)


@app.get("/auth/google")
async def auth_google():
    """Redirect to Google's OAuth consent screen."""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "access_type": "offline",
        "prompt": "consent",
    }
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return RedirectResponse(url)


@app.get("/auth/callback")
async def auth_callback(code: str):
    """Handle Google OAuth callback — exchange code for tokens and create user."""
    from database import create_user, get_user_by_google_id, update_user_tokens

    # Exchange authorization code for tokens
    token_response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )

    if token_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

    tokens = token_response.json()
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token", "")

    if not refresh_token:
        raise HTTPException(
            status_code=400,
            detail="No refresh token received. Please revoke access at https://myaccount.google.com/permissions and try again.",
        )

    # Get user info from Google
    userinfo_response = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    userinfo = userinfo_response.json()
    if userinfo_response.status_code != 200 or "id" not in userinfo:
        print(f"[OAuth] Userinfo error: {userinfo}")
        raise HTTPException(status_code=400, detail=f"Failed to get user info from Google: {userinfo.get('error', 'unknown error')}")
    google_id = userinfo["id"]
    email = userinfo.get("email", "")
    name = userinfo.get("name", "")

    # Create or update user
    existing = get_user_by_google_id(google_id)
    if existing:
        update_user_tokens(
            google_id,
            access_token,
            tokens.get("expires_in", ""),
            refresh_token,
        )
        api_token = existing["api_token"]
    else:
        user = create_user(google_id, email, name, refresh_token)
        api_token = user["api_token"]

        # Bootstrap contacts for new user in background
        try:
            from auth import get_credentials_for_user
            creds = get_credentials_for_user(google_id)
            detect_user_email(user_id=google_id, credentials=creds)
            bootstrap_contacts(user_id=google_id, credentials=creds)
            print(f"[OAuth] Bootstrapped contacts for {email}")
        except Exception as e:
            print(f"[OAuth] Contact bootstrap failed for {email}: {e}")

    return RedirectResponse(f"/dashboard?token={api_token}&name={name}")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(token: str, name: str = ""):
    """Show post-login page with API token and setup instructions."""
    # Derive server URL from GOOGLE_REDIRECT_URI (strip /auth/callback)
    server_url = GOOGLE_REDIRECT_URI.replace("/auth/callback", "")
    return HTMLResponse(_dashboard_html(token, name, server_url))


# --- HTML Templates ---

_LOGIN_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Layla — Sign In</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    min-height: 100vh;
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 40%, #16213e 70%, #1a1035 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, 'Segoe UI', sans-serif; color: white;
  }
  .card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px; padding: 48px;
    text-align: center; max-width: 420px; width: 90%;
    backdrop-filter: blur(20px);
  }
  .logo { font-size: 56px; font-weight: 700; margin-bottom: 8px;
    background: linear-gradient(135deg, #FF9A6C, #FF6B8A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .subtitle { color: rgba(255,255,255,0.5); font-size: 18px; margin-bottom: 32px; }
  .desc { color: rgba(255,255,255,0.4); font-size: 14px; line-height: 1.6; margin-bottom: 32px; }
  .google-btn {
    display: inline-flex; align-items: center; gap: 12px;
    background: white; color: #333; border: none; border-radius: 12px;
    padding: 14px 32px; font-size: 16px; font-weight: 500;
    cursor: pointer; text-decoration: none;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .google-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
  .google-btn svg { width: 20px; height: 20px; }
</style>
</head>
<body>
<div class="card">
  <div class="logo">Layla</div>
  <div class="subtitle">Voice AI Personal Assistant</div>
  <div class="desc">
    Sign in with your Google account to let Layla manage your Gmail,
    Calendar, and more — all through natural voice conversation.
  </div>
  <a class="google-btn" href="/auth/google">
    <svg viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
    Sign in with Google
  </a>
</div>
</body>
</html>"""


def _dashboard_html(token: str, name: str, server_url: str = "") -> str:
    display_name = name or "there"
    api_url = f"{server_url}/api/chat" if server_url else "/api/chat"
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Layla — Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    min-height: 100vh;
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 40%, #16213e 70%, #1a1035 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, 'Segoe UI', sans-serif; color: white;
  }}
  .card {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px; padding: 48px;
    max-width: 620px; width: 90%;
    backdrop-filter: blur(20px);
  }}
  .logo {{ font-size: 36px; font-weight: 700;
    background: linear-gradient(135deg, #FF9A6C, #FF6B8A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .welcome {{ font-size: 22px; margin: 12px 0 24px; color: rgba(255,255,255,0.8); }}
  .section {{ margin-bottom: 24px; }}
  .label {{ font-size: 13px; color: rgba(255,255,255,0.4); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px; }}
  .token-box {{
    background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px; padding: 14px 16px; font-family: monospace;
    font-size: 14px; word-break: break-all; color: #FF9A6C;
    position: relative; cursor: pointer;
  }}
  .token-box:hover {{ background: rgba(0,0,0,0.5); }}
  .copied {{ position: absolute; right: 12px; top: 50%; transform: translateY(-50%);
    color: #34A853; font-size: 13px; font-family: sans-serif; }}
  .steps {{ color: rgba(255,255,255,0.5); font-size: 14px; line-height: 1.8; }}
  .steps li {{ margin-bottom: 8px; }}
  .steps code {{ background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;
    font-size: 13px; color: #FF9A6C; }}
  .config-row {{ display: flex; gap: 8px; align-items: center; margin: 4px 0; }}
  .config-label {{ color: rgba(255,255,255,0.35); font-size: 12px; min-width: 70px;
    text-transform: uppercase; letter-spacing: 0.5px; }}
  .config-value {{ background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 8px 12px; font-family: monospace; font-size: 12px;
    color: #FF9A6C; word-break: break-all; flex: 1; cursor: pointer; }}
  .config-value:hover {{ background: rgba(0,0,0,0.5); }}
</style>
</head>
<body>
<div class="card">
  <div class="logo">Layla</div>
  <div class="welcome">Welcome, {display_name}!</div>

  <div class="section">
    <div class="label">Your API Token</div>
    <div class="token-box" onclick="navigator.clipboard.writeText('{token}'); let c=this.querySelector('.copied'); c.style.display='inline'; setTimeout(()=>c.style.display='none',2000)">
      {token}<span class="copied" style="display:none">Copied!</span>
    </div>
  </div>

  <div class="section">
    <div class="label">iOS Shortcut Setup</div>
    <ol class="steps">
      <li>Download the <strong>Layla</strong> iOS Shortcut (ask the developer for the link)</li>
      <li>Find the <code>Get Contents of URL</code> action and set:
        <div style="margin-top: 6px;">
          <div class="config-row">
            <span class="config-label">URL</span>
            <div class="config-value" onclick="navigator.clipboard.writeText('{api_url}')">{api_url}</div>
          </div>
          <div class="config-row">
            <span class="config-label">Header</span>
            <div class="config-value" onclick="navigator.clipboard.writeText('Bearer {token}')">Authorization: Bearer {token}</div>
          </div>
        </div>
      </li>
      <li>That's it — say <strong>"Hi Layla"</strong> to start!</li>
    </ol>
  </div>

  <div class="section">
    <div class="label">API Usage (curl)</div>
    <div class="token-box" style="font-size: 12px; cursor: pointer; color: rgba(255,255,255,0.6);" onclick="navigator.clipboard.writeText(this.innerText.replace('Copied!','').trim()); let c=this.querySelector('.copied'); c.style.display='inline'; setTimeout(()=>c.style.display='none',2000)">
curl -X POST {api_url} \\
  -H "Authorization: Bearer {token}" \\
  -H "Content-Type: application/json" \\
  -d '{{"message": "read my emails"}}'<span class="copied" style="display:none">Copied!</span>
    </div>
  </div>
</div>
</body>
</html>"""
