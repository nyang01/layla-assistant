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
<title>Layla — Setup Guide</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    min-height: 100vh;
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 40%, #16213e 70%, #1a1035 100%);
    font-family: -apple-system, 'Segoe UI', sans-serif; color: white;
    padding: 40px 16px;
  }}
  .container {{ max-width: 540px; margin: 0 auto; }}

  /* Header */
  .header {{ text-align: center; margin-bottom: 36px; }}
  .logo {{ font-size: 42px; font-weight: 700;
    background: linear-gradient(135deg, #FF9A6C, #FF6B8A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .welcome {{ font-size: 18px; margin-top: 8px; color: rgba(255,255,255,0.6); }}

  /* Steps */
  .step {{ display: flex; gap: 16px; margin-bottom: 28px; }}
  .step-num {{
    flex-shrink: 0; width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #FF9A6C, #FF6B8A);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; color: #0d0d1a; margin-top: 2px;
  }}
  .step-num.done {{ background: #34A853; font-size: 16px; }}
  .step-body {{ flex: 1; }}
  .step-title {{ font-size: 17px; font-weight: 600; margin-bottom: 6px; color: rgba(255,255,255,0.9); }}
  .step-desc {{ font-size: 14px; color: rgba(255,255,255,0.45); line-height: 1.6; }}
  .step-desc strong {{ color: rgba(255,255,255,0.7); }}

  /* Copyable value */
  .copy-box {{
    background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px; padding: 12px 14px; font-family: 'SF Mono', monospace;
    font-size: 13px; word-break: break-all; color: #FF9A6C;
    cursor: pointer; position: relative; margin: 8px 0;
    transition: background 0.15s;
  }}
  .copy-box:hover {{ background: rgba(0,0,0,0.55); }}
  .copy-box .hint {{ position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
    font-family: sans-serif; font-size: 11px; color: rgba(255,255,255,0.25);
    pointer-events: none; }}
  .copy-box .hint.ok {{ color: #34A853; }}

  /* Token box (larger) */
  .token-box {{ font-size: 15px; padding: 14px 16px; letter-spacing: 0.3px; }}

  /* Config rows */
  .config {{ margin: 10px 0 4px; }}
  .config-label {{ font-size: 11px; color: rgba(255,255,255,0.3); text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 4px; }}

  /* Download buttons */
  .dl-btn {{
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px; padding: 14px 18px; text-decoration: none;
    color: white; font-size: 15px; font-weight: 500;
    margin: 8px 0; transition: all 0.15s;
  }}
  .dl-btn:hover {{ background: rgba(255,255,255,0.14); transform: translateY(-1px); }}
  .dl-btn .icon {{ font-size: 22px; }}
  .dl-btn .meta {{ flex: 1; }}
  .dl-btn .dl-name {{ font-weight: 600; }}
  .dl-btn .dl-desc {{ font-size: 12px; color: rgba(255,255,255,0.4); margin-top: 2px; }}
  .dl-btn .arrow {{ color: rgba(255,255,255,0.3); font-size: 18px; }}

  /* Sub-steps */
  .sub-steps {{ margin: 10px 0 0; padding-left: 0; list-style: none; }}
  .sub-steps li {{
    font-size: 13px; color: rgba(255,255,255,0.45); line-height: 1.7;
    padding: 3px 0 3px 20px; position: relative;
  }}
  .sub-steps li::before {{
    content: ''; position: absolute; left: 4px; top: 10px;
    width: 6px; height: 6px; border-radius: 50%;
    background: rgba(255,154,108,0.4);
  }}
  .sub-steps code {{
    background: rgba(255,255,255,0.08); padding: 1px 6px; border-radius: 4px;
    font-size: 12px; color: #FF9A6C;
  }}

  /* Divider */
  .divider {{ border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 32px 0; }}

  /* Curl section */
  .curl-toggle {{
    font-size: 13px; color: rgba(255,255,255,0.3); cursor: pointer;
    text-align: center; padding: 8px;
  }}
  .curl-toggle:hover {{ color: rgba(255,255,255,0.5); }}
  .curl-content {{ display: none; margin-top: 8px; }}
  .curl-content.show {{ display: block; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div class="logo">Layla</div>
    <div class="welcome">Welcome, {display_name}! Follow the steps below to get started.</div>
  </div>

  <!-- Step 1: API Token -->
  <div class="step">
    <div class="step-num">1</div>
    <div class="step-body">
      <div class="step-title">Copy Your API Token</div>
      <div class="step-desc">This is your personal token. Tap to copy it — you'll paste it into the shortcuts below.</div>
      <div class="copy-box token-box" onclick="copyText(this, '{token}')">
        {token}
        <span class="hint">tap to copy</span>
      </div>
    </div>
  </div>

  <!-- Step 2: Download Shortcuts -->
  <div class="step">
    <div class="step-num">2</div>
    <div class="step-body">
      <div class="step-title">Install the Shortcuts</div>
      <div class="step-desc">Download both shortcuts to your iPhone. Open each link in Safari and tap <strong>Add Shortcut</strong>.</div>
      <a class="dl-btn" href="https://www.icloud.com/shortcuts/d57fa0a81e7945498f10f074d1cbf3a3" target="_blank">
        <span class="icon">&#9749;</span>
        <span class="meta">
          <span class="dl-name">Layla</span>
          <span class="dl-desc">Main voice assistant — starts the conversation</span>
        </span>
        <span class="arrow">&#8250;</span>
      </a>
      <a class="dl-btn" href="https://www.icloud.com/shortcuts/71534523de9747c281c8cc6eb8276a5f" target="_blank">
        <span class="icon">&#128172;</span>
        <span class="meta">
          <span class="dl-name">Layla Chat</span>
          <span class="dl-desc">Conversation handler — keeps the dialogue going</span>
        </span>
        <span class="arrow">&#8250;</span>
      </a>
    </div>
  </div>

  <!-- Step 3: Configure -->
  <div class="step">
    <div class="step-num">3</div>
    <div class="step-body">
      <div class="step-title">Configure the Shortcuts</div>
      <div class="step-desc">Open <strong>each</strong> shortcut in the Shortcuts app, tap the <strong>&#8943;</strong> menu to edit, and find the <code>Text</code> actions at the top. Replace the placeholder values:</div>
      <div class="config">
        <div class="config-label">Server URL</div>
        <div class="copy-box" onclick="copyText(this, '{api_url}')">{api_url}<span class="hint">tap to copy</span></div>
      </div>
      <div class="config">
        <div class="config-label">API Token</div>
        <div class="copy-box" onclick="copyText(this, '{token}')">{token}<span class="hint">tap to copy</span></div>
      </div>
      <ul class="sub-steps">
        <li>Open each shortcut and tap the <strong>&#8943;</strong> (three dots) to edit</li>
        <li>Find the <code>Text</code> field containing the server URL — paste yours</li>
        <li>Find the <code>Text</code> field containing the API token — paste yours</li>
        <li>Tap <strong>Done</strong> to save</li>
      </ul>
    </div>
  </div>

  <!-- Step 4: Vocal Shortcut -->
  <div class="step">
    <div class="step-num">4</div>
    <div class="step-body">
      <div class="step-title">Set Up "Hi Layla" Voice Trigger</div>
      <div class="step-desc">This lets you start Layla hands-free — just say <strong>"Hi Layla"</strong> anytime.</div>
      <ul class="sub-steps">
        <li>Open <strong>Settings</strong> on your iPhone</li>
        <li>Go to <strong>Accessibility</strong> &#8250; <strong>Vocal Shortcuts</strong></li>
        <li>Enable <strong>Vocal Shortcuts</strong> toggle</li>
        <li>Tap <strong>Add Action</strong></li>
        <li>Choose <strong>Run Shortcut</strong>, then select <strong>Layla</strong></li>
        <li>Set the custom phrase to <strong>"Hi Layla"</strong></li>
        <li>Tap <strong>Save</strong></li>
      </ul>
    </div>
  </div>

  <!-- Step 5: Done -->
  <div class="step">
    <div class="step-num done">&#10003;</div>
    <div class="step-body">
      <div class="step-title">You're All Set!</div>
      <div class="step-desc">
        Say <strong>"Hi Layla"</strong> to start a conversation. Layla can read your emails,
        manage your calendar, search the web, and remember things for you.
        Say <strong>"Goodbye"</strong> to end a session.
      </div>
    </div>
  </div>

  <hr class="divider">

  <!-- Developer API -->
  <div class="curl-toggle" onclick="this.nextElementSibling.classList.toggle('show')">
    Developer? Show API usage &#9662;
  </div>
  <div class="curl-content">
    <div class="copy-box" style="font-size: 12px; color: rgba(255,255,255,0.5); white-space: pre; overflow-x: auto;" onclick="copyText(this, 'curl -X POST {api_url} -H &quot;Authorization: Bearer {token}&quot; -H &quot;Content-Type: application/json&quot; -d \\'{{\\'message\\': \\'read my emails\\'}}\\'' )">curl -X POST {api_url} \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{{"message": "read my emails"}}'<span class="hint">tap to copy</span></div>
  </div>
</div>

<script>
function copyText(el, text) {{
  navigator.clipboard.writeText(text).then(function() {{
    var hint = el.querySelector('.hint');
    hint.textContent = 'Copied!';
    hint.classList.add('ok');
    setTimeout(function() {{
      hint.textContent = 'tap to copy';
      hint.classList.remove('ok');
    }}, 2000);
  }});
}}
</script>
</body>
</html>"""
