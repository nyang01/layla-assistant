"""Gemini-powered AI personal assistant with extensible tool calling."""

import contextvars
import json
import os
from datetime import datetime

import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types

from tools.gmail_tools import read_emails, send_email, search_emails, archive_email, get_full_email
from tools.calendar_tools import read_calendar, create_event, delete_event, modify_event, list_calendars
from memory import load_memory, save_fact, delete_fact

# Context variables for multi-user credential/identity threading.
# Set before each agent loop; read by tool wrapper functions.
_current_credentials = contextvars.ContextVar('_current_credentials', default=None)
_current_user_id = contextvars.ContextVar('_current_user_id', default=None)

def _build_system_prompt(user_id: str | None = None) -> str:
    """Build the system prompt with the current date/time and long-term memory."""
    now = datetime.now()
    memory = load_memory(user_id)

    # Build memory context
    memory_lines = []
    if memory.get("user_name"):
        memory_lines.append(f"User's name: {memory['user_name']}")
    if memory.get("user_email"):
        memory_lines.append(f"User's email: {memory['user_email']}")
    if memory.get("contacts"):
        contacts_str = ", ".join(
            f"{name} ({email})" for name, email in memory["contacts"].items()
        )
        memory_lines.append(f"Known contacts: {contacts_str}")
    if memory.get("facts"):
        memory_lines.append("Remembered facts about the user:")
        for fact in memory["facts"]:
            memory_lines.append(f"  - {fact}")

    memory_section = "\n".join(memory_lines) if memory_lines else "No stored memory yet."

    hour = now.hour
    if hour < 12:
        time_greeting = "Good morning"
    elif hour < 17:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    name = (memory.get("user_name") or "").split()[0] or "there"

    return f"""You are Layla, your personal AI assistant. You help the user get things done
through natural voice conversation. You can manage emails, calendar, search the web,
remember things, and anything else you have access to. You take action directly — you
don't just talk, you do.

Today is {now.strftime("%A, %B %d, %Y")}. Current time is {now.strftime("%I:%M %p")}.
The appropriate greeting right now is: "{time_greeting}".

USER MEMORY (persisted across sessions):
{memory_section}

PERSONALITY:
- Warm, friendly, and efficient — like a trusted human assistant
- Always confirm actions taken ("I've sent the reply" not "Would you like me to send it?")
- Keep responses under 3 sentences for voice readability
- Use natural time references ("tomorrow at 3pm" not "2026-03-13T15:00:00")
- Address the user by name when appropriate

BEHAVIOR:
- You are an AGENT. Take actions directly. Do not ask for confirmation unless
  the request is truly ambiguous.
- NEVER archive, delete, or send emails unless the user EXPLICITLY asks you to.
  "Read my emails" is NOT a request to archive. "Summarise" is NOT a request to
  archive. Only archive when the user says "archive", "remove", or "clean up".
- If a tool call fails, explain what went wrong in simple, non-technical terms.
- Never expose technical details (API errors, JSON, IDs) to the user.
- NEVER respond with just "..." or empty text. Always give a clear spoken response.
- Proactively offer helpful follow-ups after completing actions.
- GREETINGS are handled automatically before you see them. You will never
  receive a bare greeting like "hi" or "hello" — those are handled by a
  fast path. So NEVER respond with just a greeting. Always answer the
  user's actual question or request.
- AFTER GREETING — READING NEW EMAILS: When the greeting mentioned new emails
  and the user then asks to "read them", "what are they", "summarise", or
  otherwise references those new emails — use search_emails with query
  "is:unread category:primary" (NOT read_emails). read_emails returns the
  latest emails regardless of read status. search_emails with is:unread
  returns exactly the unread emails the greeting reported.

DATA FRESHNESS:
- Tool results are LIVE from the API — always treat them as the current truth.
- If the user asks about something that may have changed since earlier in the
  conversation, call the tool again to get fresh results instead of relying on
  old data from conversation history.

WEB SEARCH:
- You have a web_search tool that gives you access to real-time information.
- Use it for: weather, news, sports, prices, store hours, recipes, travel,
  translations, calculations, general knowledge — anything you don't already know.
- When the user asks a question that needs current data, use web_search.
- Summarize the result concisely for voice (1-3 sentences).

MEMORY:
- When you learn key information about the user — preferences, facts, habits,
  important details — use save_memory to remember it for future conversations.
- Use known contacts to resolve names to email addresses (e.g., if user says
  "send email to Sarah", look up Sarah's email from known contacts).
- Don't save trivial or temporary information. Focus on reusable facts.

YOUR LIMITATIONS — be honest about these:
- You can ONLY respond when the user speaks to you. You cannot initiate
  conversations, send scheduled messages, or wake yourself up.
- You cannot schedule recurring tasks like "brief me every day at 7pm" — you
  have no timer, no cron, no background process. Be upfront about this.
- If the user asks for something you truly can't do, say so clearly and suggest
  an alternative.
- Do NOT save facts about capabilities you don't have (e.g., scheduled briefings).

TOOL-SPECIFIC GUIDANCE:

EMAIL:
- CHOOSING THE RIGHT EMAIL TOOL:
  * read_emails: ONLY for "read my emails" or "check my inbox" with NO qualifiers
    at all. If the user mentions ANY filter — time, sender, topic, unread, today,
    this week, new — you MUST use search_emails instead.
  * search_emails: Use this whenever the user mentions ANY qualifier. Examples:
    - "emails today" → search_emails(query="newer_than:1d category:primary")
    - "new emails" / "unread" → search_emails(query="is:unread category:primary")
    - "emails from Sarah" → search_emails(query="from:sarah category:primary")
    - "emails this week" → search_emails(query="newer_than:7d category:primary")
    You know Gmail operators: newer_than:, older_than:, from:, to:, subject:,
    is:unread, has:attachment, after:, before:, category:, etc.
    IMPORTANT: Always include "category:primary" in search queries.
    Only omit it when the user specifically asks about promotions, social, or
    all emails.
  * get_full_email: When the user wants to read the full content of a specific
    email. Call read_emails or search_emails first to get the message_id.
- Summarize each email in one sentence: who it's from, what it's about
- When the user says "reply to [name]", find the most recent email from that person
  in the conversation history and use its message ID
- When replying to an email, always use the reply_to_message_id parameter to
  thread the reply correctly.
- When the user asks to archive an email, use the message ID from the most recent
  email results in the conversation. Confirm: "I've archived the email about X from Y."
- You can archive emails (remove from inbox) but CANNOT permanently delete them.
  If the user asks to "delete" an email, archive it instead and explain that
  it's been archived (moved out of inbox but still searchable).

CALENDAR:
- When creating calendar events, infer reasonable defaults:
  - No end time specified? Default to 1 hour
  - "Friday" means the next upcoming Friday
  - "Tomorrow" means the next calendar day
- When reading calendar, group events chronologically and state times naturally
- When deleting or cancelling a calendar event, first call read_calendar to find the
  event, then use delete_event with its ID. Confirm: "I've cancelled your Meeting at 3pm."
- When modifying a calendar event, first call read_calendar to find the event. Only
  provide the fields that are changing — duration is preserved automatically.
  "Move to 4pm" → only change start_time. Confirm what changed.
- IMPORTANT — NAMED CALENDARS: When the user mentions ANY calendar by name (e.g.,
  "school calendar", "work calendar", "personal"), you MUST call list_calendars FIRST
  to find the correct calendar ID, THEN call read_calendar with that calendar_id.
  NEVER guess the calendar ID. NEVER use "primary" when the user names a specific
  calendar. The correct flow is: list_calendars → find matching ID → read_calendar.
  When no specific calendar is mentioned, use "primary" (the default).
- REMINDERS: If the user asks you to remind them about something, create a calendar
  event at that time instead. The phone's calendar notification will act as the reminder.
  Example: "Remind me to call Sarah at 3pm" → create event "Call Sarah" at 3pm.
"""

# Tool declarations for Gemini
# Tool wrapper functions with docstrings that Gemini uses as tool declarations.
# The google-generativeai SDK auto-generates schemas from these.

def tool_read_emails(max_results: int = 10) -> dict:
    """Read the user's recent Gmail emails. Returns sender, subject, snippet, and whether each email is unread.
    Adjust max_results based on what the user asks: "read my emails" → 10, "show me all emails today" → 20,
    "show me the latest email" → 1. Listen to the user's intent.

    Args:
        max_results: Number of emails to return. Adjust based on the user's request. Default 10.
    """
    return read_emails(max_results=max_results, credentials=_current_credentials.get(), user_id=_current_user_id.get())


def tool_send_email(to: str, subject: str, body: str, reply_to_message_id: str = "") -> dict:
    """Send an email via Gmail. Can send new emails or reply to existing ones.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
        reply_to_message_id: Gmail message ID of the email being replied to. Use this to thread replies correctly. Leave empty for new emails.
    """
    return send_email(
        to=to,
        subject=subject,
        body=body,
        reply_to_message_id=reply_to_message_id if reply_to_message_id else None,
        credentials=_current_credentials.get(),
    )


def tool_search_emails(query: str, max_results: int = 10) -> dict:
    """Search the user's Gmail using a search query. Supports Gmail search syntax
    like "from:name", "subject:topic", "is:unread", "has:attachment", "newer_than:2d".
    Translate the user's natural language request into an appropriate Gmail search query.

    Args:
        query: Gmail search query string. Examples: "from:sarah contract", "subject:invoice newer_than:7d".
        max_results: Maximum number of results to return. Default 10.
    """
    return search_emails(query=query, max_results=max_results, credentials=_current_credentials.get())


def tool_archive_email(message_id: str) -> dict:
    """Archive an email by removing it from the inbox. The email is NOT deleted — it can
    still be found via search. Use the message ID from a previous read_emails or search_emails result.

    Args:
        message_id: The Gmail message ID to archive. Get this from previous email results in the conversation.
    """
    return archive_email(message_id=message_id, credentials=_current_credentials.get())


def tool_get_full_email(message_id: str) -> dict:
    """Read the complete untruncated body of a specific email. Use this when the user wants
    to hear the full content of an email. Call read_emails or search_emails first to get
    the message_id, then use this tool with that ID.

    Args:
        message_id: The Gmail message ID to read in full. Get this from previous read_emails or search_emails results.
    """
    return get_full_email(message_id=message_id, credentials=_current_credentials.get())


def tool_read_calendar(date: str = "", calendar_id: str = "primary") -> dict:
    """Read events from the user's Google Calendar for a specific date.

    Args:
        date: Date in YYYY-MM-DD format. If not specified, defaults to today.
        calendar_id: Calendar ID. Use "primary" for the main calendar, or a specific ID from list_calendars. Default "primary".
    """
    return read_calendar(date=date if date else None, calendar_id=calendar_id, credentials=_current_credentials.get())


def tool_create_event(summary: str, date: str, start_time: str, end_time: str = "", description: str = "", reminder_minutes: int = 10, calendar_id: str = "primary") -> dict:
    """Create a new event on the user's Google Calendar with a phone notification reminder.

    Args:
        summary: Event title.
        date: Date in YYYY-MM-DD format.
        start_time: Start time in HH:MM 24-hour format.
        end_time: End time in HH:MM 24-hour format. Defaults to 1 hour after start if omitted.
        description: Optional event description.
        reminder_minutes: Minutes before the event to trigger a phone notification. Default 10 minutes.
        calendar_id: Calendar ID. Use "primary" for the main calendar, or a specific ID from list_calendars. Default "primary".
    """
    return create_event(
        summary=summary,
        date=date,
        start_time=start_time,
        end_time=end_time if end_time else None,
        description=description,
        reminder_minutes=reminder_minutes,
        calendar_id=calendar_id,
        credentials=_current_credentials.get(),
    )


def tool_delete_event(event_id: str, calendar_id: str = "primary") -> dict:
    """Delete (cancel) an event from Google Calendar. Use the event ID from a previous
    read_calendar result. Returns the event summary and time for confirmation.

    Args:
        event_id: The Google Calendar event ID to delete. Get this from previous calendar results.
        calendar_id: Calendar ID. Default "primary".
    """
    return delete_event(event_id=event_id, calendar_id=calendar_id, credentials=_current_credentials.get())


def tool_modify_event(event_id: str, summary: str = "", date: str = "", start_time: str = "", end_time: str = "", description: str = "", calendar_id: str = "primary") -> dict:
    """Modify an existing Google Calendar event. Only provide the fields that need to change —
    all other fields remain unchanged. When changing start_time, the event duration is preserved
    automatically unless end_time is also specified.

    Args:
        event_id: The Google Calendar event ID to modify. Get this from previous calendar results.
        summary: New event title. Leave empty to keep current.
        date: New date in YYYY-MM-DD format. Leave empty to keep current.
        start_time: New start time in HH:MM 24-hour format. Leave empty to keep current.
        end_time: New end time in HH:MM 24-hour format. Leave empty to keep current.
        description: New description. Leave empty to keep current.
        calendar_id: Calendar ID. Default "primary".
    """
    return modify_event(
        event_id=event_id,
        summary=summary if summary else None,
        date=date if date else None,
        start_time=start_time if start_time else None,
        end_time=end_time if end_time else None,
        description=description if description else None,
        calendar_id=calendar_id,
        credentials=_current_credentials.get(),
    )


def tool_list_calendars() -> dict:
    """List all Google Calendars the user has access to. Returns the name and ID for each calendar.
    Use this when the user mentions a specific calendar by name (e.g., "work calendar", "personal calendar")
    to find the correct calendar ID.
    """
    return list_calendars(credentials=_current_credentials.get())


def tool_save_memory(fact: str) -> dict:
    """Save an important fact or preference about the user to long-term memory.
    Use this when you learn key information that should be remembered across conversations.
    Examples: user preferences, habits, important contacts, work details.

    Args:
        fact: A concise statement about the user to remember. Example: "Prefers morning meetings"
    """
    return save_fact(fact, user_id=_current_user_id.get())


def tool_delete_memory(fact_keyword: str) -> dict:
    """Delete a remembered fact or preference from long-term memory. Use this when the user
    asks you to forget something or says a previously saved fact is no longer true.
    Matches any fact containing the keyword.

    Args:
        fact_keyword: A keyword or phrase to match against stored facts. All facts containing this keyword will be removed.
    """
    return delete_fact(fact_keyword, user_id=_current_user_id.get())


_web_search_client = None


def _get_web_search_client():
    """Get or create cached web search client."""
    global _web_search_client
    if _web_search_client is None:
        _web_search_client = genai_new.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _web_search_client


def tool_web_search(query: str) -> dict:
    """Search the web for real-time information. Use this for anything you don't already know:
    weather, news, sports scores, stock prices, store hours, recipes, travel info,
    general knowledge questions, or any factual question the user asks.

    Args:
        query: The search query. Be specific and concise.
    """
    try:
        client = _get_web_search_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            ),
        )
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


TOOL_FUNCTIONS = [
    tool_read_emails, tool_send_email, tool_search_emails, tool_archive_email, tool_get_full_email,
    tool_read_calendar, tool_create_event, tool_delete_event, tool_modify_event, tool_list_calendars,
    tool_save_memory, tool_delete_memory, tool_web_search,
]

# Map tool names to functions for dispatch.
# Gemini may call functions with or without the "tool_" prefix, so register both.
TOOL_REGISTRY = {
    "tool_read_emails": tool_read_emails,
    "tool_send_email": tool_send_email,
    "tool_search_emails": tool_search_emails,
    "tool_archive_email": tool_archive_email,
    "tool_get_full_email": tool_get_full_email,
    "tool_read_calendar": tool_read_calendar,
    "tool_create_event": tool_create_event,
    "tool_delete_event": tool_delete_event,
    "tool_modify_event": tool_modify_event,
    "tool_list_calendars": tool_list_calendars,
    "tool_save_memory": tool_save_memory,
    "tool_delete_memory": tool_delete_memory,
    "tool_web_search": tool_web_search,
    # Unprefixed aliases (Gemini sometimes omits the "tool_" prefix)
    "read_emails": tool_read_emails,
    "send_email": tool_send_email,
    "search_emails": tool_search_emails,
    "archive_email": tool_archive_email,
    "get_full_email": tool_get_full_email,
    "read_calendar": tool_read_calendar,
    "create_event": tool_create_event,
    "delete_event": tool_delete_event,
    "modify_event": tool_modify_event,
    "list_calendars": tool_list_calendars,
    "save_memory": tool_save_memory,
    "delete_memory": tool_delete_memory,
    "web_search": tool_web_search,
}


def _is_greeting(message: str) -> bool:
    """Check if a message is a simple greeting."""
    clean = message.lower().strip().rstrip("!.,")
    greetings = [
        "hi layla", "hey layla", "hello layla", "good morning layla",
        "good afternoon layla", "good evening layla",
        "hi", "hey", "hello", "good morning", "good afternoon", "good evening",
    ]
    return clean in greetings


def _fast_greeting(history: list, user_id: str | None = None) -> str:
    """Handle greeting without calling Gemini — always returns a response.

    Fetches unread emails and today's events directly, builds a natural
    voice-friendly summary. No Gemini round-trip needed.
    """
    from googleapiclient.discovery import build as build_service

    now = datetime.now()
    hour = now.hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    memory = load_memory(user_id)
    name = memory.get("user_name", "").split()[0] or "there"

    # If there's already conversation history, skip the briefing
    if history:
        return f"{greeting}, {name}! What can I help with?"

    # Get credentials for the current user
    creds = _current_credentials.get()
    if not creds:
        from auth import get_credentials
        creds = get_credentials()

    # First message — quick count of unread emails and upcoming events
    parts = [f"{greeting}, {name}!"]

    # Count NEW unread emails since last session (or last 24h if no previous session)
    try:
        gmail = build_service("gmail", "v1", credentials=creds)
        last_ts = memory.get("last_session_timestamp")
        if last_ts:
            # Convert ISO timestamp to Gmail after: format (YYYY/MM/DD)
            last_dt = datetime.fromisoformat(last_ts)
            gmail_date = last_dt.strftime("%Y/%m/%d")
            email_query = f"is:unread category:primary after:{gmail_date}"
        else:
            email_query = "is:unread category:primary newer_than:1d"
        unread_result = gmail.users().messages().list(
            userId="me", q=email_query, maxResults=100,
        ).execute()
        unread_count = unread_result.get("resultSizeEstimate", 0)
        if unread_count > 0:
            parts.append(f"You have {unread_count} new email{'s' if unread_count != 1 else ''}.")
    except Exception:
        pass

    # Count today's remaining events
    try:
        local_tz = now.astimezone().tzinfo
        now_iso = now.replace(tzinfo=local_tz).isoformat()
        end_of_day = now.replace(hour=23, minute=59, second=59, tzinfo=local_tz).isoformat()
        cal = build_service("calendar", "v3", credentials=creds)
        events_result = cal.events().list(
            calendarId="primary",
            timeMin=now_iso,
            timeMax=end_of_day,
            singleEvents=True,
        ).execute()
        event_count = len(events_result.get("items", []))
        if event_count > 0:
            parts.append(f"You have {event_count} event{'s' if event_count != 1 else ''} ahead today.")
    except Exception:
        pass

    # Wrap up
    if len(parts) == 1:
        parts.append("What can I help with?")
    else:
        parts.append("What would you like to do?")

    return " ".join(parts)


MAX_HISTORY_MESSAGES = 20
KEEP_RECENT = 6


def _compact_history(history: list):
    """Summarize older conversation history to preserve context while reducing tokens.

    When history exceeds MAX_HISTORY_MESSAGES, uses Gemini to create a concise
    summary of older messages (preserving key IDs, names, actions), then keeps
    the summary + the most recent messages. Falls back to simple trimming if
    summarization fails.
    """
    if len(history) <= MAX_HISTORY_MESSAGES:
        return

    old_messages = history[:-KEEP_RECENT]
    recent = history[-KEEP_RECENT:]

    # Build text representation of older messages for summarization
    lines = []
    for msg in old_messages:
        role = msg.role if hasattr(msg, 'role') else 'unknown'
        for part in msg.parts:
            if hasattr(part, 'thought') and part.thought:
                continue  # Skip model thinking parts
            if hasattr(part, 'text') and part.text:
                lines.append(f"{role}: {part.text}")
            elif part.function_call and part.function_call.name:
                lines.append(f"[Tool: {part.function_call.name}]")
            elif part.function_response and part.function_response.name:
                try:
                    result = dict(part.function_response.response)
                    result_str = json.dumps(result, default=str)
                    if len(result_str) > 300:
                        result_str = result_str[:300] + "..."
                    lines.append(f"[{part.function_response.name}: {result_str}]")
                except Exception:
                    lines.append(f"[{part.function_response.name}: completed]")

    if not lines:
        # No extractable text — fall back to simple trim
        history.clear()
        history.extend(recent)
        return

    conversation_text = "\n".join(lines)

    try:
        summary_model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = summary_model.generate_content(
            "Summarize this conversation concisely. Preserve: email message IDs, "
            "calendar event IDs, contact names/emails, and actions taken. "
            "Keep it under 5 sentences:\n\n" + conversation_text
        )
        summary = response.text.strip()
    except Exception:
        # Fallback: simple trim
        history.clear()
        history.extend(recent)
        return

    # Rebuild history: summary context pair + recent messages
    history.clear()
    history.append(genai.protos.Content(
        role="user",
        parts=[genai.protos.Part(text=f"[Earlier conversation summary: {summary}]")]
    ))
    history.append(genai.protos.Content(
        role="model",
        parts=[genai.protos.Part(text="Understood, I have the context from our earlier conversation.")]
    ))
    history.extend(recent)


def process_message(user_message: str, history: list, user_id: str | None = None) -> str:
    """Process a user message through the Gemini agent with tool calling.

    Args:
        user_message: The user's text input.
        history: Conversation history (list of Content objects).
        user_id: Authenticated user ID (multi-user mode) or None (legacy).

    Returns:
        The agent's text response.
    """
    # Set context variables for multi-user credential threading
    if user_id:
        from auth import get_credentials_for_user
        _current_credentials.set(get_credentials_for_user(user_id))
        _current_user_id.set(user_id)
    else:
        _current_credentials.set(None)
        _current_user_id.set(None)

    # Fast path for greetings — handles everything without Gemini
    if _is_greeting(user_message):
        greeting_reply = _fast_greeting(history, user_id=user_id)
        # Add the greeting exchange to history so Gemini has context for follow-up messages
        history.append(genai.protos.Content(
            role="user",
            parts=[genai.protos.Part(text=user_message)]
        ))
        history.append(genai.protos.Content(
            role="model",
            parts=[genai.protos.Part(text=greeting_reply)]
        ))
        return greeting_reply

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=_build_system_prompt(user_id),
        tools=TOOL_FUNCTIONS,
    )

    chat = model.start_chat(history=history)
    response = chat.send_message(user_message)

    # Agent loop: keep processing until we get a text response
    max_iterations = 10  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Collect ALL function calls from this response (Gemini may send multiple in parallel)
        candidate = response.candidates[0]
        function_calls = []

        for part in candidate.content.parts:
            if part.function_call and part.function_call.name:
                function_calls.append(part.function_call)

        if not function_calls:
            break  # No more tool calls — we have the final text response

        # Execute all function calls and collect results
        result_parts = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            # Gemini returns all numbers as floats — convert to int where needed
            for key, val in tool_args.items():
                if isinstance(val, float) and val == int(val):
                    tool_args[key] = int(val)

            print(f"  [Tool call] {tool_name}({tool_args})")

            try:
                tool_fn = TOOL_REGISTRY.get(tool_name)
                if not tool_fn:
                    result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    result = tool_fn(**tool_args)
            except Exception as e:
                result = {"error": str(e)}

            print(f"  [Tool result] {json.dumps(result, default=str)[:200]}")

            result_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": result},
                    )
                )
            )

        # Send ALL tool results back to Gemini in one message
        response = chat.send_message(
            genai.protos.Content(parts=result_parts)
        )

    # Extract the final text response
    reply_parts = []
    thought_text = ""
    for part in response.candidates[0].content.parts:
        has_text = hasattr(part, 'text') and part.text
        is_thought = hasattr(part, 'thought') and part.thought
        print(f"  [Debug] Part: text={repr(part.text) if has_text else 'None'}, thought={is_thought}")
        if has_text and not is_thought:
            reply_parts.append(part.text)
        elif has_text and is_thought:
            thought_text = part.text  # Save thought text as fallback

    reply = " ".join(reply_parts).strip() if reply_parts else ""

    # Guard against empty or meaningless responses (e.g., just "..." or whitespace)
    if not reply or reply.replace(".", "").strip() == "":
        reply = "I'm sorry, I couldn't process that request. Could you try again?"

    # Update history with the full conversation
    # The chat object maintains history internally, so we sync it back
    history.clear()
    history.extend(chat.history)

    # Compact history if it's getting too long (prevents unbounded token growth)
    _compact_history(history)

    return reply
