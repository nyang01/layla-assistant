"""Google Calendar API tools for reading, creating, modifying, and deleting events."""

from datetime import datetime, timedelta

from googleapiclient.discovery import build

from auth import get_credentials


_calendar_service = None


def _get_calendar_service(credentials=None):
    """Build Calendar service. Uses provided credentials (multi-user) or legacy singleton."""
    if credentials:
        return build("calendar", "v3", credentials=credentials)
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = build("calendar", "v3", credentials=get_credentials())
    return _calendar_service


def read_calendar(date: str | None = None, calendar_id: str = "primary", credentials=None) -> dict:
    """Read events from Google Calendar for a specific date.

    Args:
        date: Date string in YYYY-MM-DD format. Defaults to today.
        calendar_id: Calendar ID. Defaults to "primary".

    Returns:
        dict with status and list of events.
    """
    service = _get_calendar_service(credentials)

    local_tz = datetime.now().astimezone().tzinfo

    if date:
        target_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=local_tz)
    else:
        target_date = datetime.now().astimezone()

    # Start and end of the target day in local timezone
    time_min = target_date.replace(hour=0, minute=0, second=0).isoformat()
    time_max = target_date.replace(hour=23, minute=59, second=59).isoformat()

    results = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = []
    for event in results.get("items", []):
        start = event["start"].get("dateTime", event["start"].get("date", ""))
        end = event["end"].get("dateTime", event["end"].get("date", ""))
        events.append({
            "id": event["id"],
            "summary": event.get("summary", "No title"),
            "start": start,
            "end": end,
            "description": event.get("description", ""),
            "location": event.get("location", ""),
        })

    date_str = target_date.strftime("%A, %B %d, %Y")
    return {
        "status": "success",
        "date": date_str,
        "event_count": len(events),
        "events": events,
    }


def create_event(
    summary: str,
    date: str,
    start_time: str,
    end_time: str | None = None,
    description: str = "",
    reminder_minutes: int = 10,
    calendar_id: str = "primary",
    credentials=None,
) -> dict:
    """Create a new event on Google Calendar.

    Args:
        summary: Event title.
        date: Date in YYYY-MM-DD format.
        start_time: Start time in HH:MM format (24-hour).
        end_time: End time in HH:MM format. Defaults to 1 hour after start.
        description: Optional event description.
        reminder_minutes: Minutes before the event to send a notification. Default 10.
        calendar_id: Calendar ID. Defaults to "primary".

    Returns:
        dict with status and created event details.
    """
    service = _get_calendar_service(credentials)

    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")

    if end_time:
        end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
    else:
        end_dt = start_dt + timedelta(hours=1)

    # Get local timezone offset
    local_tz = datetime.now().astimezone().tzinfo
    start_dt = start_dt.replace(tzinfo=local_tz)
    end_dt = end_dt.replace(tzinfo=local_tz)

    event_body = {
        "summary": summary,
        "start": {"dateTime": start_dt.isoformat()},
        "end": {"dateTime": end_dt.isoformat()},
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": reminder_minutes},
            ],
        },
    }

    if description:
        event_body["description"] = description

    created = service.events().insert(
        calendarId=calendar_id,
        body=event_body,
    ).execute()

    return {
        "status": "success",
        "event_id": created["id"],
        "summary": created.get("summary", ""),
        "start": created["start"].get("dateTime", ""),
        "end": created["end"].get("dateTime", ""),
        "link": created.get("htmlLink", ""),
    }


def delete_event(event_id: str, calendar_id: str = "primary", credentials=None) -> dict:
    """Delete (cancel) an event from Google Calendar.

    Args:
        event_id: The Google Calendar event ID to delete.
        calendar_id: Calendar ID. Defaults to "primary".

    Returns:
        dict with status and details of the deleted event.
    """
    service = _get_calendar_service(credentials)

    # Fetch event details before deleting for confirmation
    event = service.events().get(
        calendarId=calendar_id,
        eventId=event_id,
    ).execute()

    summary = event.get("summary", "No title")
    start = event["start"].get("dateTime", event["start"].get("date", ""))

    service.events().delete(
        calendarId=calendar_id,
        eventId=event_id,
    ).execute()

    return {
        "status": "success",
        "deleted_event_id": event_id,
        "summary": summary,
        "start": start,
    }


def modify_event(
    event_id: str,
    summary: str | None = None,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    description: str | None = None,
    calendar_id: str = "primary",
    credentials=None,
) -> dict:
    """Modify an existing Google Calendar event. Only provided fields are updated.

    Args:
        event_id: The Google Calendar event ID to modify.
        summary: New event title. None to keep current.
        date: New date in YYYY-MM-DD format. None to keep current.
        start_time: New start time in HH:MM 24-hour format. None to keep current.
        end_time: New end time in HH:MM 24-hour format. None to keep current.
        description: New description. None to keep current.
        calendar_id: Calendar ID. Defaults to "primary".

    Returns:
        dict with status and updated event details.
    """
    service = _get_calendar_service(credentials)
    local_tz = datetime.now().astimezone().tzinfo

    patch_body = {}

    if summary is not None:
        patch_body["summary"] = summary

    if description is not None:
        patch_body["description"] = description

    # Handle time changes — need to fetch existing event if partial time info given
    if date is not None or start_time is not None or end_time is not None:
        existing = service.events().get(
            calendarId=calendar_id,
            eventId=event_id,
        ).execute()

        # Parse existing start/end
        existing_start_str = existing["start"].get("dateTime", "")
        existing_end_str = existing["end"].get("dateTime", "")

        if existing_start_str:
            existing_start = datetime.fromisoformat(existing_start_str)
        else:
            existing_start = datetime.now().astimezone()

        if existing_end_str:
            existing_end = datetime.fromisoformat(existing_end_str)
        else:
            existing_end = existing_start + timedelta(hours=1)

        # Determine new date (use existing if not provided)
        new_date = date if date else existing_start.strftime("%Y-%m-%d")

        # Determine new start time
        if start_time:
            new_start = datetime.strptime(f"{new_date} {start_time}", "%Y-%m-%d %H:%M")
        else:
            new_start = datetime.strptime(
                f"{new_date} {existing_start.strftime('%H:%M')}", "%Y-%m-%d %H:%M"
            )

        # Determine new end time
        if end_time:
            new_end = datetime.strptime(f"{new_date} {end_time}", "%Y-%m-%d %H:%M")
        else:
            # If start_time changed but end_time didn't, shift end to preserve duration
            if start_time:
                original_duration = existing_end - existing_start
                new_end = new_start + original_duration
            else:
                new_end = datetime.strptime(
                    f"{new_date} {existing_end.strftime('%H:%M')}", "%Y-%m-%d %H:%M"
                )

        new_start = new_start.replace(tzinfo=local_tz)
        new_end = new_end.replace(tzinfo=local_tz)

        patch_body["start"] = {"dateTime": new_start.isoformat()}
        patch_body["end"] = {"dateTime": new_end.isoformat()}

    if not patch_body:
        return {"status": "error", "message": "No fields to update were provided."}

    updated = service.events().patch(
        calendarId=calendar_id,
        eventId=event_id,
        body=patch_body,
    ).execute()

    return {
        "status": "success",
        "event_id": updated["id"],
        "summary": updated.get("summary", ""),
        "start": updated["start"].get("dateTime", ""),
        "end": updated["end"].get("dateTime", ""),
    }


def list_calendars(credentials=None) -> dict:
    """List all calendars accessible to the user.

    Returns:
        dict with status and list of calendars with name and ID.
    """
    service = _get_calendar_service(credentials)

    results = service.calendarList().list().execute()

    calendars = []
    for cal in results.get("items", []):
        calendars.append({
            "id": cal["id"],
            "name": cal.get("summaryOverride") or cal.get("summary", "Untitled"),
            "is_primary": cal.get("primary", False),
            "access_role": cal.get("accessRole", ""),
        })

    return {"status": "success", "calendar_count": len(calendars), "calendars": calendars}
