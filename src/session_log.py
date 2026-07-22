"""
Per-session logging.

Each call gets its own files rather than everything landing in one console
stream. LiveKit runs every job in its own process, so a handler attached to
the root logger inside a job only ever sees that one conversation.

Two files per session, in LOG_DIR (default ./logs):

    <session>.transcript.txt   readable turn-by-turn record, tool calls,
                               tool errors, and the final outcome
    <session>.debug.log        the full DEBUG stream, for when the
                               transcript is not enough

Plus one shared index, sessions.jsonl, with a line per session so you can
see at a glance which calls booked and which did not.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(os.environ.get("SADA_LOG_DIR", "logs"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def session_slug(room_name: str) -> str:
    """Filesystem-safe session id, sortable by time."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in room_name)
    return f"{stamp}_{safe[:48]}"


class SessionLogger:
    """Writes one readable transcript per conversation.

    Deliberately plain text: these files get read by a person trying to work
    out why a booking went wrong, so they are laid out for scanning rather
    than for parsing.
    """

    def __init__(self, slug: str, room: str, participant: str = "") -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.slug = slug
        self.room = room
        self.participant = participant
        self.started = time.time()
        self.path = LOG_DIR / f"{slug}.transcript.txt"
        self._fh = self.path.open("a", encoding="utf-8")
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.booking_ref: str | None = None

        self._write_raw(
            f"SADA hotel agent — session transcript\n"
            f"  room        {room}\n"
            f"  participant {participant or 'unknown'}\n"
            f"  started     {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"{'-' * 72}\n"
        )

    # ── writing ──────────────────────────────────────────────────────

    def _write_raw(self, text: str) -> None:
        self._fh.write(text)
        self._fh.flush()  # flush every line: a crashed session still leaves a usable file

    def _line(self, marker: str, body: str) -> None:
        self._write_raw(f"{_now()}  {marker}  {body}\n")

    def turn(self, role: str, text: str) -> None:
        """A committed conversation turn."""
        self.turns += 1
        marker = "USER " if role == "user" else "SADA "
        self._line(marker, text.replace("\n", " ").strip())

    def tool(self, tool_name: str, /, **params) -> None:
        """A tool invocation, with its arguments.

        tool_name is positional-only so that a tool can log a parameter
        genuinely called "name" without colliding with it.
        """
        self.tool_calls += 1
        args = ", ".join(f"{k}={v!r}" for k, v in params.items())
        self._line("  >>>", f"{tool_name}({args})")

    def tool_error(self, tool_name: str, /, message: str) -> None:
        """A tool refusing to proceed. These are the interesting ones."""
        self.tool_errors += 1
        self._line("  !!!", f"{tool_name} REJECTED: {message.splitlines()[0]}")

    def outcome(self, text: str, reference: str | None = None) -> None:
        if reference:
            self.booking_ref = reference
        self._line("  ***", text)

    def note(self, text: str) -> None:
        self._line("  ...", text)

    def timing(self, **parts) -> None:
        """Record model latencies for one turn.

        Written as its own line so the cost of a slow reply can be attributed
        to the language model or to speech synthesis, rather than guessed at
        from the gap between transcript lines.
        """
        shown = ", ".join(f"{k} {v}" for k, v in parts.items() if v is not None)
        if shown:
            self._line("  ---", shown)

    # ── closing ──────────────────────────────────────────────────────

    def close(self, reason: str = "") -> None:
        duration = time.time() - self.started
        mins, secs = divmod(int(duration), 60)
        summary = (
            f"{'-' * 72}\n"
            f"  ended     {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
            f"  ({mins}m {secs}s)\n"
            f"  turns     {self.turns}\n"
            f"  tools     {self.tool_calls} called, {self.tool_errors} rejected\n"
            f"  booking   {self.booking_ref or 'none completed'}\n"
        )
        if reason:
            summary += f"  reason    {reason}\n"
        self._write_raw(summary)
        self._fh.close()

        # One line in the shared index, so a directory of sessions can be
        # skimmed without opening each transcript.
        index = LOG_DIR / "sessions.jsonl"
        with index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "session": self.slug,
                "room": self.room,
                "participant": self.participant,
                "started": datetime.fromtimestamp(self.started, timezone.utc).isoformat(timespec="seconds"),
                "duration_s": round(duration, 1),
                "turns": self.turns,
                "tool_calls": self.tool_calls,
                "tool_errors": self.tool_errors,
                "booking": self.booking_ref,
                "reason": reason,
            }) + "\n")


def attach_debug_log(slug: str) -> logging.Handler:
    """Send the full DEBUG stream for this job to its own file.

    Safe because each job runs in a separate process — this handler will
    only ever see records from one conversation.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / f"{slug}.debug.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)
    return handler


def detach_debug_log(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
