"""
SADA Hotel Booking Voice Agent

A LiveKit voice agent that helps callers search for and book hotels by voice.
Uses OpenAI's Realtime API for speech-to-speech — a single model that hears
and speaks directly, replacing the separate STT → LLM → TTS pipeline.

Requires: LIVEKIT_* credentials + OPENAI_API_KEY

Hotel data comes from mock mode (default) or the Hotelbeds API.

Run:
    uv run src/agent.py dev       # dev mode, connects to LiveKit Cloud
    uv run src/agent.py console   # terminal-only, no room needed
"""

import inspect
import logging
from datetime import date, datetime

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.agents.llm import ToolError
from livekit.plugins import openai

from hotel_client import (
    resolve_city_code,
    search_hotels,
    search_offers,
    book_hotel,
)
from session_log import (
    SessionLogger,
    attach_debug_log,
    detach_debug_log,
    session_slug,
)
from voice_utils import (
    letters_to_word,
    name_matches_email,
    normalise_phone,
    parse_spoken_email,
    parse_spoken_text,
    spell_phonetically,
    validate_email,
    validate_name,
    validate_phone,
)

load_dotenv()

logger = logging.getLogger("sada-hotel-agent")


# ── System prompt ────────────────────────────────────────────────────

def build_instructions() -> str:
    """Build the system prompt with today's date injected at runtime."""
    today = date.today().isoformat()
    return f"""You are SADA, an AI hotel booking assistant. You help \
callers find and reserve hotel rooms.

# Language
Always speak English by default. If the caller speaks to you in Arabic or \
any other language, switch to that language for the rest of the conversation. \
Never start in Arabic unless the caller does first.

# Voice output rules (critical)
Your replies are spoken aloud. Therefore:
- Never say IDs or codes out loud. Hotel IDs and offer IDs are internal.
- No markdown, asterisks, bullets, or emoji. Plain sentences only.
- Keep replies to one or two sentences unless reading details back.
- Say prices naturally: "one hundred and eighty dollars", not "USD 180".
- Say dates naturally: "the twenty-fifth of July", not "2026-07-25".

# Handling names, emails and phone numbers
These are the hardest things to get right over a phone line, and getting one
wrong ruins the booking. So:
- Wait for the caller to finish. A number like "plus four four" is the start
  of a sentence, not a phone number. Never act on a fragment.
- Always ask the caller to spell their first name and surname, and keep the
  letters they give you. Speech recognition mishears names constantly: it
  will report "Kyle" for someone who said Kayode. The letters they spell are
  the truth; what you think you heard is not.
- Never invent phonetic words. Call spell_back to get the correct script and
  read it back word for word. It is "O for oscar", never "O for osmium".
- For email addresses always use confirm_email, never spell_back. Pass it
  exactly what the caller said, including the words "at" and "dot" — it
  converts them. Do not try to assemble the address yourself.
- The word "at" is very often lost or misheard on a phone line, coming
  through as "app", "paw", or nothing at all. If the address has no at
  symbol, ask ONLY which part comes before the at. Never make the caller
  repeat the whole address — asking three times in a row is why a caller
  gives up.
- When something fails twice, change the question. Ask for one small piece
  instead of the whole thing again.
- Always ask for the country code with a phone number.
- Phone numbers often arrive split across two turns because the caller
  pauses. If a number looks short, do not use it — ask for the whole thing
  again and wait for them to finish. The tools will reject a short number.
- If the caller corrects a detail after booking, call prepare_booking and
  book_hotel_room again. This amends the existing reservation and keeps the
  same reference. It does not create a second booking.
- If the caller says a detail is wrong, ask only for the part that is wrong,
  not the whole thing again.

# Today's date
Today is {today}. Use it to resolve relative dates like "next Friday".
Tool calls need YYYY-MM-DD, so convert before calling.

# Booking flow, in order
1. Greet the caller and ask how you can help.
2. Ask which city.
3. Ask which nights they want to stay. You MUST ask. Never assume dates,
   never default to tonight or tomorrow.
4. Ask how many adults. Assume one only if they decline to say.
5. Call search_hotels_in_city.
6. Read out three to five hotel names. Never read IDs.
7. Once they choose, call get_hotel_offers.
8. Describe the room, the price, and the cancellation policy.
9. If they want it, collect first name, surname, email, and phone number.
   Ask for one at a time. Do not rush. Use confirm_email for the address.
10. Call prepare_booking. This does not book anything — it checks the details
    and gives you a script to read back.
11. Read the whole script back and ask if it is all correct.
12. Only when they confirm, call book_hotel_room with caller_confirmed set
    to true. Then read the reference back phonetically.

# Rules
- Never invent a hotel name, price, availability, or date. Only state what
  the tools return.
- If a tool returns an error, it is telling you how to recover. Follow it.
- Before booking, say this is a demonstration and no real charge is made.
- Never claim a confirmation email has been sent. None is.
- If the caller wants something unrelated to hotels, say politely that hotel
  bookings are all you can help with."""


# ── Agent ────────────────────────────────────────────────────────────

class HotelBookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=build_instructions())
        # Per-session state. A new agent instance is created per session,
        # so these are safe as plain instance attributes.
        self._hotels: dict[str, dict] = {}
        self._offers: dict[str, dict] = {}
        # Booking is a two-step commit. prepare_booking() validates and stages
        # the details here; book_hotel_room() can only act on what is staged.
        # This makes it structurally impossible to book from a half-finished
        # turn, because the tool that writes takes no detail arguments.
        self._pending: dict | None = None
        # Completed bookings, keyed by offer_id. Guards against booking the
        # same room twice: on a real call the caller spotted a wrong phone
        # number after confirming, and the agent silently created a second
        # reservation instead of amending the first.
        self._booked: dict[str, dict] = {}
        # Set by confirm_email once an address has been parsed and validated.
        self._email: str | None = None
        # Escalation counter: repeating the same question after it has failed
        # twice is what turns a booking into an eleven-minute call.
        self._email_attempts = 0
        # Name/email pairs already queried once. A caller whose address
        # genuinely differs from their name should not be trapped in a loop,
        # so the second attempt with the same values is allowed through.
        self._name_email_queried: set[tuple[str, str]] = set()
        # Set by the entrypoint. Tools write to it so the transcript shows
        # not just what was said, but what the agent actually did and which
        # guards fired.
        self.log: SessionLogger | None = None

    def _rejected(self, message: str) -> ToolError:
        """Record a guard firing, then hand back the error to raise.

        Rejections are the most informative lines in a transcript: they show
        which safeguard caught what, and how the agent was told to recover.
        """
        if self.log:
            caller = inspect.stack()[1].function
            self.log.tool_error(caller, message)
        return ToolError(message)

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Greet the caller, say you are SADA, and ask how you can help "
                "with their hotel booking. One short sentence."
            )
        )

    # ── Tool 1: Search hotels by city ────────────────────────────────

    @function_tool()
    async def search_hotels_in_city(
        self,
        context: RunContext,
        city: str,
    ) -> str:
        """Find hotels in a city. Call this once you know where the caller
        wants to stay.

        Args:
            city: The city name, for example "Dubai", "London", or "Paris".
        """
        logger.info("Searching hotels in %s", city)
        if self.log:
            self.log.tool("search_hotels_in_city", city=city)

        city_code = resolve_city_code(city)
        if not city_code:
            raise self._rejected(
                f"'{city}' is not a supported city. Ask the caller for a major "
                "city such as Dubai, Abu Dhabi, London, Paris, or New York."
            )

        hotels = search_hotels(city_code, max_results=6)
        if not hotels:
            raise self._rejected(
                f"No hotels are available in {city} right now. Ask the caller "
                "to try a different city."
            )

        self._hotels = {h["hotel_id"]: h for h in hotels}

        lines = [f"{len(hotels)} hotels found in {city}."]
        for h in hotels:
            distance = (
                f", {h['distance_km']} km from the centre"
                if h.get("distance_km")
                else ""
            )
            lines.append(f"- {h['name']}{distance}. id={h['hotel_id']}")
        lines.append(
            "Read the hotel names to the caller. Never read the id values aloud."
        )
        return "\n".join(lines)

    # ── Tool 2: Get room offers ──────────────────────────────────────

    @function_tool()
    async def get_hotel_offers(
        self,
        context: RunContext,
        hotel_ids: str,
        check_in: str,
        check_out: str,
        caller_stated_dates: bool,
        adults: int = 1,
    ) -> str:
        """Get available rooms and prices for one or more hotels. Call this
        after the caller has chosen a hotel AND told you their dates.

        Args:
            hotel_ids: One or more hotel id values from search_hotels_in_city,
                separated by commas if there is more than one.
            check_in: Check-in date, formatted as YYYY-MM-DD.
            check_out: Check-out date, formatted as YYYY-MM-DD.
            caller_stated_dates: True only if the caller actually told you
                these dates. False if you guessed, assumed, or defaulted them.
                Never choose dates on the caller's behalf.
            adults: Number of adults staying. Defaults to 1.
        """
        if not caller_stated_dates:
            raise self._rejected(
                "The caller has not given you dates yet. Ask which nights "
                "they want to stay, then call this again. Do not invent dates."
            )

        ids = [h.strip() for h in hotel_ids.split(",") if h.strip()]
        if not ids:
            raise self._rejected("No hotel id was provided. Search for hotels first.")

        # Validate format before comparing. A plain string comparison would
        # silently misreport a malformed date as a date-ordering problem.
        try:
            ci = datetime.strptime(check_in, "%Y-%m-%d").date()
            co = datetime.strptime(check_out, "%Y-%m-%d").date()
        except ValueError:
            raise self._rejected(
                "Dates must be in YYYY-MM-DD format. Convert the caller's "
                f"dates and call again. Received check_in={check_in!r}, "
                f"check_out={check_out!r}."
            )

        if co <= ci:
            raise self._rejected(
                "The check-out date must be after the check-in date. Ask the "
                "caller to confirm their dates."
            )

        if ci < date.today():
            raise self._rejected(
                "That check-in date is in the past. Ask the caller which "
                "upcoming dates they mean."
            )

        logger.info("Fetching offers for %s, %s to %s", ids, check_in, check_out)
        if self.log:
            self.log.tool("get_hotel_offers", hotels=ids, check_in=check_in,
                          check_out=check_out, adults=adults)

        try:
            offers = search_offers(
                hotel_ids=ids,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
            )
        except ValueError as e:
            # Malformed dates from the LLM — tell it exactly what went wrong
            # so it can retry with the correct format.
            raise self._rejected(str(e))
        except Exception:
            logger.exception("Offer search failed")
            raise self._rejected(
                "The booking system did not respond. Ask the caller if they "
                "would like you to try again."
            )

        if not offers:
            raise self._rejected(
                "No rooms are available at that hotel for those dates. Suggest "
                "different dates or another hotel from the list."
            )

        self._offers = {o["offer_id"]: o for o in offers}

        lines = [f"{len(offers)} room option(s) available."]
        for o in offers:
            policy = (
                f"free cancellation until {o['cancellation_deadline']}"
                if o.get("cancellation_deadline") not in (None, "", "N/A")
                else "non-refundable"
            )
            nightly = (
                f", {o['currency']} {o['price_per_night']} per night"
                if o.get("price_per_night")
                else ""
            )
            lines.append(
                f"- {o['hotel_name']}, {o['room_description']}, "
                f"{o['bed_type'].lower()} bed. "
                f"{o['currency']} {o['price_total']} total{nightly}. "
                f"Policy: {policy}. offer_id={o['offer_id']}"
            )
        lines.append(
            "Describe these to the caller. Never read the offer_id values aloud."
        )
        return "\n".join(lines)

    # ── Tool: phonetic script ────────────────────────────────────────

    @function_tool()
    async def spell_back(
        self,
        context: RunContext,
        text: str,
    ) -> str:
        """Get the exact phonetic script for reading something back to the
        caller. Use this to confirm a name or any spelling. Never invent your
        own phonetic words — call this and read what it returns, word for word.

        For email addresses use confirm_email instead, which parses the
        spoken form properly.

        Args:
            text: What to spell out. May be a word ("Kayode") or dictated
                letters ("k a y o d e"); both are handled.
        """
        # Words like "four" and "dot" must become characters first. Spelling
        # the literal string reads "four" back as F-O-U-R, which is what
        # made an earlier call unusable.
        resolved = parse_spoken_text(text) if " " in text.strip() else text
        return (
            f"This is '{resolved}'. Read the following back exactly as "
            f"written, word for word:\n{spell_phonetically(resolved)}"
        )

    # ── Tool: email capture ──────────────────────────────────────────

    @function_tool()
    async def confirm_email(
        self,
        context: RunContext,
        spoken_email: str,
        caller_rejected_previous: bool = False,
    ) -> str:
        """Parse an email address the caller has dictated, and get a script
        to read it back. Use this every time the caller gives or corrects
        their email. Pass exactly what you heard, including words like "at",
        "dot" and number words — this tool converts them.

        Args:
            spoken_email: What the caller said, e.g.
                "k a y o d e at o four j dot co dot u k".
            caller_rejected_previous: Set True if the caller just said the
                address you read back was wrong. This matters: an address can
                parse perfectly and still be the wrong address.
        """
        if self.log:
            self.log.tool("confirm_email", heard=spoken_email,
                          rejected_previous=caller_rejected_previous)

        if caller_rejected_previous:
            self._email_attempts += 1

        address, problem = parse_spoken_email(spoken_email)

        if problem:
            self._email_attempts += 1
            # Spelling letter by letter degrades badly on a phone line. After
            # two failures, stop asking for letters and ask for the domain as
            # a whole word instead — "o4j.co.uk" survives where "o, four, j"
            # does not.
            if self._email_attempts >= 3:
                raise self._rejected(
                    f"{problem}\n\nThis has now failed "
                    f"{self._email_attempts} times, so change approach. Do "
                    "not ask for more letters. Ask the caller to say the part "
                    "after the at symbol as a whole word, the way they would "
                    "say it to a person — for example 'gmail dot com' or "
                    "'o four j dot co dot uk'. If that also fails, offer to "
                    "take the booking without an email."
                )
            raise self._rejected(problem)

        self._email = address
        if self.log:
            self.log.note(f"email resolved to {address}")

        if self._email_attempts >= 3:
            return (
                f"The address is {address}. Read this back exactly as "
                f"written, word for word:\n{spell_phonetically(address)}\n\n"
                f"This has been rejected {self._email_attempts} times. If it "
                "is wrong again, stop asking for letters — ask the caller to "
                "say the part after the at symbol as a whole word, the way "
                "they would to a person, for example 'gmail dot com'. If that "
                "fails too, offer to complete the booking without an email."
            )

        return (
            f"The address is {address}. Read this back exactly as written, "
            f"word for word, then ask if it is correct:\n"
            f"{spell_phonetically(address)}"
        )

    # ── Tool 3: Stage the booking and read it back ───────────────────

    @function_tool()
    async def prepare_booking(
        self,
        context: RunContext,
        offer_id: str,
        first_name_spelling: str,
        last_name_spelling: str,
        email: str,
        phone: str,
    ) -> str:
        """Validate the caller's details and stage the booking for
        confirmation. This does NOT book anything. Call it once you have all
        the details. It returns a script to read back to the caller.

        Names must be given as the letters the caller actually spelled out,
        not as what you thought you heard. Speech recognition mishears names
        constantly; the spelling is the only reliable source.

        Args:
            offer_id: The offer_id value from get_hotel_offers.
            first_name_spelling: The caller's first name as separated letters,
                exactly as they spelled it, e.g. "k a y o d e".
            last_name_spelling: The caller's surname as separated letters,
                e.g. "o l a j i d e".
            email: The guest's email address.
            phone: The guest's phone number including the country code.
        """
        # The confirmed letters are authoritative. The STT's guess at the
        # spoken name is not — it produced "Kyle" for a caller who had just
        # spelled out K-A-Y-O-D-E.
        first_name = letters_to_word(first_name_spelling)
        last_name = letters_to_word(last_name_spelling)
        if self.log:
            self.log.tool("prepare_booking", name=f"{first_name} {last_name}",
                          email=email, phone=phone)
        if offer_id not in self._offers:
            raise self._rejected(
                "That offer is not one of the options presented. Call "
                "get_hotel_offers again and confirm the choice with the caller."
            )

        for value, field in ((first_name, "first name"), (last_name, "surname")):
            ok, why = validate_name(value, field)
            if not ok:
                raise self._rejected(why)

        # Prefer the address already parsed and confirmed by confirm_email.
        # Re-parsing a spoken string here is how a mangled address slips in.
        email_clean = self._email or parse_spoken_email(email)[0]
        ok, why = validate_email(email_clean)
        if not ok:
            raise self._rejected(
                f"{why} Use confirm_email to capture the address first."
            )

        phone_clean = normalise_phone(phone)
        ok, why = validate_phone(phone_clean)
        if not ok:
            raise self._rejected(why)

        # A spelled name and a spoken email are two independent captures of
        # the same person. When they nearly agree, one of them was misheard —
        # and a booking went out as "Teyode Olajide" against kayode@... for
        # exactly this reason, with nothing to catch it.
        pair = (first_name.lower(), email_clean)
        ok, why = name_matches_email(first_name, email_clean)
        if not ok and pair not in self._name_email_queried:
            self._name_email_queried.add(pair)
            raise self._rejected(why)

        offer = self._offers[offer_id]
        self._pending = {
            "offer_id": offer_id,
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "email": email_clean,
            "phone": phone_clean,
            "hotel_name": offer["hotel_name"],
            "check_in": offer["check_in"],
            "check_out": offer["check_out"],
            "price_total": offer["price_total"],
            "currency": offer["currency"],
        }

        return (
            "Details staged. Nothing is booked yet.\n"
            "Confirm these in FOUR separate turns. Ask one question, wait "
            "for the answer, then ask the next. Never bundle two of them into "
            "one question: a caller asked to confirm a name and an email "
            "together will say yes to both, and a wrong name goes through "
            "unnoticed.\n"
            f"Turn 1, the stay: {offer['hotel_name']}, "
            f"{offer['check_in']} to {offer['check_out']}, "
            f"{offer['currency']} {offer['price_total']} total. "
            "Ask if that is right.\n"
            f"Turn 2, the phone number ONLY, digit by digit: "
            f"{' '.join(phone_clean)}. Ask if that is right. Phone numbers "
            "are the most common thing to get wrong, so give this its own "
            "question.\n"
            f"Turn 3, the NAME ONLY, spelled out: "
            f"{spell_phonetically(first_name)}, then "
            f"{spell_phonetically(last_name)}. Ask if that is right. Spell it "
            "rather than saying it — a misheard letter in a surname is "
            "invisible when the name is merely spoken.\n"
            f"Turn 4, the EMAIL ONLY, read exactly as written here: "
            f"{spell_phonetically(email_clean)}. Ask if that is right.\n"
            "If anything is wrong, call prepare_booking again with the "
            "correction. Only when all four are confirmed, call "
            "book_hotel_room."
        )

    # ── Tool 4: Commit the booking ───────────────────────────────────

    @function_tool()
    async def book_hotel_room(
        self,
        context: RunContext,
        caller_confirmed: bool,
    ) -> str:
        """Complete the reservation using the details staged by
        prepare_booking. Only call this after you have read the details back
        and the caller has explicitly said they are correct.

        Args:
            caller_confirmed: True only if the caller has just confirmed, in
                their own words, that the details you read back are correct.
        """
        if self._pending is None:
            raise self._rejected(
                "No booking has been staged. Call prepare_booking first and "
                "read the details back to the caller."
            )

        if not caller_confirmed:
            raise self._rejected(
                "The caller has not confirmed yet. Read the staged details "
                "back and wait for them to say they are correct."
            )

        p = self._pending
        if self.log:
            self.log.tool("book_hotel_room", name=f"{p['first_name']} {p['last_name']}",
                          email=p["email"], phone=p["phone"], hotel=p["hotel_name"])

        # Safe to block interruptions now: everything was validated and
        # confirmed in earlier turns, so this call is short and committed.
        context.disallow_interruptions()

        # If this room is already booked, the caller is correcting something,
        # not booking again. Amend in place and keep the original reference.
        existing = self._booked.get(p["offer_id"])
        if existing:
            changed = [
                field
                for field in ("first_name", "last_name", "email", "phone")
                if existing["details"].get(field) != p.get(field)
            ]
            existing["details"] = {k: p[k] for k in p}
            self._pending = None

            if not changed:
                return (
                    f"This room is already booked under reference "
                    f"{existing['reference']}. Nothing has changed, so no "
                    "second reservation was made. Reassure the caller they "
                    "have one booking, not two, and read the reference back "
                    "using spell_back."
                )

            if self.log:
                self.log.outcome(
                    f"AMENDED {existing['reference']} — changed: {', '.join(changed)}",
                    reference=existing["reference"],
                )
            return (
                f"Booking {existing['reference']} has been AMENDED, not "
                f"duplicated. Updated: {', '.join(changed)}. Tell the caller "
                "their existing booking has been corrected and the reference "
                "is unchanged, then read it back using spell_back."
            )

        logger.info("Booking %s for %s %s", p["offer_id"], p["first_name"], p["last_name"])

        try:
            result = book_hotel(
                offer_id=p["offer_id"],
                guest_first_name=p["first_name"],
                guest_last_name=p["last_name"],
                guest_email=p["email"],
                guest_phone=p["phone"],
            )
        except Exception:
            logger.exception("Booking failed")
            raise self._rejected(
                "The booking could not be completed. Apologise and offer to "
                "try again."
            )

        if not result.get("success"):
            raise self._rejected(
                f"The booking was declined: {result.get('error', 'unknown reason')}. "
                "Apologise and suggest a different room."
            )

        reference = result["booking_id"]
        if self.log:
            self.log.outcome(
                f"BOOKED {p['hotel_name']} for {p['first_name']} {p['last_name']} "
                f"({p['email']}, {p['phone']})",
                reference=reference,
            )
        self._booked[p["offer_id"]] = {
            "reference": reference,
            "details": {k: p[k] for k in p},
        }
        self._pending = None

        return (
            f"Booked at {p['hotel_name']}.\n"
            f"Reference, read it back exactly like this: "
            f"{spell_phonetically(reference)}\n"
            "Do not claim a confirmation email has been sent — none is sent "
            "in this demonstration. Tell the caller to write the reference "
            "down instead."
        )


# ── Agent server ─────────────────────────────────────────────────────

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # One set of log files per conversation. Each job is its own process, so
    # the root-logger handler below only ever sees this session.
    slug = session_slug(ctx.room.name)
    debug_handler = attach_debug_log(slug)
    agent = HotelBookingAgent()
    agent.log = SessionLogger(slug, room=ctx.room.name)
    logger.info("session transcript: %s", agent.log.path)

    session = AgentSession(
        # OpenAI Realtime API: a single model that hears and speaks directly.
        # Replaces the separate STT → LLM → TTS pipeline, cutting latency
        # from 15-26s to sub-second, and bypassing LiveKit Inference entirely
        # (which had exhausted its free tier).
        llm=openai.realtime.RealtimeModel(
            model="gpt-realtime",
            voice="ash",
            temperature=0.7,
        ),
    )

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        text = getattr(item, "text_content", None) or ""
        role = getattr(item, "role", "")
        if text and role in ("user", "assistant"):
            agent.log.turn(role, text)

    @session.on("metrics_collected")
    def _on_metrics(event) -> None:
        # Field names vary across metric types and releases, so read
        # defensively — a missing attribute must never break a call.
        m = getattr(event, "metrics", event)
        parts = {}
        ttft = getattr(m, "ttft", None)
        if ttft and ttft > 0:
            parts["llm first token"] = f"{ttft:.2f}s"
        ttfb = getattr(m, "ttfb", None)
        if ttfb and ttfb > 0:
            parts["tts first byte"] = f"{ttfb:.2f}s"
        eou = getattr(m, "end_of_utterance_delay", None)
        if eou and eou > 0:
            parts["end of turn"] = f"{eou:.2f}s"
        if parts:
            agent.log.timing(**parts)

    @session.on("error")
    def _on_error(event) -> None:
        agent.log.note(f"SESSION ERROR: {event}")
        logger.error("session error: %s", event)

    async def _shutdown() -> None:
        agent.log.close(reason="session ended")
        detach_debug_log(debug_handler)

    ctx.add_shutdown_callback(_shutdown)

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
