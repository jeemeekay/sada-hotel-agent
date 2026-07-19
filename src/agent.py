"""
SADA Hotel Booking Voice Agent

A LiveKit voice agent that helps callers search for and book hotels by voice.
All models run through LiveKit Inference, so only LIVEKIT_* credentials
are required — no Deepgram, Cartesia, OpenAI, or Anthropic keys.

Pipeline:
    STT  → deepgram/nova-3
    LLM  → google/gemini-2.5-flash
    TTS  → cartesia/sonic-3

Hotel data comes from mock mode (default) or the Hotelbeds API.

Run:
    uv run src/agent.py dev       # dev mode, connects to LiveKit Cloud
    uv run src/agent.py console   # terminal-only, no room needed
"""

import logging
from datetime import date, datetime

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
)
from livekit.agents.llm import ToolError

from hotel_client import (
    resolve_city_code,
    search_hotels,
    search_offers,
    book_hotel,
)
from voice_utils import (
    letters_to_word,
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
    return f"""You are SADA, an AI hotel booking assistant. You speak with \
callers over the phone and help them find and reserve hotel rooms.

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

        city_code = resolve_city_code(city)
        if not city_code:
            raise ToolError(
                f"'{city}' is not a supported city. Ask the caller for a major "
                "city such as Dubai, Abu Dhabi, London, Paris, or New York."
            )

        hotels = search_hotels(city_code, max_results=6)
        if not hotels:
            raise ToolError(
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
            raise ToolError(
                "The caller has not given you dates yet. Ask which nights "
                "they want to stay, then call this again. Do not invent dates."
            )

        ids = [h.strip() for h in hotel_ids.split(",") if h.strip()]
        if not ids:
            raise ToolError("No hotel id was provided. Search for hotels first.")

        # Validate format before comparing. A plain string comparison would
        # silently misreport a malformed date as a date-ordering problem.
        try:
            ci = datetime.strptime(check_in, "%Y-%m-%d").date()
            co = datetime.strptime(check_out, "%Y-%m-%d").date()
        except ValueError:
            raise ToolError(
                "Dates must be in YYYY-MM-DD format. Convert the caller's "
                f"dates and call again. Received check_in={check_in!r}, "
                f"check_out={check_out!r}."
            )

        if co <= ci:
            raise ToolError(
                "The check-out date must be after the check-in date. Ask the "
                "caller to confirm their dates."
            )

        if ci < date.today():
            raise ToolError(
                "That check-in date is in the past. Ask the caller which "
                "upcoming dates they mean."
            )

        logger.info("Fetching offers for %s, %s to %s", ids, check_in, check_out)

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
            raise ToolError(str(e))
        except Exception:
            logger.exception("Offer search failed")
            raise ToolError(
                "The booking system did not respond. Ask the caller if they "
                "would like you to try again."
            )

        if not offers:
            raise ToolError(
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
    ) -> str:
        """Parse an email address the caller has dictated, and get a script
        to read it back. Use this every time the caller gives or corrects
        their email. Pass exactly what you heard, including words like "at",
        "dot" and number words — this tool converts them.

        Args:
            spoken_email: What the caller said, e.g.
                "k a y o d e at o four j dot co dot u k".
        """
        address, problem = parse_spoken_email(spoken_email)

        if problem:
            raise ToolError(problem)

        self._email = address
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
        if offer_id not in self._offers:
            raise ToolError(
                "That offer is not one of the options presented. Call "
                "get_hotel_offers again and confirm the choice with the caller."
            )

        for value, field in ((first_name, "first name"), (last_name, "surname")):
            ok, why = validate_name(value, field)
            if not ok:
                raise ToolError(why)

        # Prefer the address already parsed and confirmed by confirm_email.
        # Re-parsing a spoken string here is how a mangled address slips in.
        email_clean = self._email or parse_spoken_email(email)[0]
        ok, why = validate_email(email_clean)
        if not ok:
            raise ToolError(
                f"{why} Use confirm_email to capture the address first."
            )

        phone_clean = normalise_phone(phone)
        ok, why = validate_phone(phone_clean)
        if not ok:
            raise ToolError(why)

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
            "Confirm these in THREE separate turns. Ask one question, wait "
            "for the answer, then ask the next. Do not bundle them: a caller "
            "asked to confirm three things at once will say yes to all of "
            "them without checking each one.\n"
            f"Turn 1, the stay: {offer['hotel_name']}, "
            f"{offer['check_in']} to {offer['check_out']}, "
            f"{offer['currency']} {offer['price_total']} total. "
            "Ask if that is right.\n"
            f"Turn 2, the phone number ONLY, digit by digit: "
            f"{' '.join(phone_clean)}. Ask if that is right. Phone numbers "
            "are the most common thing to get wrong, so give this its own "
            "question.\n"
            f"Turn 3, name and email: the name {first_name} {last_name}, and "
            f"the email read exactly as written here: "
            f"{spell_phonetically(email_clean)}. Ask if that is right.\n"
            "If anything is wrong, call prepare_booking again with the "
            "correction. Only when all three are confirmed, call "
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
            raise ToolError(
                "No booking has been staged. Call prepare_booking first and "
                "read the details back to the caller."
            )

        if not caller_confirmed:
            raise ToolError(
                "The caller has not confirmed yet. Read the staged details "
                "back and wait for them to say they are correct."
            )

        p = self._pending
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
            raise ToolError(
                "The booking could not be completed. Apologise and offer to "
                "try again."
            )

        if not result.get("success"):
            raise ToolError(
                f"The booking was declined: {result.get('error', 'unknown reason')}. "
                "Apologise and suggest a different room."
            )

        reference = result["booking_id"]
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

    session = AgentSession(
        # All three models run through LiveKit Inference. The only
        # credentials needed are LIVEKIT_URL / API_KEY / API_SECRET.
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # gpt-4.1-mini chosen over gemini-2.5-flash for time-to-first-token;
        # on a phone call the caller hears every extra second as dead air.
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        turn_handling=TurnHandlingOptions(
            interruption={
                # Phone lines are noisy; don't let background sound
                # permanently cut the agent off mid-sentence.
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.0,
            },
            # Start generating before the caller fully stops speaking.
            preemptive_generation={"enabled": True, "max_retries": 3},
        ),
        # Give the client 3s to calibrate echo cancellation before the
        # agent can be interrupted.
        aec_warmup_duration=3.0,
        # Tool results contain ids and punctuation that must never be spoken.
        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    await session.start(
        agent=HotelBookingAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
