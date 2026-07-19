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

load_dotenv()

logger = logging.getLogger("sada-hotel-agent")


# ── System prompt ────────────────────────────────────────────────────

def build_instructions() -> str:
    """Build the system prompt with today's date injected at runtime."""
    today = date.today().isoformat()
    return f"""You are SADA, an AI hotel booking assistant. You speak with \
callers over the phone and help them find and reserve hotel rooms.

# Voice output rules (critical)
Your replies are converted to speech. Therefore:
- Never say IDs, codes, or anything in square brackets out loud. Hotel IDs \
and offer IDs are for your internal use only — the caller must never hear them.
- No markdown, asterisks, bullets, or emoji. Speak in plain sentences.
- Keep replies to one or two sentences. This is a conversation, not a document.
- Say prices naturally: "one hundred and eighty dollars a night", not "USD 180".
- Say dates naturally: "the twenty-fifth of July", not "2026-07-25".

# Your manner
Professional, warm, and efficient. Confirm details back to the caller before \
you act on them. Do not ramble.

# Today's date
Today is {today}. Use this to resolve relative dates like "next Friday" or \
"this weekend". Tool calls always require dates in YYYY-MM-DD format, so \
convert before calling.

# Booking flow
1. Greet the caller and ask how you can help.
2. Find out which city they want to stay in.
3. Find out their check-in and check-out dates.
4. Find out how many adults are travelling. Assume one if they don't say.
5. Call search_hotels_in_city to find hotels.
6. Read out three to five hotel names and let the caller pick one. Do not read \
the IDs.
7. Call get_hotel_offers with the chosen hotel's ID and the dates.
8. Describe the room and the price. Mention the cancellation policy briefly.
9. If they want to book, collect their first name, last name, email, and \
phone number. Read the email back to confirm you heard it correctly.
10. Call book_hotel_room. Then read out the booking reference slowly, \
character by character, since references are hard to hear over the phone.

# Rules
- Never invent a hotel name, a price, or an availability. Only state what the \
tools return.
- If a tool reports an error, tell the caller plainly and offer an alternative.
- Before booking, mention that this is a demonstration booking and no real \
charge will be made.
- If the caller wants something unrelated to hotels, politely say that hotel \
bookings are all you can help with."""


# ── Agent ────────────────────────────────────────────────────────────

class HotelBookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=build_instructions())
        # Per-session state. A new agent instance is created per session,
        # so these are safe as plain instance attributes.
        self._hotels: dict[str, dict] = {}
        self._offers: dict[str, dict] = {}

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
        adults: int = 1,
    ) -> str:
        """Get available rooms and prices for one or more hotels. Call this
        after the caller has chosen a hotel and given you their dates.

        Args:
            hotel_ids: One or more hotel id values from search_hotels_in_city,
                separated by commas if there is more than one.
            check_in: Check-in date, formatted as YYYY-MM-DD.
            check_out: Check-out date, formatted as YYYY-MM-DD.
            adults: Number of adults staying. Defaults to 1.
        """
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

    # ── Tool 3: Book a room ──────────────────────────────────────────

    @function_tool()
    async def book_hotel_room(
        self,
        context: RunContext,
        offer_id: str,
        first_name: str,
        last_name: str,
        email: str,
        phone: str,
    ) -> str:
        """Reserve a room. Only call this once the caller has explicitly
        confirmed they want to book, and you have all their contact details.

        Args:
            offer_id: The offer_id value from get_hotel_offers.
            first_name: The guest's first name.
            last_name: The guest's last name.
            email: The guest's email address.
            phone: The guest's phone number, including country code.
        """
        # Booking writes to an external system and cannot be rolled back,
        # so block interruptions until it completes.
        context.disallow_interruptions()

        if offer_id not in self._offers and self._offers:
            raise ToolError(
                "That offer is not one of the options presented. Call "
                "get_hotel_offers again and re-confirm the choice with the caller."
            )

        logger.info("Booking offer %s for %s %s", offer_id, first_name, last_name)

        try:
            result = book_hotel(
                offer_id=offer_id,
                guest_first_name=first_name,
                guest_last_name=last_name,
                guest_email=email,
                guest_phone=phone,
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

        offer = self._offers.get(offer_id, {})
        hotel = offer.get("hotel_name", "the hotel")

        return (
            f"Booking confirmed at {hotel}. "
            f"Reference: {result['booking_id']}. "
            f"Status: {result['status']}. "
            "Read the reference back slowly, one character at a time, then "
            "confirm a copy has been sent to the caller's email."
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
        llm=inference.LLM(model="google/gemini-2.5-flash"),
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
