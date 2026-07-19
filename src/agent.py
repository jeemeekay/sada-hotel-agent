"""
SADA Hotel Booking Voice Agent

A LiveKit voice agent that helps users search for and book hotels.
Uses Claude as the LLM, Deepgram for STT, and Cartesia for TTS.
Hotel data comes from mock mode (default) or Hotelbeds API.

Run:
    uv run src/agent.py dev
"""

import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    RunContext,
    cli,
    function_tool,
)
from livekit.agents.llm import ToolError
from livekit.plugins import anthropic, cartesia, deepgram

from amadeus_client import (
    resolve_city_code,
    search_hotels,
    search_offers,
    book_hotel,
)

load_dotenv()

logger = logging.getLogger("sada-hotel-agent")

# ── System prompt ────────────────────────────────────────────────────

AGENT_INSTRUCTIONS = """You are SADA, an AI hotel booking assistant. You help \
callers find and reserve hotel rooms by voice.

## Your personality
- Professional yet warm and conversational
- Efficient — don't ramble, get to the point
- Confirm details back to the caller before acting

## Booking flow
1. **Greet** the caller and ask how you can help.
2. **Collect destination**: Ask which city they want to stay in.
3. **Collect dates**: Ask for check-in and check-out dates. Dates must be in \
YYYY-MM-DD format when calling tools — convert conversational dates like \
"next Friday" or "July 25th" to this format. Today is 2026-07-19.
4. **Collect guests**: Ask how many guests (default to 1 adult if not specified).
5. **Search hotels**: Use the `search_hotels_in_city` tool to find hotels.
6. **Present options**: Read out the top 3-5 hotel names and let the caller choose.
7. **Get offers**: Use `get_hotel_offers` with the chosen hotel(s) to fetch \
available rooms and prices.
8. **Present offers**: Read prices, room types, and cancellation policies. \
Keep it concise — e.g. "The Marriott has a deluxe room at $180 per night \
with free cancellation until July 23rd."
9. **Confirm booking**: Once the caller picks an offer, collect their name, \
email, and phone number.
10. **Book**: Use `book_hotel_room` to complete the reservation. Read back \
the confirmation ID.

## Important rules
- Never invent hotel names or prices — only use data from tool calls.
- If a city isn't found, ask the caller to try a different city or spelling.
- If no hotels have availability, say so honestly.
- Amounts are always spoken naturally: "$180" not "one hundred and eighty \
dollars".
- When listing multiple hotels, number them: "Option 1, Option 2" etc.
- For the booking step, remind the caller this is a test/demo booking in \
sandbox mode — no real charges will apply.
- If the caller asks about something other than hotel booking, politely \
redirect them.
"""


# ── Agent class with tool definitions ────────────────────────────────

class HotelBookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=AGENT_INSTRUCTIONS)

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the caller warmly and ask how you can help them with hotel booking today. Keep it brief — one or two sentences."
        )

    # ── Tool 1: Search hotels by city ────────────────────────────────

    @function_tool()
    async def search_hotels_in_city(
        self,
        context: RunContext,
        city: str,
    ) -> str:
        """Search for hotels in a given city.

        Args:
            city: The city name to search hotels in, e.g. "Dubai", "London", "Paris".
        """
        city_code = resolve_city_code(city)
        if not city_code:
            raise ToolError(
                f"I couldn't find the city code for '{city}'. "
                "Try a major city name like Dubai, London, Paris, New York, etc."
            )

        hotels = search_hotels(city_code, max_results=8)
        if not hotels:
            raise ToolError(f"No hotels found in {city}. The Amadeus test data may not cover this city.")

        # Store hotels in session state for later reference
        context.userdata["hotels"] = {h["hotel_id"]: h for h in hotels}
        context.userdata["city"] = city

        lines = [f"Found {len(hotels)} hotels in {city}:\n"]
        for i, h in enumerate(hotels, 1):
            name = h["name"]
            dist = f" ({h['distance_km']} km from centre)" if h.get("distance_km") else ""
            lines.append(f"{i}. {name}{dist} [ID: {h['hotel_id']}]")

        return "\n".join(lines)

    # ── Tool 2: Get available offers for selected hotel(s) ───────────

    @function_tool()
    async def get_hotel_offers(
        self,
        context: RunContext,
        hotel_ids: str,
        check_in: str,
        check_out: str,
        adults: int = 1,
    ) -> str:
        """Get available room offers and prices for one or more hotels.

        Args:
            hotel_ids: Comma-separated Amadeus hotel IDs, e.g. "MCLONGHM,HSLONBAL".
            check_in: Check-in date in YYYY-MM-DD format.
            check_out: Check-out date in YYYY-MM-DD format.
            adults: Number of adult guests. Defaults to 1.
        """
        ids = [hid.strip() for hid in hotel_ids.split(",")]

        offers = search_offers(
            hotel_ids=ids,
            check_in=check_in,
            check_out=check_out,
            adults=adults,
        )

        if not offers:
            raise ToolError(
                "No available rooms found for those hotels and dates. "
                "Try different dates or other hotels from the list."
            )

        # Store offers for booking step
        context.userdata["offers"] = {o["offer_id"]: o for o in offers}

        lines = [f"Found {len(offers)} available room(s):\n"]
        for i, o in enumerate(offers, 1):
            cancel = (
                f"Free cancellation until {o['cancellation_deadline']}"
                if o["cancellation_deadline"] != "N/A"
                else "Non-refundable"
            )
            lines.append(
                f"{i}. {o['hotel_name']} — {o['room_description']}\n"
                f"   Price: {o['currency']} {o['price_total']} total\n"
                f"   Beds: {o['bed_type']} x{o['beds']}\n"
                f"   Policy: {cancel}\n"
                f"   [Offer ID: {o['offer_id']}]"
            )

        return "\n".join(lines)

    # ── Tool 3: Book a hotel room ────────────────────────────────────

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
        """Book a hotel room using a specific offer ID. Call this only after \
the caller has confirmed they want to proceed with the booking.

        Args:
            offer_id: The Amadeus offer ID from get_hotel_offers results.
            first_name: Guest's first name.
            last_name: Guest's last name.
            email: Guest's email address.
            phone: Guest's phone number including country code.
        """
        # Prevent interruptions — this is a mutating action
        context.disallow_interruptions()

        result = book_hotel(
            offer_id=offer_id,
            guest_first_name=first_name,
            guest_last_name=last_name,
            guest_email=email,
            guest_phone=phone,
        )

        if not result.get("success"):
            raise ToolError(
                f"Booking failed: {result.get('error', 'Unknown error')}. "
                "Please try again or choose a different offer."
            )

        return (
            f"Booking confirmed!\n"
            f"Booking ID: {result['booking_id']}\n"
            f"Provider Confirmation: {result['provider_confirmation']}\n"
            f"Status: {result['status']}"
        )


# ── Session entrypoint ───────────────────────────────────────────────

async def entrypoint(ctx):
    session = AgentSession()
    await session.start(
        agent=HotelBookingAgent(),
        room=ctx.room,
        # ── STT: Deepgram Nova-3 (multilingual) ─────────────
        stt=deepgram.STT(model="nova-3", language="en"),
        # ── LLM: Anthropic Claude ───────────────────────────
        llm=anthropic.LLM(model="claude-sonnet-4-20250514"),
        # ── TTS: Cartesia Sonic ─────────────────────────────
        tts=cartesia.TTS(
            model="sonic",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
    )


if __name__ == "__main__":
    cli.run_app(entrypoint)
