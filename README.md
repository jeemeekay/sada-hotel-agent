# SADA Hotel Booking Voice Agent

A LiveKit voice agent that books hotel rooms via the **Amadeus Self-Service API**. Users call in (SIP or WebRTC), speak naturally, and the agent searches hotels, presents options, and completes reservations — all by voice.

## Architecture

```
Caller → LiveKit (WebRTC/SIP) → Agent
                                  │
                          ┌───────┴────────┐
                          │   STT          │  Deepgram Nova-3
                          │   LLM          │  Anthropic Claude
                          │   TTS          │  Cartesia Sonic
                          └───────┬────────┘
                                  │
                          ┌───────┴────────┐
                          │  Amadeus API   │
                          │  Hotel List    │  → find hotels by city
                          │  Hotel Search  │  → get rooms & prices
                          │  Hotel Booking │  → create reservation
                          └────────────────┘
```

## Conversation Flow

1. Agent greets the caller
2. Collects city, dates, and number of guests
3. Searches hotels via Amadeus → reads out options
4. Caller picks a hotel → agent fetches room offers with prices
5. Caller confirms → agent collects guest details (name, email, phone)
6. Agent books the room → reads back confirmation ID

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for:
  - **LiveKit Cloud** — [cloud.livekit.io](https://cloud.livekit.io)
  - **Amadeus Self-Service** — [developers.amadeus.com](https://developers.amadeus.com) (free account)
  - **Anthropic** — [console.anthropic.com](https://console.anthropic.com)
  - **Deepgram** — [console.deepgram.com](https://console.deepgram.com)
  - **Cartesia** — [play.cartesia.ai](https://play.cartesia.ai)

## Setup

```bash
# Clone the repo
git clone https://github.com/jeemeekay/sada-hotel-agent.git
cd sada-hotel-agent

# Copy env template and fill in your API keys
cp .env.example .env

# Install dependencies
uv sync
# or: pip install -e .
```

## Get Your Amadeus API Keys

1. Go to [developers.amadeus.com](https://developers.amadeus.com)
2. Create a free account
3. Go to **My Self-Service Workspace** → **Create a new app**
4. Copy the **API Key** and **API Secret**
5. Paste them into your `.env` file as `AMADEUS_CLIENT_ID` and `AMADEUS_CLIENT_SECRET`

The free test environment gives you limited API calls per month with sandbox hotel data — enough to build and demo.

## Run

```bash
# Development mode (connects to LiveKit Cloud)
uv run src/agent.py dev
```

Then open the [LiveKit Agents Playground](https://agents-playground.livekit.io) or connect via SIP to talk to the agent.

## Project Structure

```
sada-hotel-agent/
├── src/
│   ├── agent.py            # LiveKit agent with tool definitions
│   └── amadeus_client.py   # Amadeus API wrapper (search, offers, book)
├── .env.example            # Environment variable template
├── .gitignore
├── pyproject.toml          # Dependencies and project metadata
└── README.md
```

## How the Tools Work

The agent has three tools that the LLM can call during conversation:

| Tool | Amadeus API | Purpose |
|---|---|---|
| `search_hotels_in_city` | Hotel List API | Find hotels by city name |
| `get_hotel_offers` | Hotel Search API v3 | Get rooms, prices, cancellation policies |
| `book_hotel_room` | Hotel Booking API v2 | Create reservation with guest details |

The LLM decides when to call each tool based on the conversation. For example, when the caller says "I need a hotel in Dubai for July 25 to 28", the LLM calls `search_hotels_in_city("Dubai")` then `get_hotel_offers(hotel_ids, "2026-07-25", "2026-07-28")`.

## Connecting via SIP (Phone Calls)

If you have LiveKit SIP configured (trunk + dispatch rule), the agent works over phone calls too. See the [SADA project docs](https://github.com/jeemeekay/sada-hotel-agent) for SIP setup with Asterisk or direct LiveKit SIP trunking.

## Sandbox vs Production

The Amadeus client is configured in **test mode** (`hostname="test"`). This means:
- Hotel data is synthetic/limited
- Bookings are not real — no charges apply
- Ideal for development and demos

To go live, change `hostname="test"` to `hostname="production"` in `src/amadeus_client.py` and ensure your Amadeus account is approved for production access.

## Part of SADA (صدى)

This agent is a vertical demo for [SADA](https://sada.sh), a UAE-compliant, Arabic-native AI voice agent platform. The same architecture powers agents for healthcare, real estate, financial services, and government across the GCC.
