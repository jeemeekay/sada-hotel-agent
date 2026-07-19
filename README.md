# SADA Hotel Booking Voice Agent

A LiveKit voice agent that books hotel rooms by voice. Users call in (SIP or WebRTC), speak naturally, and the agent searches hotels, presents options, and completes reservations.

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
                          │  Hotel API     │
                          │  (mock or      │  → find hotels by city
                          │   Hotelbeds)   │  → get rooms & prices
                          │                │  → create reservation
                          └────────────────┘
```

## Conversation Flow

1. Agent greets the caller
2. Collects city, dates, and number of guests
3. Searches hotels → reads out options
4. Caller picks a hotel → agent fetches room offers with prices
5. Caller confirms → agent collects guest details (name, email, phone)
6. Agent books the room → reads back confirmation ID

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for:
  - **LiveKit Cloud** — [cloud.livekit.io](https://cloud.livekit.io)
  - **Anthropic** — [console.anthropic.com](https://console.anthropic.com)
  - **Deepgram** — [console.deepgram.com](https://console.deepgram.com)
  - **Cartesia** — [play.cartesia.ai](https://play.cartesia.ai)

No hotel API key needed for demo mode — the agent uses built-in mock hotel data (real hotel names, realistic prices) that works out of the box.

## Setup

```bash
git clone https://github.com/jeemeekay/sada-hotel-agent.git
cd sada-hotel-agent

cp env.example .env
# Fill in your LiveKit, Anthropic, Deepgram, and Cartesia keys

uv sync
```

## Run

```bash
uv run src/agent.py dev
```

Then open the [LiveKit Agents Playground](https://agents-playground.livekit.io) or connect via SIP.

Try: *"I'd like to book a hotel in Dubai for July 25th to the 28th."*

## Project Structure

```
sada-hotel-agent/
├── src/
│   ├── agent.py            # LiveKit agent with tool definitions
│   └── amadeus_client.py   # Hotel API (mock data + Hotelbeds support)
├── env.example             # Environment variable template
├── .gitignore
├── pyproject.toml
└── README.md
```

## Hotel API Modes

The agent supports two modes, set via `HOTEL_API_MODE` in `.env`:

| Mode | Description | API Key Needed? |
|---|---|---|
| `mock` (default) | Realistic demo data — real hotel names in Dubai, Abu Dhabi, London, Paris, NYC. Deterministic prices. Demo booking confirmations. | No |
| `hotelbeds` | Live hotel data via [Hotelbeds APItude API](https://developer.hotelbeds.com). Real availability, real bookings. | Yes |

### Why mock mode?

Amadeus Self-Service APIs were [decommissioned on July 17, 2026](https://www.phocuswire.com/amadeus-shut-down-self-service-apis-portal-developers). Mock mode lets you demo the full voice booking flow without any hotel API dependency. The architecture is provider-agnostic — swap in Hotelbeds, Expedia Rapid, or any other provider by implementing the same three functions: `search_hotels`, `search_offers`, `book_hotel`.

## How the Tools Work

| Tool | Purpose |
|---|---|
| `search_hotels_in_city` | Find hotels by city name (30+ cities supported) |
| `get_hotel_offers` | Get rooms, prices, cancellation policies for selected hotels |
| `book_hotel_room` | Create reservation with guest details → returns confirmation ID |

## Connecting via SIP (Phone Calls)

Works with the existing LiveKit SIP setup (Asterisk → LiveKit Cloud trunk). Dial extension 700 from the softphone to reach the agent.

## Part of SADA (صدى)

This agent is a vertical demo for [SADA](https://sada.sh), a UAE-compliant, Arabic-native AI voice agent platform for the Middle East.
