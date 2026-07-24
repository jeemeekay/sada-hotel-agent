# SADA Hotel Booking Voice Agent

A voice AI agent that books hotel rooms through natural conversation. Built on [LiveKit](https://livekit.io) for real-time audio and [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) for speech-to-speech processing.

Try it: [sada.sh/sada-hotel-agent.html](https://sada.sh/sada-hotel-agent.html)

## How it works

You speak, SADA listens. Tell it a city, dates, and how many guests. It searches hotels, reads out options, and books the one you choose — collecting your name, email, and phone along the way. The entire interaction happens by voice.

Under the hood, a single OpenAI Realtime model handles speech recognition, reasoning, and voice synthesis in one step. No STT → LLM → TTS pipeline. Sub-second latency.

For the full system design, evolution from the original three-model pipeline, and the validation guards built from real call failures, see [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/jeemeekay/sada-hotel-agent.git
cd sada-hotel-agent

cp env.example .env
# Fill in: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY

uv sync
uv run src/agent.py dev
```

Open [sada.sh/sada-hotel-agent.html](https://sada.sh/sada-hotel-agent.html) or the [LiveKit Playground](https://agents-playground.livekit.io) and start talking.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [LiveKit Cloud](https://cloud.livekit.io) account (free tier available)
- [OpenAI](https://platform.openai.com) API key with Realtime API access

No hotel API key needed — the agent ships with realistic mock hotel data for 36 cities.

## Project structure

```
sada-hotel-agent/
├── src/
│   ├── agent.py           # Agent, tools, system prompt, session setup
│   ├── hotel_client.py    # Hotel search/book (mock data + Hotelbeds)
│   ├── voice_utils.py     # Phonetic spelling, email parsing, validation
│   └── session_log.py     # Per-session transcript and debug logging
├── web/
│   └── sada-hotel-agent.html  # Browser voice client
├── server/
│   └── token_server.py    # Standalone token server (alternative to Vercel)
├── deploy/
│   └── sada-agent.service # systemd unit for the agent
├── docs/
│   └── architecture.md    # System design, evolution, guard documentation
├── env.example
└── pyproject.toml
```

## Agent tools

| Tool | Purpose |
|---|---|
| `search_hotels_in_city` | Find hotels by city name (36 cities) |
| `get_hotel_offers` | Rooms, prices, cancellation policies |
| `confirm_email` | Parse spoken email ("kayode at o four j dot co dot uk") |
| `spell_back` | NATO phonetic script for any string |
| `prepare_booking` | Validate and stage details (does not book) |
| `book_hotel_room` | Commit the reservation (requires prior confirmation) |

## Logging

Every conversation writes a transcript to `logs/` (local) or `/var/log/sada/sessions/` (server):

```
15:16:30    >>>  search_hotels_in_city(city='Dubai')
15:17:01    >>>  prepare_booking(name='Kayode Olajide', phone='+44')
15:17:01    !!!  prepare_booking REJECTED: '+44' is only 2 digits...
```

See [docs/architecture.md](docs/architecture.md) for the full logging format and `jq` queries for filtering sessions.

## Deployment

The live deployment uses three services:

- **Vercel** — serves the page and mints LiveKit tokens (`sada-landing` repo)
- **Hetzner** — runs the agent process (`sada-agent.service`)
- **LiveKit Cloud** — routes audio between browser and agent

See [docs/architecture.md](docs/architecture.md) for the deployment topology and server setup.

## Part of SADA (صدى)

This agent is a vertical demo for [SADA](https://sada.sh) — AI voice agents the Middle East can trust.
