# Architecture

SADA Hotel Booking Agent — how the system works and how it got here.

## Current architecture (v3 — OpenAI Realtime)

A single speech-to-speech model replaces the three-stage pipeline. The
browser connects directly to LiveKit Cloud for audio transport; the agent
connects outbound from a Hetzner server. No inbound ports are open on the
server at all.

```
┌──────────────────────────────────────────────────────────────────────┐
│  BROWSER  (sada.sh/sada-hotel-agent.html)                           │
│                                                                      │
│  ┌─────────┐    fetch /api/token    ┌────────────────────┐           │
│  │  Page   │ ──────────────────────▶│  Vercel function   │           │
│  │  (HTML) │ ◀───────── JWT ────────│  (api/token.js)    │           │
│  └────┬────┘                        └────────────────────┘           │
│       │                                                              │
│       │ WebRTC (audio + data)                                        │
│       ▼                                                              │
│  ┌──────────────────────────────────────────┐                        │
│  │           LiveKit Cloud                  │                        │
│  │       (media routing only)               │                        │
│  │                                          │                        │
│  │   browser ◄──── room ────► agent         │                        │
│  └──────────────────┬───────────────────────┘                        │
│                     │                                                │
└─────────────────────┼────────────────────────────────────────────────┘
                      │
                      │ WebSocket (outbound from server)
                      │
┌─────────────────────┼────────────────────────────────────────────────┐
│  HETZNER SERVER     │  (ubuntu-4gb-hel1-1 / 46.62.222.72)           │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                        │
│  │  sada-agent (systemd)                    │                        │
│  │                                          │                        │
│  │  ┌────────────────────────────────────┐  │                        │
│  │  │  LiveKit AgentSession              │  │                        │
│  │  │                                    │  │                        │
│  │  │  ┌──────────────────────────────┐  │  │                        │
│  │  │  │  OpenAI Realtime API         │  │  │                        │
│  │  │  │  (gpt-realtime)              │  │  │                        │
│  │  │  │                              │  │  │                        │
│  │  │  │  audio in ──► reasoning ──►  │  │  │                        │
│  │  │  │              + tool calls    │  │  │                        │
│  │  │  │              ──► audio out   │  │  │                        │
│  │  │  └──────────────────────────────┘  │  │                        │
│  │  │                │                   │  │                        │
│  │  │  ┌─────────────▼────────────────┐  │  │                        │
│  │  │  │  Agent tools                 │  │  │                        │
│  │  │  │                              │  │  │                        │
│  │  │  │  search_hotels_in_city       │  │  │                        │
│  │  │  │  get_hotel_offers            │  │  │                        │
│  │  │  │  confirm_email               │  │  │                        │
│  │  │  │  spell_back                  │  │  │                        │
│  │  │  │  prepare_booking             │  │  │                        │
│  │  │  │  book_hotel_room             │  │  │                        │
│  │  │  └──────────────────────────────┘  │  │                        │
│  │  └────────────────────────────────────┘  │                        │
│  │                                          │                        │
│  │  ┌──────────────────────┐                │                        │
│  │  │  Hotel data          │                │                        │
│  │  │  (mock / Hotelbeds)  │                │                        │
│  │  └──────────────────────┘                │                        │
│  │                                          │                        │
│  │  ┌──────────────────────┐                │                        │
│  │  │  Session logs        │                │                        │
│  │  │  /var/log/sada/      │                │                        │
│  │  │  sessions/           │                │                        │
│  │  │   ├ *.transcript.txt │                │                        │
│  │  │   ├ *.debug.log      │                │                        │
│  │  │   └ sessions.jsonl   │                │                        │
│  │  └──────────────────────┘                │                        │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
│  No inbound ports. Agent connects outbound to LiveKit Cloud          │
│  and outbound to OpenAI. Server runs only the agent process.         │
└──────────────────────────────────────────────────────────────────────┘
```

### Key properties

- **No media touches the server.** Audio flows browser → LiveKit Cloud → OpenAI and back. The server only runs the agent logic and tool calls.
- **Each visitor gets an isolated room.** The Vercel function mints a fresh room name per token request. The agent auto-dispatches into any new room.
- **Concurrent calls are independent.** Each room spawns its own agent process with its own state. No shared mutable state between conversations.
- **Latency is sub-second.** OpenAI Realtime processes audio natively — no STT → LLM → TTS chain. Measured LLM first-token times of 0.29–0.70s.

### Credentials flow

```
Vercel env vars ──► api/token.js ──► JWT (room + identity)
                    (LIVEKIT_URL,     sent to browser,
                     API_KEY,         valid 30 minutes
                     API_SECRET)

Server .env     ──► agent.py
                    LIVEKIT_URL      → registers worker with LiveKit Cloud
                    LIVEKIT_API_KEY  → authenticates to LiveKit Cloud
                    LIVEKIT_API_SECRET
                    OPENAI_API_KEY   → authenticates to OpenAI Realtime API
```

---

## Previous architectures

### v1 — Three-model pipeline via LiveKit Inference (July 19–22)

The original design routed speech through three separate models, all proxied
through LiveKit Cloud's inference service.

```
Browser/SIP ──► LiveKit Cloud ──► Agent
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                     ▼              ▼              ▼
                  Deepgram       Gemini 2.5     Cartesia
                  Nova-3         Flash /        Sonic-3
                  (STT)          GPT-4.1-mini   (TTS)
                                 (LLM)
                     │              │              │
                     └──────── all via ────────────┘
                          LiveKit Inference
```

**What worked:** The agent logic, tools, and booking flow were proven across
multiple SIP calls. Contact validation, phonetic readback, and the two-step
booking commit were all developed and tested in this phase.

**What didn't:** Three sequential network hops produced 15–26 second gaps
between a caller finishing and hearing a response. The LiveKit free tier was
exhausted after approximately 40 minutes of total usage, leaving every
subsequent STT call returning HTTP 429 and making the agent unusable.

### v1.5 — Three-model pipeline via direct APIs (considered, not shipped)

To bypass the LiveKit free tier limits, the plan was to sign up for each
provider directly (Deepgram, OpenAI, Cartesia) and use their SDKs instead
of the inference proxy. Each provider's free tier is independent and more
generous.

**Why it was skipped:** Signing up for three separate services and managing
three API keys to solve a latency problem that was architectural (three
sequential hops) rather than provider-specific. OpenAI Realtime collapsed
the pipeline instead.

### v2 — SIP via Asterisk (July 19, parallel to v1)

The first working demo used a SIP softphone calling through an Asterisk PBX
on the same Hetzner server, which trunked into LiveKit Cloud.

```
Odemis Softphone ──► Asterisk (PBX) ──► LiveKit SIP Trunk ──► Agent
     (macOS)         (Hetzner VPS)       (LiveKit Cloud)
```

**What worked:** Proved the end-to-end voice pipeline including SIP call
setup, DTMF, and the full booking conversation.

**What didn't:** Phone-quality audio (8kHz narrowband) degraded STT accuracy
significantly. Spelled letters were misheard constantly ("K-A-Y-O-D-E" →
"a a y o d e" or "Kyle"). Email dictation required 6–15 minutes per address.
The spelling-related guards (phonetic readback, name reconstruction from
confirmed letters, spoken email parsing) were all developed to compensate
for this, but the root cause was the audio channel, not the agent.

### Browser vs SIP — what changed

Moving from SIP to WebRTC browser audio improved STT accuracy enough that
most of the contact-capture guards stopped firing. A booking that took
11 minutes over SIP took 4 minutes over the browser. The guards remain in
the code because they are still correct — a phone caller will still need
them — but the primary interface is now the web page.

---

## Booking flow

The conversation follows a structured flow enforced by tool definitions
and validation guards rather than prompt instructions alone.

```
                    ┌──────────────────┐
                    │   Greeting       │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Collect city   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Collect dates  │◄── caller_stated_dates
                    │                  │    guard prevents the
                    └────────┬─────────┘    LLM from inventing
                             │              dates
                    ┌────────▼─────────┐
                    │  search_hotels   │
                    │  _in_city        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  get_hotel       │
                    │  _offers         │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Collect contact │
                    │  details         │
                    │                  │
                    │  • name (spelled)│
                    │  • confirm_email │◄── spoken-form parser
                    │  • phone         │◄── country-code + length
                    │                  │    validation
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  prepare_booking │◄── name/email cross-check
                    │  (stage only)    │    phone trunk-0 removal
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  4-turn readback │
                    │  1. stay details │
                    │  2. phone        │
                    │  3. name         │
                    │  4. email        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  book_hotel_room │◄── requires staged pending
                    │  (commit)        │    + caller_confirmed
                    │                  │    duplicate detection
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Confirmation    │
                    │  reference       │
                    │  (phonetic)      │
                    └──────────────────┘
```

### Validation guards

Every guard was built in response to a real failure observed on a live call.

| Guard | What it prevents | Origin |
|---|---|---|
| `caller_stated_dates` | LLM inventing dates the caller never gave | Call 1: agent booked tomorrow with no dates asked |
| Phone country-code required | Booking with a domestic number that can't receive international calls | Call 3: `7517247059` accepted without `+44` |
| Phone national-digit length | Booking on a partial number from a fragmented turn | Call 3: `+4475172470` accepted (8 digits, UK needs 10) |
| Phone trunk-0 removal | `+4407517247059` stored instead of `+447517247059` | Call 4: domestic and international prefix combined |
| Name from confirmed spelling | STT mishearing overriding the letters the caller confirmed | Call 2: "Kyle" booked despite caller spelling K-A-Y-O-D-E |
| Name/email cross-check | Misheard name slipping past when the email is correct | Call 5: "Teyode" booked against kayode@... |
| Spoken email parser | `spell_phonetically` treating "four" as F-O-U-R | Call 4: phonetic readback was gibberish |
| Email escalation | Asking for the whole address repeatedly after it fails | Call 4: 6 identical attempts, caller gave up |
| Two-step booking commit | Booking fired on a partial turn mid-sentence | Call 1: `+44` booked as a complete phone number |
| Duplicate booking prevention | Correction after booking creating a second reservation | Call 3: two bookings for the same room |
| Confirmation split into 4 turns | Wrong detail hidden inside a bundled confirmation | Call 4: "Olajmde" approved alongside correct email |

---

## Per-session logging

Each conversation produces two files and one index entry.

```
/var/log/sada/sessions/
├── sessions.jsonl                                  ← one line per call
├── 20260724-151617_web-f2fc044c-fd4.transcript.txt ← readable record
└── 20260724-151617_web-f2fc044c-fd4.debug.log      ← full DEBUG stream
```

**Transcript** — the primary record. Shows turns, tool calls with arguments,
guard rejections with reasons, booking outcomes, and per-turn latency.

```
15:16:22  SADA   Hello, this is SADA. How can I help you with your hotel booking?
15:16:30  USER   I'd like a hotel in Dubai please
15:16:30    >>>  search_hotels_in_city(city='Dubai')
15:16:31    ---  llm first token 0.42s
15:16:31  SADA   I found several hotels in Dubai. Here are some options...
15:17:01    >>>  prepare_booking(name='Kayode Olajide', phone='+44')
15:17:01    !!!  prepare_booking REJECTED: '+44' is only 2 digits...
```

**Index** — one JSON line per session for filtering:

```bash
# Failed bookings
jq -c 'select(.booking == null)' sessions.jsonl

# Calls where guards fired repeatedly
jq -c 'select(.tool_errors > 2)' sessions.jsonl
```

---

## Deployment topology

```
┌────────────────────────────────────────────────────┐
│  Vercel  (sada-landing project)                    │
│                                                    │
│  sada.sh/                    → index.html (landing)│
│  sada.sh/sada-hotel-agent.html → voice agent page  │
│  sada.sh/api/token           → token.js (function) │
│                                                    │
│  Env vars: LIVEKIT_URL, API_KEY, API_SECRET        │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  LiveKit Cloud  (project: poc-4bhx5srf)            │
│                                                    │
│  Media routing between browser and agent.          │
│  No application logic runs here.                   │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  Hetzner  (ubuntu-4gb-hel1-1 / 46.62.222.72)      │
│                                                    │
│  sada-agent.service                                │
│    → /opt/sada-hotel-agent/src/agent.py start      │
│    → logs to /var/log/sada/sessions/               │
│                                                    │
│  Env vars: LIVEKIT_*, OPENAI_API_KEY               │
│  No inbound ports. No nginx. No TLS.               │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  OpenAI  (Realtime API)                            │
│                                                    │
│  gpt-realtime model.                               │
│  Agent connects outbound via WebSocket.             │
│  ~$0.30/min of conversation.                       │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  Namecheap  (sada.sh domain)                       │
│                                                    │
│  A record → Vercel (76.76.21.21)                   │
│  MX → Namecheap email forwarding                   │
│  No DNS pointing at Hetzner for this service.      │
└────────────────────────────────────────────────────┘
```

---

## Future considerations

**Arabic support.** The current agent defaults to English and auto-detects if
the caller switches language. For production Arabic support, Google Gemini
Live is worth evaluating — Google's Arabic speech recognition leads the
market, and the swap is a single line change in the agent.

**Real hotel inventory.** Mock mode works for demos. For production, the
`hotel_client.py` interface (`search_hotels`, `search_offers`, `book_hotel`)
is provider-agnostic. Hotelbeds, Expedia Rapid, or a direct chain API
(Marriott, Hilton) can be plugged in by implementing those three functions.

**Persistent booking store.** Bookings currently live in the agent's process
memory and die with the call. A caller who rings back to amend gets a fresh
agent with no record. A database (even SQLite) would close this gap.

**DTMF for contact capture.** Letter-by-letter spelling over audio is the
weakest part of the flow, even with all the guards. Sending a "press 1 to
confirm" DTMF prompt, or texting a link for the caller to type their details,
would remove the problem at the source rather than patching it downstream.
