"""
Hotel API client — mock mode for demos, Hotelbeds for production.

In MOCK mode (default): returns realistic UAE/global hotel data so the
voice agent works end-to-end without any external API keys.

In HOTELBEDS mode: connects to the Hotelbeds APItude API for real
availability and booking.

Set HOTEL_API_MODE=hotelbeds in .env to switch. Default is "mock".
"""

import os
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta

logger = logging.getLogger("hotel-client")

# ── City code mapping ────────────────────────────────────────────────
CITY_CODES: dict[str, str] = {
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "riyadh": "RUH",
    "jeddah": "JED",
    "doha": "DOH",
    "muscat": "MCT",
    "kuwait city": "KWI",
    "manama": "BAH",
    "cairo": "CAI",
    "london": "LON",
    "paris": "PAR",
    "new york": "NYC",
    "los angeles": "LAX",
    "tokyo": "TYO",
    "singapore": "SIN",
    "bangkok": "BKK",
    "istanbul": "IST",
    "mumbai": "BOM",
    "sydney": "SYD",
    "berlin": "BER",
    "rome": "ROM",
    "barcelona": "BCN",
    "amsterdam": "AMS",
    "zurich": "ZRH",
    "hong kong": "HKG",
    "kuala lumpur": "KUL",
    "lagos": "LOS",
    "nairobi": "NBO",
    "johannesburg": "JNB",
    "toronto": "YTO",
    "san francisco": "SFO",
    "miami": "MIA",
    "chicago": "CHI",
    "washington": "WAS",
    "boston": "BOS",
    "seattle": "SEA",
}

# ── Mock hotel data by city ──────────────────────────────────────────
MOCK_HOTELS: dict[str, list[dict]] = {
    "DXB": [
        {"hotel_id": "HTDXB001", "name": "Jumeirah Beach Hotel", "iata_code": "DXB", "latitude": 25.1412, "longitude": 55.1855, "distance_km": 2.1},
        {"hotel_id": "HTDXB002", "name": "Marriott Resort Palm Jumeirah", "iata_code": "DXB", "latitude": 25.1120, "longitude": 55.1380, "distance_km": 4.5},
        {"hotel_id": "HTDXB003", "name": "Rove Downtown Dubai", "iata_code": "DXB", "latitude": 25.1972, "longitude": 55.2744, "distance_km": 0.8},
        {"hotel_id": "HTDXB004", "name": "Hilton Dubai Creek", "iata_code": "DXB", "latitude": 25.2630, "longitude": 55.3215, "distance_km": 3.2},
        {"hotel_id": "HTDXB005", "name": "Address Downtown", "iata_code": "DXB", "latitude": 25.1950, "longitude": 55.2780, "distance_km": 0.5},
        {"hotel_id": "HTDXB006", "name": "Radisson Blu Hotel Dubai Waterfront", "iata_code": "DXB", "latitude": 25.1800, "longitude": 55.2500, "distance_km": 1.9},
    ],
    "AUH": [
        {"hotel_id": "HTAUH001", "name": "Emirates Palace Mandarin Oriental", "iata_code": "AUH", "latitude": 24.4615, "longitude": 54.3173, "distance_km": 1.5},
        {"hotel_id": "HTAUH002", "name": "Yas Hotel Abu Dhabi", "iata_code": "AUH", "latitude": 24.4677, "longitude": 54.6030, "distance_km": 8.0},
        {"hotel_id": "HTAUH003", "name": "Rotana Khalidiya Palace", "iata_code": "AUH", "latitude": 24.4700, "longitude": 54.3400, "distance_km": 2.1},
        {"hotel_id": "HTAUH004", "name": "Hilton Abu Dhabi Yas Island", "iata_code": "AUH", "latitude": 24.4900, "longitude": 54.6100, "distance_km": 7.5},
    ],
    "LON": [
        {"hotel_id": "HTLON001", "name": "The Savoy", "iata_code": "LON", "latitude": 51.5104, "longitude": -0.1202, "distance_km": 0.3},
        {"hotel_id": "HTLON002", "name": "Premier Inn London City Tower Hill", "iata_code": "LON", "latitude": 51.5100, "longitude": -0.0760, "distance_km": 1.2},
        {"hotel_id": "HTLON003", "name": "Hilton London Bankside", "iata_code": "LON", "latitude": 51.5050, "longitude": -0.1010, "distance_km": 0.8},
        {"hotel_id": "HTLON004", "name": "citizenM Tower of London", "iata_code": "LON", "latitude": 51.5098, "longitude": -0.0770, "distance_km": 1.1},
        {"hotel_id": "HTLON005", "name": "Marriott Hotel County Hall", "iata_code": "LON", "latitude": 51.5020, "longitude": -0.1170, "distance_km": 0.5},
    ],
    "PAR": [
        {"hotel_id": "HTPAR001", "name": "Hôtel Plaza Athénée", "iata_code": "PAR", "latitude": 48.8660, "longitude": 2.3040, "distance_km": 0.7},
        {"hotel_id": "HTPAR002", "name": "Ibis Paris Bastille Opera", "iata_code": "PAR", "latitude": 48.8530, "longitude": 2.3700, "distance_km": 2.1},
        {"hotel_id": "HTPAR003", "name": "Novotel Paris Centre Tour Eiffel", "iata_code": "PAR", "latitude": 48.8480, "longitude": 2.2900, "distance_km": 1.5},
        {"hotel_id": "HTPAR004", "name": "Le Marais Boutique Hotel", "iata_code": "PAR", "latitude": 48.8570, "longitude": 2.3620, "distance_km": 1.8},
    ],
    "NYC": [
        {"hotel_id": "HTNYC001", "name": "The Plaza Hotel", "iata_code": "NYC", "latitude": 40.7645, "longitude": -73.9744, "distance_km": 0.5},
        {"hotel_id": "HTNYC002", "name": "Pod 51 Hotel", "iata_code": "NYC", "latitude": 40.7550, "longitude": -73.9700, "distance_km": 1.0},
        {"hotel_id": "HTNYC003", "name": "Hilton Midtown Manhattan", "iata_code": "NYC", "latitude": 40.7620, "longitude": -73.9800, "distance_km": 0.3},
        {"hotel_id": "HTNYC004", "name": "Marriott Marquis Times Square", "iata_code": "NYC", "latitude": 40.7580, "longitude": -73.9860, "distance_km": 0.6},
    ],
}

# Pricing tiers for mock offers (per night in USD)
MOCK_PRICING = {
    "budget":   {"min": 80, "max": 150, "room": "Standard Room", "bed": "DOUBLE", "beds": 1},
    "mid":      {"min": 150, "max": 300, "room": "Deluxe Room", "bed": "KING", "beds": 1},
    "premium":  {"min": 300, "max": 600, "room": "Executive Suite", "bed": "KING", "beds": 1},
    "luxury":   {"min": 600, "max": 1200, "room": "Presidential Suite", "bed": "KING", "beds": 2},
}

# Map hotel name keywords to price tier
def _price_tier(hotel_name: str) -> str:
    name_lower = hotel_name.lower()
    if any(w in name_lower for w in ["plaza", "palace", "savoy", "athénée", "address", "emirates"]):
        return "luxury"
    elif any(w in name_lower for w in ["hilton", "marriott", "novotel", "radisson"]):
        return "mid"
    elif any(w in name_lower for w in ["pod", "ibis", "premier inn", "rove", "citizenm"]):
        return "budget"
    else:
        return "mid"


def _get_api_mode() -> str:
    return os.environ.get("HOTEL_API_MODE", "mock").lower()


def resolve_city_code(city_name: str) -> str | None:
    """Resolve a free-text city name to an IATA city code."""
    return CITY_CODES.get(city_name.strip().lower())


def search_hotels(city_code: str, max_results: int = 10) -> list[dict]:
    """Find hotels in a city. Uses mock data or Hotelbeds based on config."""
    if _get_api_mode() == "hotelbeds":
        return _hotelbeds_search_hotels(city_code, max_results)
    return _mock_search_hotels(city_code, max_results)


def search_offers(
    hotel_ids: list[str],
    check_in: str,
    check_out: str,
    adults: int = 1,
    rooms: int = 1,
) -> list[dict]:
    """Get available room offers. Uses mock data or Hotelbeds based on config."""
    if _get_api_mode() == "hotelbeds":
        return _hotelbeds_search_offers(hotel_ids, check_in, check_out, adults, rooms)
    return _mock_search_offers(hotel_ids, check_in, check_out, adults, rooms)


def book_hotel(
    offer_id: str,
    guest_first_name: str,
    guest_last_name: str,
    guest_email: str,
    guest_phone: str,
    card_number: str = "4111111111111111",
    card_expiry: str = "2026-12",
    card_holder: str | None = None,
) -> dict:
    """Create a booking. Uses mock confirmation or Hotelbeds based on config."""
    if _get_api_mode() == "hotelbeds":
        return _hotelbeds_book(offer_id, guest_first_name, guest_last_name,
                               guest_email, guest_phone)
    return _mock_book(offer_id, guest_first_name, guest_last_name,
                      guest_email, guest_phone)


# ═══════════════════════════════════════════════════════════════════════
#  MOCK IMPLEMENTATION — realistic demo data, no API keys needed
# ═══════════════════════════════════════════════════════════════════════

def _generate_fallback_hotels(city_code: str) -> list[dict]:
    """Deterministically generate hotels for a city with no curated data.

    Must be deterministic: _mock_search_offers regenerates this same list to
    resolve a hotel_id back to its name, so IDs stay valid across calls.
    """
    chains = [
        "Grand Plaza",
        "Marriott",
        "Hilton",
        "Novotel",
        "Ibis",
    ]
    return [
        {
            "hotel_id": f"HT{city_code}{i:03d}",
            "name": f"{chain} {city_code}",
            "iata_code": city_code,
            "latitude": 0.0,
            "longitude": 0.0,
            "distance_km": round(0.5 + i * 1.2, 1),
        }
        for i, chain in enumerate(chains, 1)
    ]


def _lookup_hotel(hotel_id: str) -> dict | None:
    """Resolve a hotel_id to its record, including generated fallback cities."""
    for city_hotels in MOCK_HOTELS.values():
        for h in city_hotels:
            if h["hotel_id"] == hotel_id:
                return h

    # Fallback IDs look like HT<CITY><NNN>, e.g. HTTYO002
    if hotel_id.startswith("HT") and len(hotel_id) >= 8:
        city_code = hotel_id[2:-3]
        for h in _generate_fallback_hotels(city_code):
            if h["hotel_id"] == hotel_id:
                return h
    return None


def _mock_search_hotels(city_code: str, max_results: int) -> list[dict]:
    hotels = MOCK_HOTELS.get(city_code) or _generate_fallback_hotels(city_code)
    return hotels[:max_results]


def _mock_search_offers(
    hotel_ids: list[str],
    check_in: str,
    check_out: str,
    adults: int,
    rooms: int,
) -> list[dict]:
    # Parse dates. Raise a clear error rather than silently continuing —
    # the caller (agent tool) turns this into a spoken message.
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d")
        co = datetime.strptime(check_out, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Dates must be YYYY-MM-DD. Got check_in={check_in!r}, "
            f"check_out={check_out!r}."
        )
    nights = max((co - ci).days, 1)

    offers = []
    for hotel_id in hotel_ids:
        hotel = _lookup_hotel(hotel_id)
        if not hotel:
            continue

        tier = _price_tier(hotel["name"])
        pricing = MOCK_PRICING[tier]

        # Deterministic "random" price based on hotel_id + dates
        seed = int(hashlib.md5(f"{hotel_id}{check_in}".encode()).hexdigest()[:8], 16)
        per_night = pricing["min"] + (seed % (pricing["max"] - pricing["min"]))
        total = per_night * nights

        # Cancellation deadline: 3 days before check-in, but never in the
        # past — quoting an expired deadline to a caller is nonsense.
        cancel_date = max(ci - timedelta(days=3), datetime.now())

        offer_id = f"OFF-{hotel_id}-{check_in.replace('-', '')}"
        offers.append({
            "offer_id": offer_id,
            "hotel_id": hotel_id,
            "hotel_name": hotel["name"],
            "check_in": check_in,
            "check_out": check_out,
            "room_description": pricing["room"],
            "room_category": tier.upper(),
            "bed_type": pricing["bed"],
            "beds": pricing["beds"],
            "price_per_night": str(per_night),
            "price_total": str(total),
            "currency": "USD",
            "cancellation_deadline": cancel_date.strftime("%Y-%m-%d"),
            "payment_type": "CREDIT_CARD",
        })

    return offers


def _mock_book(
    offer_id: str,
    guest_first_name: str,
    guest_last_name: str,
    guest_email: str,
    guest_phone: str,
) -> dict:
    # Generate a deterministic confirmation number
    confirm_hash = hashlib.md5(
        f"{offer_id}{guest_last_name}{time.time()}".encode()
    ).hexdigest()[:8].upper()

    return {
        "success": True,
        "booking_id": f"SADA-{confirm_hash}",
        "provider_confirmation": f"DEMO-{confirm_hash}",
        "status": "confirmed",
        "note": "This is a demo booking — no real reservation was created.",
    }


# ═══════════════════════════════════════════════════════════════════════
#  HOTELBEDS IMPLEMENTATION — real API (requires API key from
#  developer.hotelbeds.com)
# ═══════════════════════════════════════════════════════════════════════

def _hotelbeds_auth_headers() -> dict:
    """Generate Hotelbeds API authentication headers (API Key + X-Signature)."""
    api_key = os.environ["HOTELBEDS_API_KEY"]
    secret = os.environ["HOTELBEDS_API_SECRET"]
    timestamp = str(int(time.time()))
    sig_raw = f"{api_key}{secret}{timestamp}"
    signature = hashlib.sha256(sig_raw.encode()).hexdigest()

    return {
        "Api-key": api_key,
        "X-Signature": signature,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _hotelbeds_base_url() -> str:
    env = os.environ.get("HOTELBEDS_ENV", "test")
    if env == "production":
        return "https://api.hotelbeds.com"
    return "https://api.test.hotelbeds.com"


def _hotelbeds_search_hotels(city_code: str, max_results: int) -> list[dict]:
    """Hotelbeds doesn't have a simple 'list hotels by city' endpoint —
    you search availability directly. This returns mock data as a fallback
    and the real search happens in search_offers."""
    logger.info("Hotelbeds: hotel list by city not directly supported, "
                "use search_offers with destination code instead.")
    return _mock_search_hotels(city_code, max_results)


def _hotelbeds_search_offers(
    hotel_ids: list[str],
    check_in: str,
    check_out: str,
    adults: int,
    rooms: int,
) -> list[dict]:
    """Search Hotelbeds for hotel availability."""
    import aiohttp
    import asyncio

    url = f"{_hotelbeds_base_url()}/hotel-api/1.0/hotels"
    headers = _hotelbeds_auth_headers()

    occupancy = [{"rooms": rooms, "adults": adults, "children": 0}]
    body = {
        "stay": {"checkIn": check_in, "checkOut": check_out},
        "occupancies": occupancy,
        "hotels": {"hotel": [int(hid) for hid in hotel_ids if hid.isdigit()]},
    }

    try:
        import requests
        resp = requests.post(url, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Hotelbeds availability error: %s", e)
        return []

    offers = []
    for hotel_data in data.get("hotels", {}).get("hotels", []):
        hotel_name = hotel_data.get("name", "Unknown")
        hotel_code = str(hotel_data.get("code", ""))

        for room in hotel_data.get("rooms", []):
            for rate in room.get("rates", []):
                offers.append({
                    "offer_id": rate.get("rateKey", ""),
                    "hotel_id": hotel_code,
                    "hotel_name": hotel_name,
                    "check_in": check_in,
                    "check_out": check_out,
                    "room_description": room.get("name", "Standard Room"),
                    "room_category": rate.get("boardName", ""),
                    "bed_type": "DOUBLE",
                    "beds": 1,
                    "price_total": rate.get("net", "N/A"),
                    "currency": data.get("hotels", {}).get("currency", "USD"),
                    "cancellation_deadline": rate.get("cancellationPolicies", [{}])[0].get("from", "N/A")
                    if rate.get("cancellationPolicies") else "N/A",
                    "payment_type": rate.get("paymentType", "AT_WEB"),
                })
    return offers


def _hotelbeds_book(
    offer_id: str,
    guest_first_name: str,
    guest_last_name: str,
    guest_email: str,
    guest_phone: str,
) -> dict:
    """Create a booking via Hotelbeds."""
    url = f"{_hotelbeds_base_url()}/hotel-api/1.0/bookings"
    headers = _hotelbeds_auth_headers()

    body = {
        "holder": {"name": guest_first_name, "surname": guest_last_name},
        "rooms": [
            {
                "rateKey": offer_id,
                "paxes": [
                    {"roomId": 1, "type": "AD", "name": guest_first_name,
                     "surname": guest_last_name}
                ],
            }
        ],
        "clientReference": f"SADA-{int(time.time())}",
    }

    try:
        import requests
        resp = requests.post(url, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        booking = data.get("booking", {})
    except Exception as e:
        logger.error("Hotelbeds booking error: %s", e)
        return {"error": str(e), "success": False}

    return {
        "success": True,
        "booking_id": booking.get("reference", ""),
        "provider_confirmation": booking.get("clientReference", ""),
        "status": booking.get("status", "CONFIRMED"),
    }
