"""
Amadeus Self-Service API client for hotel search and booking.

Uses the official Amadeus Python SDK for authentication and API calls.
The SDK handles OAuth2 token management automatically.

API flow:
  1. Hotel List API    → find hotels by city code
  2. Hotel Search API  → get available offers (rooms + prices)
  3. Hotel Booking API → create reservation with guest + payment details
"""

import os
import logging
from amadeus import Client, ResponseError

logger = logging.getLogger("amadeus-client")

# ── City code mapping ────────────────────────────────────────────────
# Common cities the agent will encounter. The Amadeus Hotel List API
# requires IATA city codes, not free-text city names.
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


def _get_amadeus_client() -> Client:
    """Initialise an Amadeus SDK client from environment variables."""
    return Client(
        client_id=os.environ["AMADEUS_CLIENT_ID"],
        client_secret=os.environ["AMADEUS_CLIENT_SECRET"],
        # Use "production" for live bookings; "test" for sandbox
        hostname="test",
    )


def resolve_city_code(city_name: str) -> str | None:
    """Resolve a free-text city name to an IATA city code."""
    return CITY_CODES.get(city_name.strip().lower())


def search_hotels(city_code: str, max_results: int = 10) -> list[dict]:
    """
    Step 1: Find hotels in a city using the Hotel List API.

    Returns a simplified list of hotels with id, name, and location.
    """
    amadeus = _get_amadeus_client()

    try:
        response = amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
    except ResponseError as e:
        logger.error("Hotel List API error: %s", e)
        return []

    hotels = []
    for h in response.data[:max_results]:
        hotels.append(
            {
                "hotel_id": h.get("hotelId", ""),
                "name": h.get("name", "Unknown"),
                "iata_code": h.get("iataCode", ""),
                "latitude": h.get("geoCode", {}).get("latitude"),
                "longitude": h.get("geoCode", {}).get("longitude"),
                "distance_km": h.get("distance", {}).get("value"),
            }
        )
    return hotels


def search_offers(
    hotel_ids: list[str],
    check_in: str,
    check_out: str,
    adults: int = 1,
    rooms: int = 1,
) -> list[dict]:
    """
    Step 2: Get available room offers for one or more hotels.

    Args:
        hotel_ids: List of Amadeus hotel IDs (from search_hotels)
        check_in:  Check-in date as YYYY-MM-DD
        check_out: Check-out date as YYYY-MM-DD
        adults:    Number of adult guests
        rooms:     Number of rooms needed

    Returns a simplified list of offers with price, room type, etc.
    """
    amadeus = _get_amadeus_client()

    try:
        response = amadeus.shopping.hotel_offers_search.get(
            hotelIds=hotel_ids,
            checkInDate=check_in,
            checkOutDate=check_out,
            adults=adults,
            roomQuantity=rooms,
        )
    except ResponseError as e:
        logger.error("Hotel Search API error: %s", e)
        return []

    offers = []
    for hotel_data in response.data:
        hotel_name = hotel_data.get("hotel", {}).get("name", "Unknown")
        hotel_id = hotel_data.get("hotel", {}).get("hotelId", "")

        for offer in hotel_data.get("offers", []):
            price_info = offer.get("price", {})
            room_info = offer.get("room", {})
            room_desc = room_info.get("description", {}).get("text", "Standard Room")
            room_type = room_info.get("typeEstimated", {})

            offers.append(
                {
                    "offer_id": offer.get("id", ""),
                    "hotel_id": hotel_id,
                    "hotel_name": hotel_name,
                    "check_in": offer.get("checkInDate", check_in),
                    "check_out": offer.get("checkOutDate", check_out),
                    "room_description": room_desc,
                    "room_category": room_type.get("category", ""),
                    "bed_type": room_type.get("bedType", ""),
                    "beds": room_type.get("beds", 1),
                    "price_total": price_info.get("total", "N/A"),
                    "currency": price_info.get("currency", "USD"),
                    "cancellation_deadline": (
                        offer.get("policies", {})
                        .get("cancellations", [{}])[0]
                        .get("deadline", "N/A")
                        if offer.get("policies", {}).get("cancellations")
                        else "N/A"
                    ),
                    "payment_type": offer.get("policies", {}).get(
                        "paymentType", "N/A"
                    ),
                }
            )
    return offers


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
    """
    Step 3: Create a hotel booking from a confirmed offer.

    In test mode, this uses the Amadeus sandbox — no real charges.
    The card_number default is the standard Amadeus test card.

    Returns booking confirmation details.
    """
    amadeus = _get_amadeus_client()

    if card_holder is None:
        card_holder = f"{guest_first_name} {guest_last_name}"

    body = {
        "data": {
            "type": "hotel-order",
            "guests": [
                {
                    "tid": 1,
                    "name": {
                        "title": "MR",
                        "firstName": guest_first_name,
                        "lastName": guest_last_name,
                    },
                    "contact": {
                        "phone": guest_phone,
                        "email": guest_email,
                    },
                }
            ],
            "travelAgent": {
                "contact": {
                    "email": "kayode@sada.sh",
                }
            },
            "roomAssociations": [
                {
                    "guestReferences": [{"guestReference": "1"}],
                    "hotelOfferId": offer_id,
                }
            ],
            "payment": {
                "id": "1",
                "method": "CREDIT_CARD",
                "paymentCard": {
                    "paymentCardInfo": {
                        "vendorCode": "VI",
                        "cardNumber": card_number,
                        "expiryDate": card_expiry,
                        "holderName": card_holder,
                    }
                },
            },
        }
    }

    try:
        response = amadeus.post("/v2/booking/hotel-orders", body)
        booking_data = response.data[0] if response.data else response.result
    except ResponseError as e:
        logger.error("Hotel Booking API error: %s", e)
        return {"error": str(e), "success": False}

    return {
        "success": True,
        "booking_id": booking_data.get("id", ""),
        "provider_confirmation": booking_data.get(
            "associatedRecords", [{}]
        )[0].get("reference", "N/A")
        if booking_data.get("associatedRecords")
        else "N/A",
        "status": booking_data.get("type", "confirmed"),
    }
