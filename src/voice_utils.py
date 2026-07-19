"""
Helpers for handling contact details over a phone line.

Spelled-out letters are where phone-quality STT degrades worst ("k a y o d e"
comes back as "a a y o d e"). Reading details back phonetically is the
standard mitigation: the NATO alphabet is designed to stay distinguishable
over a noisy channel, which plain letter names are not.
"""

import re

NATO = {
    "a": "alpha", "b": "bravo", "c": "charlie", "d": "delta",
    "e": "echo", "f": "foxtrot", "g": "golf", "h": "hotel",
    "i": "india", "j": "juliet", "k": "kilo", "l": "lima",
    "m": "mike", "n": "november", "o": "oscar", "p": "papa",
    "q": "quebec", "r": "romeo", "s": "sierra", "t": "tango",
    "u": "uniform", "v": "victor", "w": "whiskey", "x": "x-ray",
    "y": "yankee", "z": "zulu",
}

SYMBOLS = {
    "@": "at",
    ".": "dot",
    "-": "dash",
    "_": "underscore",
    "+": "plus",
}


def spell_phonetically(text: str) -> str:
    """Render a string as a phonetic readback script.

    "kayode@o4j.co.uk" ->
        "K for kilo, A for alpha, Y for yankee, ... at ... "
    """
    parts: list[str] = []
    for ch in text:
        low = ch.lower()
        if low in NATO:
            parts.append(f"{ch.upper()} for {NATO[low]}")
        elif ch.isdigit():
            parts.append(f"the number {ch}")
        elif ch in SYMBOLS:
            parts.append(SYMBOLS[ch])
        elif ch == " ":
            parts.append("space")
        else:
            parts.append(ch)
    return ", ".join(parts)


def normalise_email(raw: str) -> str:
    """Clean up an email the STT may have mangled into spoken form."""
    cleaned = raw.strip().lower()
    cleaned = cleaned.replace(" at ", "@").replace(" dot ", ".")
    cleaned = cleaned.replace(" underscore ", "_").replace(" dash ", "-")
    # Collapse the spaces left behind by spelled-out letters
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$")


def validate_email(email: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Reason is phrased for the LLM to act on."""
    if not email or "@" not in email:
        return False, "The email address is missing an at symbol."
    if not EMAIL_RE.match(email):
        return False, (
            f"'{email}' is not a valid email address. It needs text before "
            "the at symbol, a domain after it, and a dot with a suffix such "
            "as com or co dot uk."
        )
    return True, ""


def normalise_phone(raw: str) -> str:
    """Strip everything except digits and a leading plus."""
    cleaned = raw.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    plus = cleaned.startswith("+")
    digits = "".join(c for c in cleaned if c.isdigit())
    return ("+" if plus else "") + digits


def validate_phone(phone: str) -> tuple[bool, str]:
    """Reject partial numbers and numbers without a country code.

    A caller saying "plus four four" mid-sentence must never be accepted as a
    complete number — that is exactly how a booking gets made against a
    fragment of a turn. A number with no country code is equally useless to a
    hotel that may be in another country.
    """
    digits = "".join(c for c in phone if c.isdigit())
    if not phone.startswith("+"):
        return False, (
            f"'{phone}' has no country code. Ask the caller which country "
            "their number is in, or for the dialling code, and include it. "
            "For the UK that is plus four four, for the UAE plus nine seven one."
        )
    if len(digits) < 8:
        return False, (
            f"'{phone}' is only {len(digits)} digits, which is an incomplete "
            "phone number. The caller was probably still speaking. Ask them "
            "for the full number and wait until they have finished."
        )
    if len(digits) > 15:
        return False, f"'{phone}' has too many digits to be a phone number."
    return True, ""


def validate_name(name: str, field: str) -> tuple[bool, str]:
    """Catch STT artefacts like a name captured as single spelled letters."""
    stripped = name.strip()
    if len(stripped) < 2:
        return False, f"The {field} '{name}' is too short to be correct."
    # "a a y o d e" — spelled letters that were never reassembled
    tokens = stripped.split()
    if len(tokens) > 2 and all(len(t.strip(".,")) == 1 for t in tokens):
        return False, (
            f"The {field} came through as separate letters ('{name}'). Join "
            "them into a single word and confirm the spelling with the caller."
        )
    return True, ""


def letters_to_word(spelling: str) -> str:
    """Turn a confirmed letter sequence into a word.

    "k a y o d e" -> "Kayode". Also copes with "K for kilo, A for alpha"
    by taking the first character of each comma-separated group.
    """
    text = spelling.strip()
    if not text:
        return ""

    # Phonetic form: "K for kilo, A for alpha, ..."
    if " for " in text.lower():
        letters = [
            part.strip()[0]
            for part in text.split(",")
            if part.strip()
        ]
        return "".join(letters).capitalize()

    # Plain spaced form: "k a y o d e"
    tokens = [t.strip(".,") for t in text.split() if t.strip(".,")]
    if all(len(t) == 1 for t in tokens):
        return "".join(tokens).capitalize()

    # Already a word
    return text.strip(".,").capitalize()


def spellings_agree(name: str, spelling: str) -> bool:
    """Check a spoken name against the letters the caller actually confirmed."""
    return name.strip().lower() == letters_to_word(spelling).lower()
