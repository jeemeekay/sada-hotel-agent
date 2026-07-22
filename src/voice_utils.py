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


# National number lengths, excluding the country code. A UK mobile is
# +44 then 10 digits; accepting anything shorter is how a truncated turn
# ("plus four four seven five one seven two four seven zero") gets booked.
NATIONAL_LENGTHS: dict[str, tuple[int, ...]] = {
    "44": (10,),          # UK
    "971": (9,),          # UAE
    "1": (10,),           # US / Canada
    "966": (9,),          # Saudi Arabia
    "974": (8,),          # Qatar
    "968": (8,),          # Oman
    "965": (8,),          # Kuwait
    "973": (8,),          # Bahrain
    "20": (10,),          # Egypt
    "234": (10,),         # Nigeria
    "91": (10,),          # India
    "33": (9,),           # France
    "49": (10, 11),       # Germany
    "353": (9,),          # Ireland
}


def _split_country_code(digits: str) -> tuple[str, str] | None:
    """Split national digits from the country code, longest match first."""
    for length in (3, 2, 1):
        code = digits[:length]
        if code in NATIONAL_LENGTHS:
            return code, digits[length:]
    return None


def normalise_phone(raw: str) -> str:
    """Normalise to international form, dropping the national trunk prefix.

    Callers naturally give the domestic form and the country code separately:
    "zero seven five one seven... and the code is plus four four". Joined
    literally that becomes +4407517247059, which is not a valid number — the
    leading 0 is a domestic trunk prefix and must be dropped when a country
    code is present.
    """
    cleaned = raw.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    plus = cleaned.startswith("+")
    digits = "".join(c for c in cleaned if c.isdigit())

    if plus:
        split = _split_country_code(digits)
        if split:
            code, national = split
            national = national.lstrip("0")
            digits = code + national

    return ("+" if plus else "") + digits


def validate_phone(phone: str) -> tuple[bool, str]:
    """Reject partial numbers, and numbers without a country code.

    Two failure modes seen on real calls, both of which booked bad data:
      - "plus four four" accepted as a whole number mid-sentence
      - "+4475172470" accepted when the caller's final "five nine" landed
        in the next turn
    Checking the length against the country code catches the second.
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

    split = _split_country_code(digits)
    if split:
        code, national = split
        national = national.lstrip("0")  # 07517... -> 7517...
        expected = NATIONAL_LENGTHS[code]
        if len(national) not in expected:
            want = " or ".join(str(e) for e in expected)
            return False, (
                f"'{phone}' has {len(national)} digits after the country code "
                f"+{code}, but numbers in that country have {want}. The caller "
                "was probably cut off. Ask them to give the whole number again "
                "in one go, and wait until they stop speaking before you act."
            )

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


# ── Spoken-form parsing ──────────────────────────────────────────────
# Callers dictate addresses as words: "k a y o d e at o four j dot co dot
# u k". Those words must become characters BEFORE anything is spelled back,
# otherwise "four" gets read out as F-O-U-R and the caller hears nonsense.

NUMBER_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9",
}

SYMBOL_WORDS = {
    "dot": ".", "point": ".", "period": ".", "full-stop": ".",
    "at": "@",
    "dash": "-", "hyphen": "-", "minus": "-",
    "underscore": "_", "under-score": "_",
    "plus": "+",
}

# Words a caller says around the address that are not part of it
FILLER_WORDS = {
    "my", "email", "address", "is", "it's", "its", "the", "and",
    "symbol", "sign", "please", "so", "that's", "thats",
}


def parse_spoken_text(spoken: str) -> str:
    """Turn a dictated string into the characters it represents.

    "k a y o d e at o four j dot co dot u k" -> "kayode@o4j.co.uk"

    Single letters are joined, number words become digits, and symbol words
    become symbols. Multi-letter tokens like "co" or "kayode" pass through
    as-is, so a caller can mix spelling and speaking freely.
    """
    out: list[str] = []
    for raw in spoken.lower().split():
        token = raw.strip(".,!?;:")
        if not token:
            continue
        if token in FILLER_WORDS:
            continue
        if token in NUMBER_WORDS:
            out.append(NUMBER_WORDS[token])
        elif token in SYMBOL_WORDS:
            out.append(SYMBOL_WORDS[token])
        else:
            out.append(token)
    return "".join(out)


def parse_spoken_email(spoken: str) -> tuple[str, str]:
    """Parse a dictated email. Returns (address, problem).

    problem is an empty string when the address is usable, otherwise a
    message phrased so the agent can ask one specific follow-up question.
    """
    address = parse_spoken_text(spoken)

    if not address:
        return "", "Nothing was captured. Ask the caller to say it again."

    if "@" not in address:
        return address, (
            f"The caller did not say where the at symbol goes — what came "
            f"through was '{address}'. Ask them only this: which part comes "
            "before the at symbol? Do not ask for the whole address again."
        )

    ok, why = validate_email(address)
    if not ok:
        return address, why

    return address, ""


def name_matches_email(first_name: str, email: str) -> tuple[bool, str]:
    """Cross-check a spelled name against the email address.

    People overwhelmingly put their own first name in their email address,
    so the two are a free check on each other. A caller who spells
    "K-A-Y-O-D-E" and gives kayode@... is consistent; one whose name comes
    through as "Teyode" against the same address has been misheard, and the
    address is the more reliable of the two because it was spoken as a whole
    rather than letter by letter.

    Returns (looks_consistent, reason). Similar-but-different is the
    suspicious case — completely unrelated is normal and passes, since
    plenty of people use nicknames or shared addresses.
    """
    from difflib import SequenceMatcher

    local = email.split("@")[0].lower()
    local = "".join(c for c in local if c.isalpha())
    name = "".join(c for c in first_name.lower() if c.isalpha())

    if not local or not name or len(name) < 3:
        return True, ""

    if name in local or local in name:
        return True, ""

    similarity = SequenceMatcher(None, name, local).ratio()

    # 0.6 catches one or two wrong letters in a short name without
    # flagging genuinely different words.
    if similarity >= 0.6:
        return False, (
            f"The name '{first_name.capitalize()}' and the email address "
            f"'{email}' are close but not the same, which usually means one "
            "letter was misheard while spelling. The email is more reliable "
            "because it was said as a whole word. Ask the caller to confirm "
            "the spelling of their first name once more, slowly. If they "
            "confirm the name as you have it, call prepare_booking again "
            "with the same values and it will go through."
        )

    return True, ""
