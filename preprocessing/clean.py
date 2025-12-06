# clean.py
# ---------------------------------------------------------
# Arabic text cleaning + diacritic handling for diacritization
# ---------------------------------------------------------

import re


# ---------------------------------------------------------
# 1. Arabic diacritics (single + combined)
# ---------------------------------------------------------
ARABIC_DIACRITICS = [
    # Single diacritics
    "\u064b",  # Tanween Fath (ً)
    "\u064c",  # Tanween Damm (ٌ)
    "\u064d",  # Tanween Kasr (ٍ)
    "\u064e",  # Fatha (َ)
    "\u064f",  # Damma (ُ)
    "\u0650",  # Kasra (ِ)
    "\u0651",  # Shadda (ّ)
    "\u0652",  # Sukun (ْ)
    "\u0670",  # Superscript Alef (ٰ)

    # Combined forms (Shadda + vowel)
    "\u0651\u064e",  # Shadda + Fatha (َّ)
    "\u0651\u064f",  # Shadda + Damma (ُّ)
    "\u0651\u0650",  # Shadda + Kasra (ِّ)

    # Combined with tanween (rare but safe)
    "\u0651\u064b",  # Shadda + Tanween Fath (ًّ)
    "\u0651\u064c",  # Shadda + Tanween Damm (ٌّ)
    "\u0651\u064d",  # Shadda + Tanween Kasr (ٍّ)
    "\u0651\u0652",  # Shadda + Sukun (ّْ)
]


# ---------------------------------------------------------
# 2. Remove all diacritics from text
# ---------------------------------------------------------
def remove_diacritics(text: str) -> str:
    """Remove all Arabic diacritics (single + combined)."""
    for d in ARABIC_DIACRITICS:
        text = text.replace(d, "")
    return text

# ---------------------------------------------------------
# 3. Normalize Arabic characters
# ---------------------------------------------------------
def normalize_arabic(text: str) -> str:
    """Normalize common Arabic letter shapes."""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    # Make ta-marbuta → ه optional — essential for accuracy:
    # if convert_ta_marbuta:
    #     text = re.sub("ة", "ه", text)
    text = text.replace("ـ", "")  # remove tatweel
    return text

# ---------------------------------------------------------
# 4. Clean text line
# ---------------------------------------------------------
ARABIC_LETTERS = "ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي"

def clean_sentence(sentence: str) -> str:
    """
    Cleans Arabic sentence while preserving diacritics defined in ARABIC_DIACRITICS.
    Steps:
        1) Normalize Arabic letters (Reduce variations)
        2) Remove English digits & characters
        3) Remove punctuation / symbols
        4) Keep ONLY Arabic letters + diacritics + spaces
        5) Compress multiple spaces
    """

    # (1) normalize 
    sentence = normalize_arabic(sentence)
    # (2) Remove English & numbers
    sentence = re.sub(r"[A-Za-z0-9]+", " ", sentence)

    # (3) Remove punctuation, brackets, symbols
    sentence = re.sub(r"[«»()\[\]{}<>؛:;/\\\-–—_.,!?+*=]", " ", sentence)

    # (4) Allow only Arabic + diacritics + whitespace
    allowed_pattern = rf"[^{ARABIC_LETTERS}{''.join(ARABIC_DIACRITICS)}\s]"
    sentence = re.sub(allowed_pattern, " ", sentence)

    # (5) Remove duplicate spaces
    sentence = re.sub(r"\s+", " ", sentence).strip()

    return sentence

# ---------------------------------------------------------
# 5. Split word into (base, diacritics)
# ---------------------------------------------------------
def split_word_into_labeled_chars(word: str):
    """
    Given a word with decomposed Unicode, group each Arabic letter
    with its following diacritics.

    Example:
        input :  "ثُمَّ"
        output:  [("ث", "ُ"), ("م", "َّ")]
    """
    result = []
    base = None
    diacritics = ""

    for ch in word:
        if ch in ARABIC_DIACRITICS:
            # Diacritic belongs to previous base letter
            if base is not None:
                diacritics += ch
            continue

        # New base letter begins
        if base is not None:
            result.append((base, diacritics))

        base = ch # store the new letter
        diacritics = "" # reset diacritics (we will collect new ones)

    # Append last letter
    if base is not None:
        result.append((base, diacritics))

    return result


# ---------------------------------------------------------
# 6. Sentence → list of (base, diacritics)
# ---------------------------------------------------------
def split_sentence_into_labeled_chars(sentence: str):
    """
    Splits a sentence into labeled chars.
    Returns list of tuples (base_char, diacritic_string).
    Spaces are preserved as (' ', 'NONE').
    """
    words = sentence.split()
    labeled = []

    for word in words:
        pairs = split_word_into_labeled_chars(word)
        labeled.extend(pairs)
        labeled.append((" ", "NONE"))  # keep space separator

    if labeled:
        labeled.pop()  # remove the last extra space

    return labeled
