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
# Alternative faster method using regex 
# DIACRITICS_PATTERN = re.compile(r"[\u064b-\u0652\u0670]")
# def remove_diacritics(text: str) -> str:
#     """Remove diacritics using one fast regex."""
#     return DIACRITICS_PATTERN.sub("", text)



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
def clean_sentence(sentence: str) -> str:
    """Normalize + remove unwanted characters."""
    sentence = normalize_arabic(sentence)

    # Remove English letters and numbers
    sentence = re.sub(r"[A-Za-z0-9]", " ", sentence)

    # Keep only Arabic letters, diacritics, and spaces
    allowed = (
        r"[^ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي"
        + "".join(ARABIC_DIACRITICS)
        + r"\s]"
    )
    sentence = re.sub(allowed, " ", sentence)

    # Collapse multiple spaces
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence

# Alternative method with predefined ranges -- after opening training set
# ARABIC = r"ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي"
# DIACRITICS = r"\u064b-\u0652\u0670"  

# def clean_sentence(sentence):

#     # 1) Remove numbers & English
#     sentence = re.sub(r"[A-Za-z0-9]+", " ", sentence)

#     # 2) Remove brackets, slashes, page references, punctuation
#     sentence = re.sub(r"[«»\(\)\[\]{}<>:;/\\\-–—_.,!?+*=]", " ", sentence)

#     # 3) Allow ONLY Arabic + harakat + spaces
#     sentence = re.sub(rf"[^{ARABIC}{DIACRITICS}\s]", " ", sentence)

#     # 4) Normalize spaces
#     sentence = re.sub(r"\s+", " ", sentence).strip()

#     return sentence


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


# ---------------------------------------------------------
# 7. Full preprocess wrapper
# ---------------------------------------------------------
def preprocess(sentence: str) -> str:
    return clean_sentence(sentence)
