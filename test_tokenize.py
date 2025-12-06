# test_tokenize.py
# ---------------------------------------------------------
# Test script for preprocessing.tokenization functions
# Mirrors the structured style used in `test_clean.py`
# ---------------------------------------------------------

from preprocessing.tokenization import (
    tokenize_chars,
    tokenize_words,
    build_char_vocab,
    encode_labeled_sentence,
)

# ---------------------------------------------------------
# Test: Character tokenization
# ---------------------------------------------------------
print("=== TEST: Tokenize Characters ===")
print("Input: 'مدرسة'")
print(tokenize_chars("مدرسة"))

# ---------------------------------------------------------
# Test: Word tokenization
# ---------------------------------------------------------
print("\n=== TEST: Tokenize Words ===")
print("Input: 'ذهب محمد الى المدرسة'")
print(tokenize_words("ذهب محمد الى المدرسة"))

# ---------------------------------------------------------
# Test: Vocabulary Builder
# ---------------------------------------------------------
sentences = ["ذهب محمد", "قال الرجل"]
print("\n=== TEST: Vocabulary Builder ===")
print("Sample sentences:", sentences)
char2id, id2char = build_char_vocab(sentences)
print("char2id mapping (sample):", char2id)

# ---------------------------------------------------------
# Test: Encoding a labeled sentence (diacritized input required)
# ---------------------------------------------------------
print("\n=== TEST: Encoding a labeled sentence ===")
X, Y = encode_labeled_sentence("ذَهَبَ")
print("X (base characters):", X)
print("Y (diacritic label IDs):", Y)
