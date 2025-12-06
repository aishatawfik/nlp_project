"""preprocessing/tokenize.py

Utilities for tokenizing and encoding Arabic text for a
diacritization model. This module contains:

- character vocabulary builder (char2id / id2char)
- diacritic label definitions (diac2id / id2diac)
- character- and word-level tokenizers
- functions to encode sentences into model-ready inputs

Small examples are included in docstrings for clarity.
"""

from clean import (
    clean_sentence,
    remove_diacritics,
    split_sentence_into_labeled_chars
)


# =========================================================
# (1) Character Vocabulary
# =========================================================
# =========================================================
# (1) Build Character Vocabulary
# ---------------------------------------------------------
# This function scans ALL sentences in the dataset and collects
# every unique Arabic character that appears (including space).
#
# WHY?
# ----
# Neural networks cannot work directly with letters → they need
# numbers. So we create two mappings:
#
#   char2id : maps each character → a numeric index
#   id2char : maps numeric index → back to the character
#
# Example:
# --------
# sentences = ["ذهب محمد", "قال الرجل"]
#
# collected characters will be:
#   {' ', 'ذ', 'ه', 'ب', 'م', 'ح', 'م', 'د', 'ق', 'ا', 'ل', 'ر', 'ج', 'ل'}
#
# After sorting and indexing:
#   char2id might become:
#       {
#         ' ' : 0,
#         'ا' : 1,
#         'ب' : 2,
#         'ح' : 3,
#         'ذ' : 4,
#         'د' : 5,
#         'ر' : 6,
#         'ج' : 7,
#         'ل' : 8,
#         'م' : 9,
#         'ه' : 10
#       }
#
# Later, when training:
#   "ذهب" → [4,10,2]   # numerical form used by embedding layer
#
# This vocabulary becomes the foundation of the entire model.
# =========================================================

# =========================================================
# build_char_vocab(sentences)
# ---------------------------------------------------------
# INPUT  : list of sentences (already cleaned)
# OUTPUT : 
#   1) char2id → dictionary mapping each character to integer ID
#   2) id2char → reverse mapping from ID to character
#
# HOW IT WORKS:
# -------------
# 1. We start with a `set` containing just space " " 
#    because spacing is important in diacritization.
#
# 2. We loop through all sentences and collect EVERY character.
#    Example:
#        sentences = ["ثُمَّ ذَهَبَ", "قال الرجل"]
#    collected characters might be:
#        {" ", "ث", "م", "ذ", "ه", "ب", "ق", "ا", "ل", "ر", "ج"}
#
# 3. We then SORT the characters — sorting keeps vocab ordering fixed
#    so results are reproducible across runs.
#
# 4. We assign a unique numerical ID for each character:
#        char2id = { ' ' : 0, 'ا' : 1, 'ب' : 2, 'ت' : 3, ... }
#
# 5. We finally build reverse mapping:
#        id2char = { 0:' ', 1:'ا', 2:'ب', ... }
#
# WHY IT IS IMPORTANT:
# --------------------
# This vocabulary is what the neural network actually receives.
# Characters must become numbers for Embedding → LSTM → CRF/Transformer.
#
# Example Encoding:
#       word = "ذهب"
#       → ['ذ','ه','ب']
#       → [4, 10, 2]    # after mapping with char2id
#
# Without this step the model CANNOT train or infer.
# =========================================================
def build_char_vocab(sentences):
    """
    Build character <-> id mappings for a list of sentences.

    Args:
        sentences (list[str]): list of cleaned sentences (strings).

    Returns:
        tuple: (char2id, id2char)

    Example:
        >>> sentences = ["ذهب محمد", "قال الرجل"]
        >>> char2id, id2char = build_char_vocab(sentences)
        >>> # Encode a word
        >>> [char2id[c] for c in list("ذهب")]
        [char2id['ذ'], char2id['ه'], char2id['ب']]
    """
    char_set = set(" ")   # include a space character explicitly

    # collect characters from dataset
    for sentence in sentences:
        for ch in sentence:
            char_set.add(ch)

    # sort for stable reproducible indexing
    # encode characters
    char2id = {ch: idx for idx, ch in enumerate(sorted(char_set))}
    # reverse mapping for decoding predictions
    id2char = {idx: ch for ch, idx in char2id.items()}

    return char2id, id2char



# =========================================================
# (2) Diacritic / Haraka Labels
# =========================================================
# You can modify if you want to combine or separate labels.
DIACRITICS = [
    "NONE",   # space or letter without haraka
    "َ",      # Fatha
    "ِ",      # Kasra
    "ُ",      # Damma
    "ّ",      # Shadda
    "ً",      # Tanween Fath
    "ٌ",      # Tanween Damm
    "ٍ",      # Tanween Kasr
    "ْ",      # Sukoon
]

# Mapping between diacritic character (or NONE) and integer label
diac2id = {d: i for i, d in enumerate(DIACRITICS)}
id2diac = {i: d for d, i in diac2id.items()}

# Example:
#   letter 'ذ' with fatha 'َ' -> label id = diac2id['َ']
#   letter 'م' without any diacritic -> label id = diac2id['NONE']



# =========================================================
# (3) Character-level Tokenizer
# =========================================================
def tokenize_chars(sentence: str):
    """
    Split a sentence into its character sequence.

    Args:
        sentence (str): an input string (cleaned, no extra markers).

    Returns:
        list[str]: list of single-character strings.

    Example:
        >>> tokenize_chars("ذهب")
        ['ذ', 'ه', 'ب']
    """
    return list(sentence)



# =========================================================
# (4) Word-level Tokenizer   (optional, useful later)
# =========================================================
def tokenize_words(sentence: str):
    """
    Simple whitespace-based word tokenizer.

    Args:
        sentence (str): input string.

    Returns:
        list[str]: tokens split on whitespace.

    Example:
        >>> tokenize_words("ذهب محمد إلى السوق")
        ['ذهب', 'محمد', 'إلى', 'السوق']
    """
    return sentence.split()



# =========================================================
# (5) Encoding Input (X) + Labels (Y)
# =========================================================
def encode_labeled_sentence(sentence: str):
    """
    Converts a fully-diacritized sentence into:

    X = characters without diacritics (model input)
    Y = diacritic labels as integers (model targets)

    Uses clean.py → split_sentence_into_labeled_chars()
    
    Encode a single fully-diacritized sentence into model inputs and
    target label IDs.

    The function expects a sentence containing diacritics. It uses
    `clean_sentence` and `split_sentence_into_labeled_chars` from
    `clean.py` to produce a sequence of (base_char, diacritic).

    Returns:
        X (list[str]): characters without diacritics (base letters)
        Y (list[int]): integer labels for each character's diacritic

    Example (conceptual):
        sentence = "ذَهَبَ"
        # after split: [('ذ','َ'), ('ه','َ'), ('ب','َ')]
        X = ['ذ', 'ه', 'ب']
        Y = [diac2id['َ'], diac2id['َ'], diac2id['َ']]
    """

    labeled = split_sentence_into_labeled_chars(clean_sentence(sentence))

    # Input text for model (base letters only)
    X = [remove_diacritics(base) for base, _ in labeled]

    # Output labels → convert diacritics to integers; missing diacritic
    # becomes the 'NONE' label.
    Y = [diac2id.get(diac if diac != "" else "NONE") for _, diac in labeled]

    return X, Y



# =========================================================
# (6) Batch Encoding for full dataset
# =========================================================
def encode_dataset(sentences):
    """
    Encode a list of diacritized sentences into lists of inputs and
    labels suitable for training or evaluation.

    Args:
        sentences (list[str]): list of fully-diacritized sentences.

    Returns:
        tuple: (X_all, Y_all) where each is a list of lists.

    Example:
        >>> sentences = ["ذَهَبَ", "قَالَ"]
        >>> X_all, Y_all = encode_dataset(sentences)
        >>> # X_all -> [['ذ','ه','ب'], ['ق','ا','ل']]
        >>> # Y_all -> [[diac2id['َ'], ...], [diac2id['َ'], ...]]
    """
    X_all, Y_all = [], []

    for sentence in sentences:
        X, Y = encode_labeled_sentence(sentence)
        X_all.append(X)
        Y_all.append(Y)

    return X_all, Y_all
