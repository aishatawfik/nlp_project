# dataset.py
# ---------------------------------------------------------
# Dataset builder for Arabic diacritization
# Uses all functions from clean.py + tokenization.py
# ---------------------------------------------------------

from clean import clean_sentence, split_sentence_into_labeled_chars
from tokenization import DIACRITIC_TO_ID, build_char_vocab


def load_dataset(path, limit=None):
    """
    Loads text → cleans it → extracts base + diacritics
    Returns X_characters, Y_labels
    """

    X_data = []
    Y_data = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if limit:
        lines = lines[:limit]    # for testing small samples

    for line in lines:

        clean = clean_sentence(line)  # ⬅ using clean.py correctly
        labeled = split_sentence_into_labeled_chars(clean)  # ⬅ uses clean+split

        X = []
        Y = []

        for base, dia in labeled:
            X.append(base)
            Y.append(DIACRITIC_TO_ID.get(dia, 0))  # NONE = 0

        X_data.append(X)
        Y_data.append(Y)

    return X_data, Y_data



def prepare_vocab(X_data):
    """
    Uses tokenizer function to build char vocab
    """

    sentences = ["".join(seq) for seq in X_data]   # flatten
    return build_char_vocab(sentences)
