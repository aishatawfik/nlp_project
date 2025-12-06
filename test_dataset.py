# test_dataset.py
# ---------------------------------------------------------
# Tests dataset loader + vocab builder
# ---------------------------------------------------------

from test_dataset import load_dataset, prepare_vocab
from preprocessing.clean import clean_sentence, split_sentence_into_labeled_chars
from preprocessing.tokenization import encode_labeled_sentence, DIACRITIC_TO_ID


print("\n==============================")
print("🔍 TEST — CLEAN + LABEL PIPELINE")
print("==============================")

sample = "وَلَوْ جَمَعَ ثُمَّ عَلِمَ"
print("\nOriginal:", sample)
print("Preprocessed:", clean_sentence(sample))
print("Labeled:", split_sentence_into_labeled_chars(sample))


print("\n==============================")
print("🔍 TEST — DATASET LOAD (first 3 samples)")
print("==============================")

X_data, Y_data = load_dataset("data/train.txt", limit=3)

for i in range(3):
    print(f"\nSample {i+1}")
    print("X chars:", X_data[i])
    print("Y labels:", Y_data[i])


print("\n==============================")
print("🔍 TEST — VOCAB GENERATION")
print("==============================")

char2id, id2char = prepare_vocab(X_data)

print("Vocabulary size:", len(char2id))
print("Sample from vocab:", list(char2id.items())[:10])


print("\n==============================")
print("🔍 TEST — ENCODE LABELED SENTENCE")
print("==============================")

text = "ذَهَبَ مُحَمَّدٌ"
X, Y = encode_labeled_sentence(text)

print("X:", X)
print("Y:", Y)

# Pretty display for Arabic direction
def pretty_arab(seq):
    return "".join(seq[::-1])

print("\nReadable:", pretty_arab(X))
print("Diacritics:", [list(DIACRITIC_TO_ID.values())[y] for y in Y])
