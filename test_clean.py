# test_clean.py
# ---------------------------------------------------------
# Test script for preprocessing.clean functions
# ---------------------------------------------------------

from preprocessing.clean import (
    clean_sentence,
    split_word_into_labeled_chars,
    split_sentence_into_labeled_chars
)

# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------
train_path = "data/train.txt"

with open(train_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

sample = lines[0]

print("=== TESTING clean.py ===\n")
print("Original:")
print(sample)
print("\nCleaned:")
print(clean_sentence(sample))

# ---------------------------------------------------------
# Test labeled word splitting
# ---------------------------------------------------------
print("\n=== Testing split_word_into_labeled_chars ===")
first_word = sample.split()[0]
print(f"Word: {first_word}")
print(split_word_into_labeled_chars(first_word))

# ---------------------------------------------------------
# Test sentence-level labeled extraction
# ---------------------------------------------------------
print("\n=== Testing split_sentence_into_labeled_chars ===")
labeled = split_sentence_into_labeled_chars(sample)

print("First 20 labeled characters:")
for base, dia in labeled[:20]:
    print(f"{base} → {dia}")
