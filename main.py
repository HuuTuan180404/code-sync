import csv
import string

# Read vocab.csv and split into two files
normal_words = []
words_with_punctuation = []

# Define punctuation to check (dấu câu)
punctuation = set(string.punctuation)

with open("vocab.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Check if gloss contains any punctuation
        has_punctuation = any(char in punctuation for char in row["gloss"])
        if has_punctuation:
            words_with_punctuation.append(row)
        else:
            normal_words.append(row)

# Write normal words to file
with open("vocab_normal.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "gloss"])
    writer.writeheader()
    writer.writerows(normal_words)

# Write words with punctuation to file
with open("vocab_with_punctuation.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "gloss"])
    writer.writeheader()
    writer.writerows(words_with_punctuation)

print(f"✓ Normal words (chỉ chữ): {len(normal_words)} entries → vocab_normal.csv")
print(
    f"✓ Words with punctuation (có dấu): {len(words_with_punctuation)} entries → vocab_with_punctuation.csv"
)
print(f"✓ Total: {len(normal_words) + len(words_with_punctuation)} entries")
