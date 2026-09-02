import re
import pandas as pd


def extract_glosses(sentence, glosses):
    """
    Tìm tất cả gloss xuất hiện trong một câu tiếng Việt.

    - Gloss có thể gồm 1 hoặc nhiều từ.
    - Không phân biệt chữ hoa/chữ thường.
    - Xử lý dấu câu.
    - Tránh match một phần của từ.
    """
    sentence = str(sentence).lower().strip()

    matched = []

    for gloss in glosses:
        gloss_clean = str(gloss).lower().strip()

        if not gloss_clean:
            continue

        # Tìm nguyên từ / nguyên cụm từ
        pattern = r"(?<!\w)" + re.escape(gloss_clean) + r"(?!\w)"

        if re.search(pattern, sentence):
            matched.append(gloss)

    return matched


def extract_from_lists(glosses, sentences):
    """
    Trả về DataFrame gồm:
        sentence
        matched_glosses
        num_glosses
    """
    results = []

    for sentence in sentences:
        matched = extract_glosses(sentence, glosses)

        results.append(
            {
                "sentence": sentence,
                "matched_glosses": matched,
                "num_glosses": len(matched),
            }
        )

    return pd.DataFrame(results)


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    VOCAB_FILE = "vocab_normal.csv"
    OUTPUT_FILE = "response.txt"

    vocab_df = pd.read_csv(VOCAB_FILE)
    gloss_to_id = {}
    for _, row in vocab_df.iterrows():
        gloss_id = row["id"]
        gloss = str(row["gloss"]).strip().lower()
        gloss_to_id[gloss] = gloss_id

    glosses = list(gloss_to_id.keys())
    glosses = sorted(glosses, key=len, reverse=True)

    sentences = read_txt(OUTPUT_FILE)

    df = extract_from_lists(glosses, sentences)
    df.to_csv(
        "sentence_glosses.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nĐã lưu: sentence_glosses.csv")
