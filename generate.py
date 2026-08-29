import re
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-8B"

VOCAB_FILE = "vocab.csv"
OUTPUT_FILE = "generated_sentences.csv"

# TEST MODE: sinh chỉ 5 câu thôi
TEST_MODE = False

# Mỗi gloss sequence sinh bao nhiêu câu
NUM_SENTENCES_PER_GLOSS = 3

# Số candidate sinh trong mỗi lần generate
NUM_RETURN_SEQUENCES = 3

MAX_NEW_TOKENS = 80

TEMPERATURE = 0.8
TOP_P = 0.9


# ============================================================
# STOP / FUNCTION WORDS
# ============================================================

STOP_WORDS = {
    # articles / quantifiers
    "một",
    "những",
    "các",
    "mọi",
    "mỗi",
    "này",
    "kia",
    "đó",
    # aspect / tense
    "đang",
    "đã",
    "sẽ",
    "vừa",
    "mới",
    "từng",
    # prepositions
    "ở",
    "tại",
    "trong",
    "ngoài",
    "trên",
    "dưới",
    "với",
    "cho",
    "từ",
    "đến",
    "về",
    "bằng",
    # conjunctions
    "và",
    "hoặc",
    "hay",
    "nhưng",
    "mà",
    "vì",
    "nên",
    "nếu",
    "thì",
    # auxiliary / function
    "là",
    "được",
    "bị",
    "có",
    "không",
    # pronouns
    "tôi",
    "ta",
    "chúng",
    "chúng tôi",
    "chúng ta",
    "bạn",
    "anh",
    "chị",
    "em",
    "ông",
    "bà",
    # particles
    "rất",
    "cũng",
    "chỉ",
    "đều",
    "lại",
    "còn",
    "đã",
}


# ============================================================
# LOAD VOCABULARY
# ============================================================

print("Loading vocabulary...")

vocab_df = pd.read_csv(VOCAB_FILE)

# Kiểm tra column
required_columns = {"id", "gloss"}

if not required_columns.issubset(vocab_df.columns):
    raise ValueError(
        f"CSV phải có 2 cột: id, gloss. " f"Hiện tại: {list(vocab_df.columns)}"
    )

gloss_to_id = {}

for _, row in vocab_df.iterrows():

    gloss_id = row["id"]
    gloss = str(row["gloss"]).strip().lower()

    gloss_to_id[gloss] = gloss_id


# ============================================================
# LOAD QWEN3
# ============================================================

print("Loading Qwen3-8B...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)

model.eval()

print("Model loaded.")


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(text):
    """
    Chuẩn hóa text để so sánh.
    """

    text = text.lower().strip()

    # bỏ punctuation
    text = re.sub(r"[.,!?;:\"'“”‘’()\[\]{}]", " ", text)

    # nhiều space -> 1 space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TOKENIZE TIẾNG VIỆT
# ============================================================


def tokenize(text):

    text = normalize_text(text)

    return text.split()


# ============================================================
# MATCH GLOSS
# ============================================================


def find_glosses_in_sentence(sentence, glosses):
    """
    Tìm các gloss xuất hiện trong câu.

    Quan trọng:
    Gloss có thể gồm nhiều từ.

    Ví dụ:
        "học sinh"

    được xem là một gloss.
    """

    sentence_norm = normalize_text(sentence)

    matches = []

    for gloss in glosses:

        gloss_norm = normalize_text(gloss)

        # tìm vị trí xuất hiện
        start = sentence_norm.find(gloss_norm)

        if start != -1:

            matches.append((start, gloss))

    # sắp xếp theo vị trí xuất hiện
    matches.sort(key=lambda x: x[0])

    return [gloss for _, gloss in matches]


# ============================================================
# EXTRACT STOP WORDS
# ============================================================


def extract_stop_words(sentence, glosses):
    """
    Tìm các stop/function words xuất hiện trong sentence
    nhưng không thuộc gloss.
    """

    sentence_words = tokenize(sentence)

    # Tạo tập các từ thuộc gloss
    gloss_words = set()

    for gloss in glosses:

        for word in tokenize(gloss):

            gloss_words.add(word)

    added_stop_words = []

    for word in sentence_words:

        if word in gloss_words:
            continue

        if word in STOP_WORDS:

            if word not in added_stop_words:
                added_stop_words.append(word)

    return added_stop_words


# ============================================================
# VALIDATE SENTENCE
# ============================================================


def validate_sentence(sentence, glosses):
    """
    Kiểm tra sentence có sử dụng:
    1. Tất cả gloss
    2. Không có content word mới
    """

    sentence_norm = normalize_text(sentence)

    # --------------------------------------------------------
    # Check tất cả gloss có xuất hiện không
    # --------------------------------------------------------

    for gloss in glosses:

        gloss_norm = normalize_text(gloss)

        if gloss_norm not in sentence_norm:

            return False

    # --------------------------------------------------------
    # Tokenize sentence
    # --------------------------------------------------------

    sentence_words = tokenize(sentence)

    # --------------------------------------------------------
    # Các từ thuộc gloss
    # --------------------------------------------------------

    gloss_words = set()

    for gloss in glosses:

        for word in tokenize(gloss):

            gloss_words.add(word)

    # --------------------------------------------------------
    # Tìm từ không hợp lệ
    # --------------------------------------------------------

    invalid_words = []

    for word in sentence_words:

        if word in gloss_words:
            continue

        if word in STOP_WORDS:
            continue

        invalid_words.append(word)

    # --------------------------------------------------------
    # Nếu có content word mới -> invalid
    # --------------------------------------------------------

    if len(invalid_words) > 0:

        return False

    return True


# ============================================================
# GENERATE SENTENCES
# ============================================================


def generate_sentences(gloss, num_sentences):
    """
    Sinh các câu tiếng Việt tự nhiên từ một gloss.
    """

    prompt = f"""
Bạn là hệ thống sinh dữ liệu cho bài toán nhận dạng ngôn ngữ ký hiệu tiếng Việt.

Gloss bắt buộc:
{gloss}

Hãy tạo {num_sentences} câu tiếng Việt tự nhiên có chứa từ/cụm từ "{gloss}".

QUY TẮC:
1. Mỗi câu phải chứa chính xác từ/cụm từ "{gloss}" hoặc cách dùng tự nhiên của nó.
2. Câu phải có nghĩa rõ ràng và tự nhiên trong tiếng Việt.
3. Có thể thêm các từ khác để tạo thành câu hoàn chỉnh.
4. Không giải thích.
5. Không đánh số.
6. Mỗi câu nằm trên một dòng riêng.

Chỉ trả về các câu tiếng Việt.
"""

    messages = [
        {"role": "system", "content": "Bạn là hệ thống sinh câu tiếng Việt từ gloss."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=num_sentences,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs.input_ids.shape[1]

    sentences = []

    for output in outputs:

        generated_tokens = output[input_length:]

        sentence = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        sentence = sentence.strip()

        # Nếu model trả nhiều dòng, lấy từng dòng
        for line in sentence.split("\n"):

            line = line.strip()

            # bỏ đánh số nếu có
            line = re.sub(r"^\d+[\.\)]\s*", "", line)

            if line:
                sentences.append(line)

    return sentences[:num_sentences]


# ============================================================
# MAIN
# ============================================================


def main():
    results = []
    sentence_id = 1

    print("Loading vocabulary...")
    vocab_df = pd.read_csv(VOCAB_FILE)

    if not required_columns.issubset(vocab_df.columns):
        raise ValueError(
            f"CSV phải có 2 cột: id, gloss. " f"Hiện tại: {list(vocab_df.columns)}"
        )

    print(f"Number of glosses: {len(vocab_df)}")
    saved_count = 0
    for _, row in tqdm(vocab_df.iterrows(), total=len(vocab_df), desc="Generating"):

        gloss_id = str(row["id"]).strip()
        gloss_string = str(row["gloss"]).strip()

        if not gloss_string:
            continue

        print(f"\nGloss: {gloss_id} | {gloss_string}")

        candidates = generate_sentences(gloss_string, NUM_RETURN_SEQUENCES)

        print(f"Generated {len(candidates)} candidates")

        for sentence in candidates:
            sentence = sentence.strip()
            if not sentence:
                continue

            results.append(
                {
                    "id": gloss_id,
                    "text": sentence,
                    "gloss": gloss_string,
                }
            )
            
            sentence_id += 1
            saved_count += 1

            if saved_count >= NUM_SENTENCES_PER_GLOSS:
                break

        # ------------------------------------------------
        # Tìm gloss theo thứ tự xuất hiện
        # ------------------------------------------------

        # used_glosses = find_glosses_in_sentence(sentence, glosses)

        # ------------------------------------------------
        # Kiểm tra lại
        # ------------------------------------------------

        # if len(used_glosses) != len(glosses):
        #     continue

        # ------------------------------------------------
        # Convert gloss -> ID
        # ------------------------------------------------

        # gloss_ids = []

        # for gloss in used_glosses:

        #     gloss_norm = normalize_text(gloss)

        #     if gloss_norm not in gloss_to_id:
        #         continue

        #     gloss_ids.append(str(gloss_to_id[gloss_norm]))

        # ------------------------------------------------
        # Stop words
        # ------------------------------------------------

        # added_stop_words = extract_stop_words(sentence, glosses)

        # ------------------------------------------------
        # Save
        # ------------------------------------------------

        # results.append(
        #     {
        #         "id": sentence_id,
        #         "text": sentence,
        #         "gloss": " ".join(gloss_ids),
        #         "stop word": ", ".join(added_stop_words),
        #     }
        # )

        # sentence_id += 1

        # ------------------------------------------------
        # Đủ số lượng chưa?
        # ------------------------------------------------

        # if (
        #     len([r for r in results if r["gloss"] == " ".join(gloss_ids)])
        #     >= NUM_SENTENCES_PER_GLOSS
        # ):

        #     break

    # ========================================================
    # SAVE CSV
    # ========================================================

    output_df = pd.DataFrame(
        results,
        columns=[
            "id",
            "text",
            "gloss",
            "stop word",
        ],
    )

    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Number of sentences: {len(output_df)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
