import re
import json
from pathlib import Path

INPUT_FILE = "DIAMON_1.MD"
OUTPUT_FILE = "dbhs_chunks.json"

# Categories based on DBHS structure
CATEGORY_MAP = {
    "Overview": "overview",
    "Contact": "contact",
    "Enrollment": "enrollment",
    "Administration": "administration",
    "Guidance": "guidance",
    "Academics": "academics",
    "Advanced Placement": "ap_program",
    "International Baccalaureate": "ib_program",
    "Brahma Tech": "engineering",
    "Career Technical Education": "cte",
    "Mathematics": "mathematics",
    "Chinese Language": "languages",
    "English Learner": "english_learner",
    "Graduation": "graduation",
    "Arts": "arts",
    "Activities": "activities",
    "Clubs": "clubs",
    "Athletics": "athletics",
    "Bell Schedules": "bell_schedule",
    "Calendar": "calendar",
    "Summer School": "summer_school",
    "Attendance": "attendance",
    "Health": "health"
}


def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def infer_category(title):
    for key, value in CATEGORY_MAP.items():
        if key.lower() in title.lower():
            return value
    return "general"


def split_sections(markdown):
    """
    Split using #, ## headers.
    """

    pattern = r"(?=^# .+|^## .+)"
    sections = re.split(pattern, markdown, flags=re.MULTILINE)

    cleaned = []

    for sec in sections:
        sec = sec.strip()

        if not sec:
            continue

        lines = sec.splitlines()

        title = lines[0].replace("#", "").strip()

        content = "\n".join(lines[1:]).strip()

        if len(content) < 50:
            continue

        cleaned.append({
            "title": title,
            "content": content
        })

    return cleaned


def compress_content(text):
    """
    Light compression.

    Removes:
      - duplicate whitespace
      - repetitive source metadata
      - access dates
    """

    text = re.sub(r"\*\*Source:\*\*.*", "", text)
    text = re.sub(r"\*\*Last accessed:\*\*.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_large_section(text, max_words=800):

    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


def build_chunks(markdown):

    sections = split_sections(markdown)

    output = []

    chunk_id = 0

    for sec in sections:

        title = sec["title"]

        category = infer_category(title)

        content = compress_content(sec["content"])

        pieces = chunk_large_section(content)

        for idx, piece in enumerate(pieces):

            output.append({
                "id": f"dbhs_{chunk_id}",
                "category": category,
                "title": title,
                "chunk_index": idx,
                "content": clean_text(piece)
            })

            chunk_id += 1

    return output


def main():

    markdown = Path(INPUT_FILE).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    chunks = build_chunks(markdown)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()