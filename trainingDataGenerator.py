import json
import re
import random
from pathlib import Path

INPUT_FILE = "dbhs_chunks.json"
OUTPUT_FILE = "train.jsonl"

TARGET_QAS_PER_CHUNK = 40

random.seed(42)


QUESTION_TEMPLATES = [
    "What is {title}?",
    "Can you explain {title}?",
    "Tell me about {title}.",
    "Provide information about {title}.",
    "What should students know about {title}?",
    "Summarize {title}.",
    "What services are provided by {title}?",
    "What information is available regarding {title}?",
    "How does {title} work?",
    "Why is {title} important?",
    "What are the requirements related to {title}?",
    "What procedures are associated with {title}?",
    "What are the key details about {title}?",
    "How can a student use {title}?",
    "What resources does {title} offer?",
]

CATEGORY_TEMPLATES = {
    "attendance": [
        "How do I report an absence?",
        "What are the attendance procedures?",
        "What attendance policies should students follow?",
        "Who should I contact regarding attendance issues?"
    ],

    "guidance": [
        "How do I request a transcript?",
        "What counseling resources are available?",
        "How can I contact my counselor?",
        "What guidance services are offered?"
    ],

    "athletics": [
        "What athletic opportunities are available?",
        "How can students participate in sports?",
        "Tell me about athletics at DBHS.",
        "What athletic programs are offered?"
    ],

    "clubs": [
        "What clubs are available at DBHS?",
        "How can I join a club?",
        "What extracurricular opportunities exist?",
        "Tell me about student clubs."
    ],

    "bell_schedule": [
        "What is the school bell schedule?",
        "When does school start?",
        "When does school end?",
        "What are the period times?"
    ],

    "enrollment": [
        "How do I enroll at DBHS?",
        "What enrollment documents are required?",
        "What are the registration procedures?",
        "How does student enrollment work?"
    ]
}


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def build_summary_answer(content):
    content = re.sub(r'\s+', ' ', content)
    return content[:1200]


def build_fact_questions(title, sentences):
    qas = []

    for sentence in sentences[:10]:

        sentence = sentence.strip()

        if len(sentence) < 20:
            continue

        qas.append({
            "question": f"What information is provided about {title}?",
            "answer": sentence
        })

        qas.append({
            "question": f"Can you tell me something about {title}?",
            "answer": sentence
        })

    return qas


def build_template_questions(title, content):

    answer = build_summary_answer(content)

    qas = []

    for template in QUESTION_TEMPLATES:
        qas.append({
            "question": template.format(title=title),
            "answer": answer
        })

    return qas


def build_category_questions(category, content):

    answer = build_summary_answer(content)

    qas = []

    templates = CATEGORY_TEMPLATES.get(category, [])

    for t in templates:
        qas.append({
            "question": t,
            "answer": answer
        })

    return qas


def build_keyword_questions(title, content):

    words = re.findall(r'\b[A-Za-z]{5,}\b', content)

    words = list(dict.fromkeys(words))

    qas = []

    for word in words[:15]:

        qas.append({
            "question": f"How is {word} related to {title}?",
            "answer": build_summary_answer(content)
        })

    return qas


def build_summary_questions(title, content):

    answer = build_summary_answer(content)

    templates = [
        f"Give me a detailed overview of {title}.",
        f"What should a new student know about {title}?",
        f"What are the most important points about {title}?",
        f"Explain {title} in detail.",
        f"Provide a complete summary of {title}.",
        f"What does the DBHS knowledge base say about {title}?"
    ]

    return [
        {
            "question": q,
            "answer": answer
        }
        for q in templates
    ]


def create_qa_pairs(chunk):

    title = chunk.get("title", "Unknown")
    category = chunk.get("category", "general")
    content = chunk.get("content", "")

    sentences = split_sentences(content)

    qa_pairs = []

    qa_pairs.extend(
        build_template_questions(title, content)
    )

    qa_pairs.extend(
        build_category_questions(category, content)
    )

    qa_pairs.extend(
        build_fact_questions(title, sentences)
    )

    qa_pairs.extend(
        build_keyword_questions(title, content)
    )

    qa_pairs.extend(
        build_summary_questions(title, content)
    )

    random.shuffle(qa_pairs)

    return qa_pairs[:TARGET_QAS_PER_CHUNK]


def to_chat_format(question, answer):

    return {
        "messages": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    dataset = []

    for chunk in chunks:

        qas = create_qa_pairs(chunk)

        for qa in qas:

            dataset.append(
                to_chat_format(
                    qa["question"],
                    qa["answer"]
                )
            )

    random.shuffle(dataset)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for item in dataset:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Generated {len(dataset)} QA pairs"
    )

    print(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
