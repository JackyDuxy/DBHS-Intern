"""
Data Quality Analysis for DBHS Training Dataset

Identifies:
1. Dataset size and distribution
2. Duplicate questions
3. Duplicate answers (semantic similarity)
4. Unusually long answers (potential noise)
5. Highly repeated phrases
6. Problematic content (e.g., "Michael Krieger" overrepresentation)
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
import numpy as np

# ============================================================
# Configuration
# ============================================================

DATA_FILE = "train.jsonl"
PHRASE_LENGTH = 5  # Words per phrase
PHRASE_MIN_FREQ = 5  # Flag if phrase appears 5+ times
LONG_ANSWER_THRESHOLD = 1000  # Chars
OVERREP_KEYWORDS = ["Michael", "Krieger", "Dr.", "bell", "period"]

# ============================================================
# Load Data
# ============================================================

print("[*] Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"[✓] Loaded {len(dataset)} examples\n")

# ============================================================
# 1. Basic Statistics
# ============================================================

print("=" * 70)
print("1. DATASET STATISTICS")
print("=" * 70)

total_examples = len(dataset)
total_questions = sum(1 for ex in dataset if ex.get("messages") and len(ex["messages"]) > 0)
total_answers = sum(1 for ex in dataset if ex.get("messages") and len(ex["messages"]) > 1)

print(f"Total examples: {total_examples}")
print(f"Examples with questions: {total_questions}")
print(f"Examples with Q&A pairs: {total_answers}")

question_lengths = []
answer_lengths = []
for ex in dataset:
    messages = ex.get("messages", [])
    if len(messages) > 0:
        question_lengths.append(len(messages[0].get("content", "")))
    if len(messages) > 1:
        answer_lengths.append(len(messages[1].get("content", "")))

print(f"\nQuestion lengths (chars):")
print(f"  Min: {min(question_lengths) if question_lengths else 'N/A'}")
print(f"  Max: {max(question_lengths) if question_lengths else 'N/A'}")
print(f"  Mean: {np.mean(question_lengths) if question_lengths else 'N/A':.0f}")
print(f"  Median: {np.median(question_lengths) if question_lengths else 'N/A':.0f}")

print(f"\nAnswer lengths (chars):")
print(f"  Min: {min(answer_lengths) if answer_lengths else 'N/A'}")
print(f"  Max: {max(answer_lengths) if answer_lengths else 'N/A'}")
print(f"  Mean: {np.mean(answer_lengths) if answer_lengths else 'N/A':.0f}")
print(f"  Median: {np.median(answer_lengths) if answer_lengths else 'N/A':.0f}")

# ============================================================
# 2. Duplicate Questions
# ============================================================

print("\n" + "=" * 70)
print("2. DUPLICATE QUESTIONS")
print("=" * 70)

question_counts = defaultdict(int)
question_indices = defaultdict(list)

for i, ex in enumerate(dataset):
    messages = ex.get("messages", [])
    if messages and len(messages) > 0:
        q = messages[0].get("content", "").strip().lower()
        question_counts[q] += 1
        question_indices[q].append(i)

exact_duplicates = sum(1 for count in question_counts.values() if count > 1)
duplicate_pairs = sum(count - 1 for count in question_counts.values() if count > 1)

print(f"Exact duplicate questions: {exact_duplicates}")
print(f"Total duplicate Q&A pairs: {duplicate_pairs}")
print(f"Deduplication would reduce dataset by: {duplicate_pairs} examples")

if exact_duplicates > 0:
    print(f"\nTop 5 most repeated questions:")
    top_dupes = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for q, count in top_dupes:
        if count > 1:
            print(f"  [{count}x] {q[:70]}")

# ============================================================
# 3. Unusually Long Answers
# ============================================================

print("\n" + "=" * 70)
print("3. UNUSUALLY LONG ANSWERS")
print("=" * 70)

long_answers = []
for i, ex in enumerate(dataset):
    messages = ex.get("messages", [])
    if len(messages) > 1:
        answer = messages[1].get("content", "")
        if len(answer) > LONG_ANSWER_THRESHOLD:
            long_answers.append((i, len(answer), answer[:100]))

print(f"Answers > {LONG_ANSWER_THRESHOLD} chars: {len(long_answers)}")
if long_answers:
    print(f"\nTop 5 longest answers:")
    for idx, length, snippet in sorted(long_answers, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  [Idx {idx}, {length} chars] {snippet}...")

# ============================================================
# 4. Repeated Phrases (n-grams)
# ============================================================

print("\n" + "=" * 70)
print("4. REPEATED PHRASES (Most Common N-grams)")
print("=" * 70)

def extract_phrases(text: str, n: int = PHRASE_LENGTH) -> List[str]:
    """Extract n-word phrases from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

all_phrases = []
for ex in dataset:
    messages = ex.get("messages", [])
    if len(messages) > 1:
        answer = messages[1].get("content", "")
        all_phrases.extend(extract_phrases(answer, PHRASE_LENGTH))

phrase_counts = Counter(all_phrases)
overrep_phrases = [(phrase, count) for phrase, count in phrase_counts.items() 
                   if count >= PHRASE_MIN_FREQ and len(phrase) > 10]

print(f"Total unique {PHRASE_LENGTH}-word phrases: {len(phrase_counts)}")
print(f"Phrases appearing {PHRASE_MIN_FREQ}+ times: {len(overrep_phrases)}")

if overrep_phrases:
    print(f"\nTop 10 most repeated phrases:")
    for phrase, count in sorted(overrep_phrases, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  [{count}x] {phrase}")

# ============================================================
# 5. Keyword Frequency Analysis
# ============================================================

print("\n" + "=" * 70)
print("5. KEYWORD FREQUENCY (Potential Content Bias)")
print("=" * 70)

keyword_occurrences = defaultdict(int)
keyword_examples = defaultdict(list)

for i, ex in enumerate(dataset):
    full_text = json.dumps(ex).lower()
    for keyword in OVERREP_KEYWORDS:
        if keyword.lower() in full_text:
            keyword_occurrences[keyword] += 1
            if len(keyword_examples[keyword]) < 3:
                keyword_examples[keyword].append(i)

print(f"\nKeyword frequency in dataset:")
for keyword in OVERREP_KEYWORDS:
    count = keyword_occurrences[keyword]
    pct = (count / len(dataset)) * 100
    print(f"  '{keyword}': {count} examples ({pct:.1f}%)")

# ============================================================
# 6. Sample Examples
# ============================================================

print("\n" + "=" * 70)
print("6. RANDOM SAMPLES")
print("=" * 70)

import random
random.seed(42)
sample_indices = random.sample(range(len(dataset)), min(3, len(dataset)))

for i, idx in enumerate(sample_indices, 1):
    ex = dataset[idx]
    messages = ex.get("messages", [])
    print(f"\n[Sample {i}, Index {idx}]")
    if len(messages) > 0:
        q = messages[0].get("content", "")[:100]
        print(f"  Q: {q}")
    if len(messages) > 1:
        a = messages[1].get("content", "")[:150]
        print(f"  A: {a}...")

# ============================================================
# 7. Recommendations
# ============================================================

print("\n" + "=" * 70)
print("7. RECOMMENDATIONS")
print("=" * 70)

recommendations = []

if duplicate_pairs > 0:
    recommendations.append(
        f"✗ Remove {duplicate_pairs} exact duplicate Q&A pairs to reduce redundancy"
    )

if len(long_answers) > len(dataset) * 0.1:
    recommendations.append(
        f"✗ {len(long_answers)} answers exceed {LONG_ANSWER_THRESHOLD} chars; consider truncating"
    )

if len(overrep_phrases) > 50:
    recommendations.append(
        f"✗ High n-gram repetition suggests significant content redundancy"
    )

if keyword_occurrences.get("Michael", 0) > len(dataset) * 0.05:
    recommendations.append(
        "✗ 'Michael' (likely 'Michael Krieger') appears >5% of examples; content is heavily skewed"
    )

if not recommendations:
    recommendations.append("✓ Dataset quality is acceptable")

for rec in recommendations:
    print(f"  {rec}")

print("\n" + "=" * 70)
print("[✓] Analysis complete")
print("=" * 70)
