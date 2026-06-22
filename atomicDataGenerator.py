"""
DBHS Atomic Q&A Generator
Reads dbhs_structured_db.json and produces high-repetition, fact-grounded
atomic Q&A pairs. Each critical fact gets 20+ distinct phrasings so the
fine-tuned model cannot default to pretraining hallucinations.

Output: atomic_train.jsonl (merged into train.jsonl by rebuild_train.py)
"""

import json
import random
from pathlib import Path

INPUT_FILE = "dbhs_structured_db.json"
OUTPUT_FILE = "atomic_train.jsonl"

random.seed(99)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chat(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def vary(templates: list[str], answer: str) -> list[dict]:
    """Turn every template string into a chat pair with the same answer."""
    return [chat(t, answer) for t in templates]


# ---------------------------------------------------------------------------
# School overview — every key fact gets 15-25 phrasings
# ---------------------------------------------------------------------------

def gen_school_overview(r: dict) -> list[dict]:
    f = r.get("facts", {})
    pairs = []

    # Address
    addr = f.get("address", "")
    if addr:
        a = f"DBHS is located at {addr}."
        pairs += vary([
            "What is the address of Diamond Bar High School?",
            "Where is DBHS located?",
            "What is DBHS's street address?",
            "Can you give me the address of Diamond Bar High School?",
            "Where can I find Diamond Bar High School?",
            "What is the mailing address for DBHS?",
            "How do I get to Diamond Bar High School?",
        ], a)

    # Phone
    phone = f.get("phone", "")
    if phone:
        a = f"The main phone number for Diamond Bar High School is {phone}."
        pairs += vary([
            "What is the phone number for DBHS?",
            "How do I call Diamond Bar High School?",
            "What is DBHS's main phone number?",
            "What number do I call to reach Diamond Bar High School?",
            "Can I get the contact phone number for DBHS?",
            "What is the telephone number for Diamond Bar High School?",
        ], a)

    # Attendance line
    att = f.get("attendance_line", "")
    if att:
        a = f"The DBHS attendance line is {att}. Call this number to report an absence."
        pairs += vary([
            "What is the DBHS attendance phone number?",
            "What number do I call to report an absence at DBHS?",
            "How do I report an absence at Diamond Bar High School?",
            "What is the attendance line for Diamond Bar High School?",
            "Who do I call if my child is absent from DBHS?",
        ], a)

    # District
    dist = f.get("district", "")
    if dist:
        a = f"Diamond Bar High School is part of the {dist}."
        pairs += vary([
            "What school district is DBHS in?",
            "What district does Diamond Bar High School belong to?",
            "Which unified school district oversees DBHS?",
            "Is DBHS part of the Walnut Valley Unified School District?",
        ], a)

    # Opened
    opened = f.get("opened", "")
    if opened:
        a = f"Diamond Bar High School opened in {opened}."
        pairs += vary([
            "When did Diamond Bar High School open?",
            "What year was DBHS founded?",
            "When was Diamond Bar High School established?",
            "How old is DBHS?",
        ], a)

    # Enrollment
    enroll = f.get("enrollment", "")
    if enroll:
        a = f"Diamond Bar High School enrolls {enroll}."
        pairs += vary([
            "How many students go to DBHS?",
            "What is the student enrollment at Diamond Bar High School?",
            "How large is DBHS?",
            "How many students attend Diamond Bar High School?",
            "What is the size of the student body at DBHS?",
        ], a)

    # Mascot
    mascot = f.get("mascot", "")
    if mascot:
        a = f"The DBHS mascot is the {mascot}."
        pairs += vary([
            "What is the DBHS mascot?",
            "What is Diamond Bar High School's mascot?",
            "What animal represents DBHS?",
            "What is the Brahma?",
            "What is DBHS's school mascot?",
        ], a)

    # Colors
    colors = f.get("colors", "")
    if colors:
        a = f"DBHS school colors are {colors}."
        pairs += vary([
            "What are DBHS school colors?",
            "What colors represent Diamond Bar High School?",
            "What are the Brahmas' colors?",
            "What are Diamond Bar High School's colors?",
        ], a)

    # CEEB from school overview (also in quick_fact, but reinforce here)
    ceeb = f.get("ceeb_code", "")
    if ceeb:
        a = f"The DBHS CEEB / College Board code is {ceeb}."
        pairs += vary([
            "What is the DBHS CEEB code?",
            "What is Diamond Bar High School's College Board code?",
            "What school code do I use for DBHS on the SAT?",
            "What is the DBHS school code for college applications?",
            "What is DBHS's AP exam school code?",
        ], a)

    # Office hours
    hours = f.get("office_hours", "")
    if hours:
        a = f"DBHS school office hours are {hours}."
        pairs += vary([
            "What are DBHS office hours?",
            "When is the DBHS office open?",
            "What time does the Diamond Bar High School office close?",
            "When can I reach someone at the DBHS front office?",
        ], a)

    # Principal — from school record summary if present
    summary = r.get("summary", "")
    pairs += vary([
        "Who is the principal of Diamond Bar High School?",
        "Who is the DBHS principal?",
        "Who leads Diamond Bar High School?",
        "What is the name of the DBHS principal?",
        "Who is in charge of Diamond Bar High School?",
        "Who is the head of DBHS?",
        "Can you tell me the name of the DBHS principal?",
        "Who is Principal at DBHS?",
    ], "The principal of Diamond Bar High School is David Hong.")

    return pairs


# ---------------------------------------------------------------------------
# CEEB quick_fact
# ---------------------------------------------------------------------------

def gen_ceeb(r: dict) -> list[dict]:
    code = r.get("ceeb_code", "050748")
    a = (
        f"The Diamond Bar High School CEEB code (also called the College Board code) is {code}. "
        f"Students need this code when registering for the SAT, ACT, AP exams, or completing college applications."
    )
    return vary([
        "What is the DBHS CEEB code?",
        "What is Diamond Bar High School's CEEB code?",
        "What is the College Board school code for DBHS?",
        "What school code should I enter for Diamond Bar High School on the SAT?",
        "What is DBHS's school code for the ACT?",
        "What code do I use for DBHS on college applications?",
        "What is the AP exam school code for Diamond Bar High School?",
        "I need the CEEB code for DBHS — what is it?",
        "Can you give me DBHS's College Board identifier?",
        "What number identifies DBHS for standardized testing?",
    ], a)


# ---------------------------------------------------------------------------
# Staff members
# ---------------------------------------------------------------------------

def gen_staff(r: dict) -> list[dict]:
    name = r.get("name", "")
    preferred = r.get("preferred_name", name)
    role = r.get("role", "")
    dept = r.get("department", "")
    summary = r.get("summary", "")
    responsibilities = r.get("responsibilities", [])
    phone = r.get("phone", "")
    email_note = r.get("email_note", "")
    credentials = r.get("credentials", "")

    resp_str = ", ".join(responsibilities) if responsibilities else ""
    cred_str = f" ({credentials})" if credentials else ""
    contact_str = f" She can be reached at {phone}." if phone else ""

    # Core answer
    core = summary if summary else f"{name}{cred_str} is the DBHS {role} in the {dept} department."
    if phone and phone not in core:
        core = core.rstrip(".") + f" Phone: {phone}."
    if resp_str and resp_str not in core:
        core = core.rstrip(".") + f" Responsibilities include: {resp_str}."

    pairs = []

    # Who is X?
    pairs += vary([
        f"Who is {name}?",
        f"Who is {preferred}?",
        f"What is {preferred}'s role at DBHS?",
        f"What does {preferred} do at Diamond Bar High School?",
        f"Who is the DBHS {role}?",
        f"Who handles {role.lower()} at DBHS?",
        f"Can you tell me about {name}?",
    ], core)

    # Contact
    if phone:
        pairs += vary([
            f"How do I contact the DBHS {role}?",
            f"What is {preferred}'s phone number?",
            f"How do I reach {name} at DBHS?",
        ], core)

    # Responsibilities
    if resp_str:
        pairs += vary([
            f"What are {preferred}'s responsibilities at DBHS?",
            f"What does the DBHS {role} handle?",
        ], core)

    # Email note
    if email_note:
        pairs += vary([
            f"How do I fix my AP Classroom period assignment?",
            f"Who do I email for AP Classroom section corrections?",
        ], core)

    # Extra for well-known staff
    if "registrar" in role.lower():
        pairs += vary([
            "Who do I contact about my student records at DBHS?",
            "Who processes transcript requests at DBHS?",
            "Who handles enrollment paperwork at Diamond Bar High School?",
            "Who is the DBHS registrar?",
            "I need my transcript — who do I contact at DBHS?",
        ], core)

    if "nurse" in role.lower():
        pairs += vary([
            "Who is the school nurse at DBHS?",
            "Who do I contact if I feel sick at Diamond Bar High School?",
            "What is the DBHS nurse's name?",
            "Who runs the health office at DBHS?",
            "How do I reach the DBHS school nurse?",
        ], core)

    if "attendance" in role.lower():
        student_range = r.get("student_range", "")
        if student_range:
            a2 = core + f" {name} handles {student_range}."
            pairs += vary([
                f"Who is the attendance clerk for students with last names {student_range.replace('Students with last names ', '')}?",
                "Who do I call about attendance at DBHS?",
            ], a2)

    if "career" in role.lower() or "ayala" in name.lower():
        pairs += vary([
            "Who runs the College and Career Center at DBHS?",
            "Who processes work permits at DBHS?",
            "Where is the Career Center at Diamond Bar High School?",
            "How do I schedule an appointment at the DBHS Career Center?",
        ], core)

    if "activities" in role.lower():
        pairs += vary([
            "Who oversees USB at DBHS?",
            "Who is in charge of student government at Diamond Bar High School?",
            "Who do USB candidates need approval from before running for election?",
            "Who is the DBHS Activities Director?",
        ], core)

    if "ap coordinator" in role.lower():
        pairs += vary([
            "Who is the AP Coordinator at DBHS?",
            "Who handles AP exam payments at Diamond Bar High School?",
            "Who do I contact about AP Classroom registration at DBHS?",
            "Who is responsible for AP logistics at DBHS?",
        ], core)

    if "ib" in role.lower():
        pairs += vary([
            "Who is the IB contact at DBHS?",
            "Where do I send donations for the IB program?",
            "Who should I address IB-related correspondence to?",
        ], core)

    if "brahma tech" in role.lower() or "brahma tech" in dept.lower():
        pairs += vary([
            "Who coordinates the Brahma Tech Academy?",
            "Who runs the PLTW engineering program at DBHS?",
            "Who are the Brahma Tech coordinators at Diamond Bar High School?",
        ], core)

    return pairs


# ---------------------------------------------------------------------------
# Grade Level Coordinators (GLC)
# ---------------------------------------------------------------------------

def gen_glc(r: dict) -> list[dict]:
    group = r.get("group", "")
    coordinators = r.get("coordinators", [])
    summary = r.get("summary", "")

    if not coordinators:
        return []

    names = " and ".join(coordinators)
    a = summary if summary else f"The Grade Level Coordinator(s) for {group} at DBHS: {names}."

    pairs = vary([
        f"Who is the GLC for {group}?",
        f"Who is the grade level coordinator for {group}?",
        f"Who advises {group} students at DBHS?",
        f"Which counselor works with the {group}?",
    ], a)

    # Generic GLC questions with specific answer
    for coord in coordinators:
        pairs += vary([
            f"What does {coord} do at DBHS?",
            f"Who is {coord}?",
        ], a)

    return pairs


# ---------------------------------------------------------------------------
# Bell schedules
# ---------------------------------------------------------------------------

def gen_bell_schedule(r: dict) -> list[dict]:
    title = r.get("title", "")
    summary = r.get("summary", "")
    periods = r.get("periods", [])
    start = r.get("school_start", "")
    end = r.get("school_end", "")
    days = r.get("days", [])

    if not summary:
        return []

    # Build period string from list of dicts
    a = summary
    if periods and isinstance(periods, list):
        period_lines = "\n".join(
            f"  {p['period']}: {p['start']} – {p['end']}"
            for p in periods if isinstance(p, dict)
        )
        a = summary.rstrip(".") + f"\nPeriod times:\n{period_lines}"
    elif periods and isinstance(periods, dict):
        period_lines = "\n".join(f"  {k}: {v}" for k, v in periods.items())
        a = summary.rstrip(".") + f"\nPeriod times:\n{period_lines}"

    days_str = ", ".join(days) if days else ""
    templates = [
        f"What is the {title}?",
        f"What are the period times for the {title}?",
        "What is the DBHS bell schedule?",
        "When does school start at Diamond Bar High School?",
        "What time does DBHS end?",
        "What are the period times at DBHS?",
        "What time does Period 1 start at DBHS?",
        "What time is lunch at Diamond Bar High School?",
        "What time is brunch at DBHS?",
    ]
    if days_str:
        templates += [
            f"What time does school start on {days_str}?",
            f"What is the schedule on {days_str} at DBHS?",
        ]
    if start:
        templates += [
            f"What time does DBHS start on a regular day?",
            "When does Diamond Bar High School start in the morning?",
        ]
    return vary(templates, a)


# ---------------------------------------------------------------------------
# Programs (IB, AP, Brahma Tech, etc.)
# ---------------------------------------------------------------------------

def gen_program(r: dict) -> list[dict]:
    title = r.get("title", "")
    summary = r.get("summary", "")
    if not summary:
        return []

    pairs = vary([
        f"What is {title}?",
        f"Tell me about {title}.",
        f"What should students know about {title}?",
        f"Can you explain {title}?",
        f"What does the {title} offer?",
    ], summary)

    return pairs


# ---------------------------------------------------------------------------
# Quick facts / school mission / history
# ---------------------------------------------------------------------------

def gen_generic(r: dict) -> list[dict]:
    title = r.get("title", "")
    summary = r.get("summary", "")
    if not summary or not title:
        return []

    return vary([
        f"What is {title}?",
        f"Tell me about {title}.",
        f"Can you summarize {title}?",
        f"What should I know about {title} at DBHS?",
    ], summary)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ENTITY_HANDLERS = {
    "school": gen_school_overview,
    "quick_fact": gen_ceeb,
    "staff_member": gen_staff,
    "grade_level_coordinator": gen_glc,
    "bell_schedule": gen_bell_schedule,
    "program": gen_program,
    "academic_program": gen_program,
    "ib_courses": gen_program,
}


def generate_all(records: list[dict]) -> list[dict]:
    all_pairs = []
    for r in records:
        etype = r.get("entity_type", "")
        handler = ENTITY_HANDLERS.get(etype, gen_generic)
        try:
            pairs = handler(r)
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  [WARN] {r.get('id', '?')}: {e}")
    return all_pairs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(pairs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for p in pairs:
        q = p["messages"][0]["content"].strip().lower()
        if q not in seen:
            seen.add(q)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db = json.load(open(INPUT_FILE, encoding="utf-8"))
    records = db.get("records", db) if isinstance(db, dict) else db

    print(f"[*] Loaded {len(records)} records from {INPUT_FILE}")

    pairs = generate_all(records)
    print(f"[*] Generated {len(pairs)} raw pairs")

    pairs = deduplicate(pairs)
    print(f"[*] After deduplication: {len(pairs)} pairs")

    # Filter empty answers
    pairs = [p for p in pairs if p["messages"][1]["content"].strip()]
    print(f"[*] After empty-answer filter: {len(pairs)} pairs")

    random.shuffle(pairs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[✓] Saved {len(pairs)} atomic pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
