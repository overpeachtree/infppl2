from pathlib import Path
from ddgs import DDGS
from google import genai
from google.genai import types
import json
import os


PERSON_FILE = Path("people/001-charles-h-bennett.md")

PERSON_NAME = "Charles H. Bennett"

PERSON_IDENTITY = """
Charles H. Bennett is the American physicist and information theorist
associated with IBM Research.

He is known for quantum information science, quantum cryptography,
BB84 with Gilles Brassard, reversible computation, quantum teleportation,
and the 2025 ACM A.M. Turing Award.

Do NOT confuse him with Charles Henry Bennett, the British Victorian
illustrator and author who lived from 1828 to 1867.
"""

MAX_SEARCH_RESULTS = 10


def search_web():

    query = (
        '"Charles Henry Bennett" OR "Charles H. Bennett" '
        'interview podcast quantum IBM'
    )

    print("Searching:", query)

    results = list(
        DDGS().text(
            query,
            region="us-en",
            max_results=MAX_SEARCH_RESULTS
        )
    )

    candidates = []

    for result in results:
        if result.get("title") and result.get("href"):

            candidates.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })

    return candidates


def evaluate_with_gemini(candidates):

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    candidate_text = ""

    for i, item in enumerate(candidates, start=1):

        candidate_text += f"""
CANDIDATE {i}

Title:
{item["title"]}

URL:
{item["url"]}

Snippet:
{item["snippet"]}

"""

    prompt = f"""
You are a research relevance evaluator.

INTENDED PERSON

Name:
{PERSON_NAME}

Identity:
{PERSON_IDENTITY}


TASK

Evaluate every search result below.

For each candidate determine:

1. Does it refer to the intended Charles H. Bennett?
2. Is it a useful research source about him?
3. How confident are you?
4. Why?

A matching name alone is NOT enough.

Use contextual evidence such as:
- occupation
- organization
- research topics
- collaborators
- awards
- historical dates

A result about the Victorian illustrator must be rejected.

SEARCH RESULTS

{candidate_text}
"""

    schema = {
        "type": "object",
        "properties": {
            "person": {
                "type": "string"
            },
            "evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_number": {
                            "type": "integer"
                        },
                        "same_person": {
                            "type": "boolean"
                        },
                        "useful_source": {
                            "type": "boolean"
                        },
                        "confidence": {
                            "type": "number"
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "candidate_number",
                        "same_person",
                        "useful_source",
                        "confidence",
                        "reason"
                    ]
                }
            }
        },
        "required": [
            "person",
            "evaluations"
        ]
    }

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema
        )
    )

    return json.loads(response.text)


def print_results(candidates, evaluation):

    print("\nLLM EVALUATION\n")

    for item in evaluation["evaluations"]:

        index = item["candidate_number"] - 1

        candidate = candidates[index]

        print("=" * 60)
        print(candidate["title"])
        print("URL:", candidate["url"])
        print("Same person:", item["same_person"])
        print("Useful source:", item["useful_source"])
        print("Confidence:", item["confidence"])
        print("Reason:", item["reason"])


def update_markdown(candidates, evaluation):

    accepted = []

    for item in evaluation["evaluations"]:

        if item["same_person"] and item["useful_source"]:

            index = item["candidate_number"] - 1

            accepted.append({
                **candidates[index],
                **item
            })

    start = "<!-- AUTO-UPDATE-START -->"
    end = "<!-- AUTO-UPDATE-END -->"

    text = PERSON_FILE.read_text(
        encoding="utf-8"
    )

    if start not in text or end not in text:
        raise RuntimeError(
            "AUTO-UPDATE markers not found."
        )

    lines = [
        start,
        "",
        "### Gemini-Validated Research Sources",
        ""
    ]

    if not accepted:

        lines.append(
            "No relevant sources passed the identity check."
        )

    else:

        for i, item in enumerate(
            accepted,
            start=1
        ):

            lines.extend([
                f"#### {i}. {item['title']}",
                "",
                f"[Open source]({item['url']})",
                "",
                f"**Identity confidence:** "
                f"{item['confidence']:.2f}",
                "",
                f"**Why accepted:** "
                f"{item['reason']}",
                ""
            ])

    lines.extend([
        end,
        ""
    ])

    new_section = "\n".join(lines)

    before = text.split(start, 1)[0]
    after = text.split(end, 1)[1]

    PERSON_FILE.write_text(
        before + new_section + after,
        encoding="utf-8"
    )


def main():

    if not os.getenv("GEMINI_API_KEY"):

        raise RuntimeError(
            "GEMINI_API_KEY is missing."
        )

    candidates = search_web()

    print(
        f"Found {len(candidates)} candidate results."
    )

    evaluation = evaluate_with_gemini(
        candidates
    )

    print_results(
        candidates,
        evaluation
    )

    update_markdown(
        candidates,
        evaluation
    )

    print(
        "\nResearch file updated."
    )


if __name__ == "__main__":
    main()
