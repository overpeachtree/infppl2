from __future__ import annotations

from pathlib import Path
from typing import List
from ddgs import DDGS
from openai import OpenAI
from pydantic import BaseModel, Field
import os


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PERSON_FILE = Path("people/001-charles-h-bennett.md")

PERSON_NAME = "Charles H. Bennett"

PERSON_IDENTITY = """
Charles H. Bennett is the American physicist and information theorist
associated with IBM Research.

He is known for:
- quantum information science
- quantum cryptography
- BB84 with Gilles Brassard
- reversible computation
- quantum teleportation
- information theory
- the 2025 ACM A.M. Turing Award with Gilles Brassard

Do NOT confuse him with Charles Henry Bennett, the British Victorian
illustrator and author who lived from 1828 to 1867.
"""

MAX_SEARCH_RESULTS = 10


# ---------------------------------------------------------
# STRUCTURED OUTPUT MODELS
# ---------------------------------------------------------

class CandidateResult(BaseModel):
    title: str
    url: str
    snippet: str


class CandidateEvaluation(BaseModel):
    title: str
    url: str

    same_person: bool = Field(
        description="True only if the source refers to the intended Charles H. Bennett."
    )

    relevant_interview_or_research_source: bool = Field(
        description=(
            "True if this is a useful interview, podcast, talk, profile, "
            "award page, or research-related source about the intended person."
        )
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence that the identity classification is correct."
    )

    reason: str = Field(
        description="Short explanation for the classification."
    )


class ResearchEvaluation(BaseModel):
    person: str
    evaluations: List[CandidateEvaluation]


# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------

def search_web() -> List[CandidateResult]:

    query = '"Charles Henry Bennett" OR "Charles H. Bennett" interview podcast quantum IBM'

    print(f"Searching web for:\n{query}\n")

    raw_results = list(
        DDGS().text(
            query,
            region="us-en",
            max_results=MAX_SEARCH_RESULTS
        )
    )

    candidates = []

    for result in raw_results:

        title = result.get("title", "")
        url = result.get("href", "")
        snippet = result.get("body", "")

        if not title or not url:
            continue

        candidates.append(
            CandidateResult(
                title=title,
                url=url,
                snippet=snippet
            )
        )

    print(f"Found {len(candidates)} candidates.")

    return candidates


# ---------------------------------------------------------
# LLM JUDGMENT
# ---------------------------------------------------------

def evaluate_with_llm(
    candidates: List[CandidateResult]
) -> ResearchEvaluation:

    client = OpenAI()

    candidate_text = "\n\n".join(
        [
            f"""
CANDIDATE {i}

Title:
{candidate.title}

URL:
{candidate.url}

Snippet:
{candidate.snippet}
"""
            for i, candidate in enumerate(candidates, start=1)
        ]
    )

    system_prompt = """
You are a research relevance evaluator.

Your task is identity disambiguation and source relevance evaluation.

You must determine whether each search result refers to the intended
person described by the user.

A matching name alone is NOT enough.

Use contextual evidence such as occupation, organization, dates,
collaborators, research topics, awards, and other identity signals.

Do not invent facts that are not supported by the supplied identity
description or the search result.

For each candidate:
1. Determine whether it refers to the intended person.
2. Determine whether it is a useful research/interview source.
3. Give a confidence score between 0 and 1.
4. Give a concise explanation.

A result about another person with the same name must be marked
same_person = false.
"""

    user_prompt = f"""
INTENDED PERSON

Name:
{PERSON_NAME}

Identity:
{PERSON_IDENTITY}


SEARCH RESULTS

{candidate_text}
"""

    response = client.responses.parse(
        model="gpt-5.6",
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        text_format=ResearchEvaluation
    )

    return response.output_parsed


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

def print_evaluations(result: ResearchEvaluation):

    print()
    print("=" * 70)
    print(f"LLM evaluation for {result.person}")
    print("=" * 70)

    for i, evaluation in enumerate(
        result.evaluations,
        start=1
    ):

        print()
        print(f"{i}. {evaluation.title}")

        print(
            "Same person:",
            evaluation.same_person
        )

        print(
            "Useful source:",
            evaluation.relevant_interview_or_research_source
        )

        print(
            "Confidence:",
            evaluation.confidence
        )

        print(
            "Reason:",
            evaluation.reason
        )

        print(
            "URL:",
            evaluation.url
        )


# ---------------------------------------------------------
# UPDATE MARKDOWN
# ---------------------------------------------------------

def update_markdown(
    evaluation: ResearchEvaluation
):

    text = PERSON_FILE.read_text(
        encoding="utf-8"
    )

    start_marker = "<!-- AUTO-UPDATE-START -->"
    end_marker = "<!-- AUTO-UPDATE-END -->"

    if (
        start_marker not in text
        or end_marker not in text
    ):
        raise RuntimeError(
            "AUTO-UPDATE markers not found."
        )

    accepted = [
        item
        for item in evaluation.evaluations
        if (
            item.same_person
            and
            item.relevant_interview_or_research_source
        )
    ]

    lines = [
        start_marker,
        "",
        "### LLM-Validated Research Sources",
        ""
    ]

    if not accepted:

        lines.append(
            "No relevant sources passed the LLM identity check."
        )

    else:

        for i, item in enumerate(
            accepted,
            start=1
        ):

            lines.extend(
                [
                    f"#### {i}. {item.title}",
                    "",
                    f"[Open source]({item.url})",
                    "",
                    f"**Identity confidence:** "
                    f"{item.confidence:.2f}",
                    "",
                    f"**Why accepted:** "
                    f"{item.reason}",
                    ""
                ]
            )

    lines.extend(
        [
            end_marker,
            ""
        ]
    )

    new_section = "\n".join(lines)

    before = text.split(
        start_marker,
        1
    )[0]

    after = text.split(
        end_marker,
        1
    )[1]

    PERSON_FILE.write_text(
        before + new_section + after,
        encoding="utf-8"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is missing."
        )

    candidates = search_web()

    if not candidates:
        raise RuntimeError(
            "No search results found."
        )

    result = evaluate_with_llm(
        candidates
    )

    print_evaluations(
        result
    )

    update_markdown(
        result
    )

    print()
    print(
        f"Updated {PERSON_FILE}"
    )


if __name__ == "__main__":
    main()
