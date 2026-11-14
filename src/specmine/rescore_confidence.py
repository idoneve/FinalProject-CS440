"""Re-derive labeler confidence scores from the rubric in the labeling prompt.

The Gemini labeler is given a confidence rubric tied to how many AI tell
categories it found, but its self-reported scores cluster at 85-90 regardless
of the evidence it cites. This script reads each row's ``reasoning``, works out
which rubric band the cited evidence falls in, and places the original score
inside that band (so Gemini's relative ordering is kept within a band).

Labels are never changed. Rows whose reasoning does not support their label
under the prompt's own rules get the minimum confidence and are listed in a
review file.

Usage:
    python src/specmine/rescore_confidence.py data/ai_analysis_results.csv
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

BACKUP_SUFFIX = "_gemini_confidence"
REVIEW_SUFFIX = "_review"

# Gemini's reported scores span 75-100; they are rescaled within a band.
REPORTED_MIN, REPORTED_MAX = 75, 100
# Label 1 bands, then label 0 bands; both mirror the rubric in the prompt.
LOW_BAND, MID_BAND, HIGH_BAND = (50, 64), (65, 84), (85, 100)
ONE_TELL_BAND, NO_EVIDENCE_BAND, HUMAN_BAND = (50, 55), (55, 65), (80, 95)
QUOTE = re.compile(r"""['"`‘“][^'"`’”]{3,}['"`’”]""")
UNSUPPORTED_CONFIDENCE = 50

CATEGORY_NAMES = {
    "A": r"chat residue",
    "B": r"tool or prompt residue|prompt residue|tool attribution",
    "C": r"template scaffolding",
    "D": r"bold[- ]label",
    "E": r"emoji",
    "F": r"uniform(?:,)? exhaustive|exhaustive(?:,| and)? uniform|uniform structure",
    "G": r"generic assistant prose|filler vocabulary",
    "H": r"unsupported precision",
    "I": r"typeset",
}
CATEGORY_LETTER = re.compile(
    r"\bcategor(?:y|ies) ([A-I])(?:\s*/\s*([A-I]))?\b|\b([A-I])\. [A-Z]",
    re.IGNORECASE,
)
# Bold Given/When/Then keywords are a requirements format, which the prompt
# lists as "not a tell", so they cannot be the evidence for category D.
GHERKIN_KEYWORDS = r"WHEN|THEN|GIVEN|AND|当|则"
GHERKIN_BOLD = re.compile(
    rf"\*\*\s*(?:{GHERKIN_KEYWORDS})\s*\*\*|- (?:GIVEN|WHEN)\b", re.IGNORECASE
)
BOLD_SPAN = re.compile(r"\*\*([^*]+)\*\*")
GHERKIN_WORD = re.compile(rf"\s*(?:{GHERKIN_KEYWORDS})\s*", re.IGNORECASE)
SINGLE_TELL = re.compile(
    r"only (?:one|a single)|single (?:potential |stylistic |possible )?"
    r"(?:ai )?(?:tell|category|stylistic element)|resulting in a single tell"
    r"|matches category [A-I]",
    re.IGNORECASE,
)
HUMAN_INDICATOR = re.compile(
    r"human indicator|typo|shorthand|first-person|jargon|terse|uneven"
    r"|inconsistent format|informal|dated note|named people|bilingual",
    re.IGNORECASE,
)
# "TBD - created by archiving change <name>. Update Purpose after archive." is
# identical across repositories: it is tool boilerplate, not a human note.
ARCHIVE_PLACEHOLDER = re.compile(r"created by archiving", re.IGNORECASE)
# Newer labeling runs start each reasoning with a fixed summary line.
SUMMARY_LINE = re.compile(
    r"\s*Tells:\s*(?P<tells>[^.]*)\.\s*Human indicators:\s*(?P<human>yes|no)",
    re.IGNORECASE,
)


def cited_categories(reasoning: str) -> set[str]:
    """Return the tell categories a reasoning string cites, by letter or name."""
    summary = SUMMARY_LINE.match(reasoning)
    if summary:
        return set(re.findall(r"\b[A-I]\b", summary["tells"].upper()))
    found = set()
    for match in CATEGORY_LETTER.finditer(reasoning):
        found.update(letter.upper() for letter in match.groups() if letter)
    for letter, pattern in CATEGORY_NAMES.items():
        if re.search(pattern, reasoning, re.IGNORECASE):
            found.add(letter)
    return found


def cites_other_bold_label(reasoning: str) -> bool:
    """Return whether a quoted bold label is something other than a Gherkin keyword."""
    return any(
        not GHERKIN_WORD.fullmatch(span) for span in BOLD_SPAN.findall(reasoning)
    )


def place_in_band(reported: int, band: tuple[int, int]) -> int:
    """Map a reported 75-100 score onto ``band``, keeping its relative position."""
    low, high = band
    span = REPORTED_MAX - REPORTED_MIN
    position = min(max((reported - REPORTED_MIN) / span, 0.0), 1.0)
    return round(low + position * (high - low))


def rescore(label: int, reported: int, reasoning: str) -> tuple[int, str]:
    """Return the rubric-based confidence and the rule that produced it."""
    if label == 1:
        categories = cited_categories(reasoning)
        if (
            "D" in categories
            and GHERKIN_BOLD.search(reasoning)
            and not cites_other_bold_label(reasoning)
        ):
            categories.discard("D")
            if len(categories) < 2:
                return UNSUPPORTED_CONFIDENCE, "review: label 1 rests on Gherkin bullets"
        if len(categories) < 2:
            return UNSUPPORTED_CONFIDENCE, "review: fewer than two categories cited"
        if categories & {"A", "B"}:
            return place_in_band(reported, HIGH_BAND), "direct residue plus another"
        if len(categories) >= 3:
            return place_in_band(reported, MID_BAND), "three or more categories"
        return place_in_band(reported, LOW_BAND), "two stylistic categories"

    summary = SUMMARY_LINE.match(reasoning)
    if summary:
        tells = cited_categories(reasoning)
        if len(tells) >= 2:
            return UNSUPPORTED_CONFIDENCE, "review: label 0 but two categories cited"
        one_tell = len(tells) == 1
        # A claimed human indicator only counts when the reasoning quotes it.
        human = summary["human"].lower() == "yes" and bool(
            QUOTE.search(reasoning[summary.end() :])
        )
    else:
        one_tell = bool(SINGLE_TELL.search(reasoning))
        human = bool(HUMAN_INDICATOR.search(reasoning))
    if one_tell:
        return place_in_band(reported, ONE_TELL_BAND), "one tell present"
    if human and not ARCHIVE_PLACEHOLDER.search(reasoning):
        return place_in_band(reported, HUMAN_BAND), "no tells, human indicators"
    return place_in_band(reported, NO_EVIDENCE_BAND), "no tells, no human indicators"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="labeler results CSV to rescore")
    args = parser.parse_args(argv)

    results_path: Path = args.results
    backup_path = results_path.with_name(
        f"{results_path.stem}{BACKUP_SUFFIX}{results_path.suffix}"
    )
    review_path = results_path.with_name(
        f"{results_path.stem}{REVIEW_SUFFIX}{results_path.suffix}"
    )
    # Always rescore from the untouched Gemini scores so reruns are idempotent.
    if not backup_path.exists():
        shutil.copy2(results_path, backup_path)
    original = pd.read_csv(backup_path, dtype=str, keep_default_na=False)
    current = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    unchanged = ["spec_id", "label", "reasoning"]
    if not original[unchanged].equals(current[unchanged]):
        print(
            f"ERROR: {backup_path} does not match {results_path}; it is from an "
            "earlier labeling run. Move or delete it, then run this again.",
            file=sys.stderr,
        )
        return 1

    scored = [
        rescore(int(row.label), int(row.confidence), row.reasoning)
        for row in original.itertuples()
    ]
    rescored = original.copy()
    rescored["confidence"] = [str(confidence) for confidence, _ in scored]
    rescored.to_csv(results_path, index=False, lineterminator="\n")

    review = original.assign(
        gemini_confidence=original["confidence"],
        confidence=rescored["confidence"],
        rule=[rule for _, rule in scored],
    )
    review = review[review["rule"].str.startswith("review")]
    review.to_csv(review_path, index=False, lineterminator="\n")

    rules = pd.Series([rule for _, rule in scored]).value_counts()
    print(rules.to_string())
    print(f"\nRescored {len(rescored)} rows in {results_path}")
    print(f"Original scores kept in {backup_path}")
    print(f"{len(review)} rows to review listed in {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
