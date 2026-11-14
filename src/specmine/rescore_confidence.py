"""Re-derive labeler confidence scores from the rubric in the labeling prompt.

The Gemini labeler is given a confidence rubric tied to how many AI tell
categories it found, but its self-reported scores cluster high regardless of
the evidence it cites. This script reads each row's ``reasoning``, keeps only
the tell categories and human indicators backed by quoted evidence, works out
which rubric band that evidence falls in, and places the reported score inside
that band (so Gemini's relative ordering is kept within a band).

The labeler's results file is never modified: it stays the append-only record
of Gemini's raw output, so ``ai-labeler.py`` can keep resuming it. Output goes
to two new files next to it:

- ``<stem>_rescored.csv``: every row, with ``confidence`` holding the
  rubric-rescored score (ready for confidence-weighted training),
  ``gemini_confidence`` holding Gemini's self-reported score, and
  ``rescore_rule`` naming the rubric rule applied.
- ``<stem>_review.csv``: the rows whose reasoning does not support their
  label under the prompt's own rules. They get the minimum confidence.

Labels are never changed. Rerun after the labeler appends rows to rescore the
whole file again.

Usage:
    python src/specmine/rescore_confidence.py data/ai_analysis_results.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

RESCORED_SUFFIX = "_rescored"
REVIEW_SUFFIX = "_review"
# Earlier versions rescored the results file in place and kept Gemini's scores
# in this backup; it is read only to recover those original scores.
LEGACY_BACKUP_SUFFIX = "_gemini_confidence"
ROW_KEY = ["repo_id", "spec_id", "label", "reasoning"]

# The revised rubric allows scores from 50 to 100; they are rescaled within a
# band. Lower scores are clamped to the bottom of the band.
REPORTED_MIN, REPORTED_MAX = 50, 100
# Label 1 bands, then label 0 bands; both mirror the rubric in the prompt.
LOW_BAND, MID_BAND, HIGH_BAND = (50, 64), (65, 84), (85, 100)
ONE_TELL_BAND, NO_EVIDENCE_BAND, HUMAN_BAND = (50, 55), (55, 65), (80, 95)
QUOTE = re.compile(r"""['"`‘“]([^'"`’”]+)['"`’”]""")
UNSUPPORTED_CONFIDENCE = 50
STYLISTIC = set("DEFGHI")

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
# Bold Given/When/Then keywords and requirement IDs are a requirements format,
# which the prompt lists as "not a tell", so they cannot be the evidence for D.
GHERKIN_KEYWORDS = r"WHEN|THEN|GIVEN|AND|当|则"
BOLD_SPAN = re.compile(r"\*\*([^*]+)\*\*")
NON_LABEL_BOLD = re.compile(
    rf"\s*(?:{GHERKIN_KEYWORDS}|[A-Z]{{1,6}}[-_]?\d[\w.-]*|[\W\d_]*)\s*:?\s*",
    re.IGNORECASE,
)
UNBOLDED_LABEL = re.compile(r"^\s*(?:[-*]\s+)?[A-Z][\w /&()-]{1,40}:\s+\S")
# Quoted evidence that can show each category. A and F have no reliable textual
# signature, so for them any quote in a sentence naming the category counts;
# for F, so does naming the boilerplate sections the prompt lists.
BOILERPLATE_SECTION = re.compile(
    r"Non-Goals|Success Metrics|Future Enhancements|Mitigations|Conclusion"
    r"|Open Questions|Risks",
)
EVIDENCE = {
    "B": re.compile(
        r"generated (?:by|with)|\$[A-Z_]{3,}|\{\{|prompt|claude|copilot|gpt|cursor",
        re.IGNORECASE,
    ),
    "C": re.compile(
        r"\((?:mandatory|optional)\)|NEEDS CLARIFICATION|ACTION REQUIRED|"
        r"\bfill (?:this|in)\b|\bdescribe\b|\[[^\]]+\]|<[^>]+>|\bTODO\b",
        re.IGNORECASE,
    ),
    "E": re.compile("[\u2600-\u27bf\u2b00-\u2bff\U0001f000-\U0001faff]"),
    "G": re.compile(
        r"comprehensive|robust|seamless|leverag|ensur|not (?:just|only)|—|–"
        r"|streamlin|holistic|empower|crucial|essenti|enhanc|intuitive|scalab|frictionless",
        re.IGNORECASE,
    ),
    "H": re.compile(r"\d"),
    "I": re.compile(r"[‘’“”→←↔⇒─│├└┌┐┘┬┴┼═║]"),
}
PERVASIVE = re.compile(
    r"pervasive|throughout|consistent(?:ly)?|repeated(?:ly)?|extensive(?:ly)?"
    r"|heav(?:y|ily)|every|nearly all|almost all|most (?:of|list|items|sections)"
    r"|numerous|widespread|frequent(?:ly)?",
    re.IGNORECASE,
)
SINGLE_TELL = re.compile(
    r"only (?:one|a single)|single (?:potential |stylistic |possible )?"
    r"(?:ai )?(?:tell|category|stylistic element)|resulting in a single tell"
    r"|matches category [A-I]",
    re.IGNORECASE,
)
# The indicator kinds the prompt accepts; "human indicator" alone, mixed
# languages, jargon, or terseness are not evidence.
HUMAN_INDICATOR = re.compile(
    r"typo|misspell|grammar|first[- ]person|named (?:people|person|author)"
    r"|dated|\bTODO\b|shorthand|informal",
    re.IGNORECASE,
)
# "TBD - created by archiving change <name>. Update Purpose after archive." is
# identical across repositories: it is tool boilerplate, not a human note.
ARCHIVE_PLACEHOLDER = re.compile(r"created by archiving", re.IGNORECASE)
# Newer labeling runs start each reasoning with a fixed summary line.
SUMMARY_LINE = re.compile(
    r"\s*Tells:\s*(?P<tells>[^.]*)\.\s*Human indicators:\s*(?P<human>yes|no)\.?",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+")
CLAUSE_END = re.compile(r"[,;]\s*|\s+along(?:side| with)\s+")


def mentioned_categories(text: str) -> set[str]:
    """Return the tell categories ``text`` names, by letter or by name."""
    found = set()
    for match in CATEGORY_LETTER.finditer(text):
        found.update(letter.upper() for letter in match.groups() if letter)
    for letter, pattern in CATEGORY_NAMES.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.add(letter)
    return found


def claimed_categories(reasoning: str) -> set[str]:
    """Return the categories the reasoning claims, before checking evidence."""
    summary = SUMMARY_LINE.match(reasoning)
    if summary:
        return set(re.findall(r"\b[A-I]\b", summary["tells"].upper()))
    return mentioned_categories(reasoning)


def evidence_text(reasoning: str) -> str:
    """Return the part of the reasoning after the summary line, if any."""
    summary = SUMMARY_LINE.match(reasoning)
    return reasoning[summary.end() :] if summary else reasoning


def sentences(text: str, boundary_pattern: re.Pattern = SENTENCE_END) -> list[str]:
    """Split ``text`` into sentences without splitting inside quotes."""
    masked = QUOTE.sub(lambda m: "\0" * len(m[0]), text)
    pieces, start = [], 0
    for boundary in boundary_pattern.finditer(masked):
        pieces.append(text[start : boundary.start()])
        start = boundary.end()
    pieces.append(text[start:])
    return [piece for piece in pieces if piece.strip()]


def quotes(text: str) -> list[str]:
    """Return quoted spans plus verbatim lines pasted after the summary."""
    found = QUOTE.findall(text)
    found += [line for line in text.splitlines() if line.lstrip().startswith("- ")]
    return found


def is_descriptive_bold_label(quote: str) -> bool:
    """Return whether a quote shows a bold-label bullet with a descriptive label."""
    spans = BOLD_SPAN.findall(quote)
    if spans:
        return any(not NON_LABEL_BOLD.fullmatch(span) for span in spans)
    return bool(UNBOLDED_LABEL.match(quote))


def cites_other_bold_label(reasoning: str) -> bool:
    """Return whether any quote shows a descriptive bold label (category D)."""
    return any(is_descriptive_bold_label(q) for q in quotes(evidence_text(reasoning)))


def supported_categories(reasoning: str) -> set[str]:
    """Return the claimed categories that quoted evidence in the reasoning shows."""
    body = evidence_text(reasoning)
    body_quotes = quotes(body)
    supported = set()
    for letter in claimed_categories(reasoning):
        if letter == "D":
            ok = any(is_descriptive_bold_label(q) for q in body_quotes)
        elif letter in EVIDENCE:
            ok = any(EVIDENCE[letter].search(q) for q in body_quotes)
        else:
            ok = any(
                letter in mentioned_categories(s)
                and (
                    QUOTE.search(s)
                    or (letter == "F" and BOILERPLATE_SECTION.search(s))
                )
                for s in sentences(body)
            )
        if ok:
            supported.add(letter)
    return supported


def is_pervasive(letter: str, reasoning: str) -> bool:
    """Return whether a clause naming ``letter`` describes it as pervasive."""
    return any(
        letter in mentioned_categories(clause) and PERVASIVE.search(clause)
        for clause in sentences(evidence_text(reasoning), CLAUSE_END)
    )


def has_quoted_human_indicator(text: str) -> bool:
    """Return whether a sentence names a human indicator and quotes it."""
    return any(
        HUMAN_INDICATOR.search(QUOTE.sub("", s))
        and any(not ARCHIVE_PLACEHOLDER.search(q) for q in QUOTE.findall(s))
        for s in sentences(text)
    )


def place_in_band(reported: int, band: tuple[int, int]) -> int:
    """Map a reported 50-100 score onto ``band``, keeping its relative position."""
    low, high = band
    span = REPORTED_MAX - REPORTED_MIN
    position = min(max((reported - REPORTED_MIN) / span, 0.0), 1.0)
    return round(low + position * (high - low))


def rescore(label: int, reported: int, reasoning: str) -> tuple[int, str]:
    """Return the rubric-based confidence and the rule that produced it."""
    claimed = claimed_categories(reasoning)
    categories = supported_categories(reasoning)
    if label == 1:
        if len(categories) < 2:
            if "D" in claimed - categories and BOLD_SPAN.search(reasoning):
                return UNSUPPORTED_CONFIDENCE, (
                    "review: label 1 rests on Gherkin bullets or requirement IDs"
                )
            if len(claimed) >= 2:
                return UNSUPPORTED_CONFIDENCE, (
                    "review: claimed categories lack quoted evidence"
                )
            return UNSUPPORTED_CONFIDENCE, "review: fewer than two categories cited"
        if categories & {"A", "B"}:
            return place_in_band(reported, HIGH_BAND), "direct residue plus another"
        if len(categories) >= 3:
            return place_in_band(reported, MID_BAND), "three or more categories"
        if all(is_pervasive(letter, reasoning) for letter in categories):
            return place_in_band(reported, MID_BAND), "two pervasive categories"
        return place_in_band(reported, LOW_BAND), "two stylistic categories"

    summary = SUMMARY_LINE.match(reasoning)
    if summary:
        if len(categories) >= 2:
            return UNSUPPORTED_CONFIDENCE, "review: label 0 but two categories cited"
        one_tell = len(categories) == 1
        human = summary["human"].lower() == "yes"
        if human and not has_quoted_human_indicator(evidence_text(reasoning)):
            return UNSUPPORTED_CONFIDENCE, "review: human indicator not quoted"
    else:
        one_tell = bool(SINGLE_TELL.search(reasoning))
        human = has_quoted_human_indicator(reasoning)
    if one_tell:
        return place_in_band(reported, ONE_TELL_BAND), "one tell present"
    if human:
        return place_in_band(reported, HUMAN_BAND), "no tells, human indicators"
    return place_in_band(reported, NO_EVIDENCE_BAND), "no tells, no human indicators"


def sibling(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to a temp file, then atomically replace ``path``."""
    if path.is_symlink():
        raise ValueError(f"Refusing to write through symlink {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def recover_gemini_scores(current: pd.DataFrame, backup_path: Path) -> pd.Series:
    """Return Gemini's scores, undoing an earlier in-place rescore if needed.

    Rows present in the legacy backup take their score from it; rows the
    labeler appended after that rescore already hold Gemini's score.
    """
    reported = current["confidence"].copy()
    if not backup_path.exists():
        return reported
    backup = pd.read_csv(backup_path, dtype=str, keep_default_na=False)
    original = dict(
        zip(map(tuple, backup[["repo_id", "spec_id"]].values), backup.itertuples())
    )
    for index, row in current.iterrows():
        old = original.get((row.repo_id, row.spec_id))
        if old is None:
            continue
        if any(getattr(old, column) != row[column] for column in ROW_KEY):
            raise ValueError(
                f"{backup_path} disagrees with the results for spec {row.spec_id}; "
                "it is from an earlier labeling run. Move it aside and rerun."
            )
        reported[index] = old.confidence
    return reported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="labeler results CSV to rescore")
    args = parser.parse_args(argv)

    results_path: Path = args.results
    rescored_path = sibling(results_path, RESCORED_SUFFIX)
    review_path = sibling(results_path, REVIEW_SUFFIX)
    current = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    try:
        reported = recover_gemini_scores(
            current, sibling(results_path, LEGACY_BACKUP_SUFFIX)
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    scored = [
        rescore(int(label), int(score), reasoning)
        if label.strip() and score.strip()
        else ("", "unlabeled")
        for label, score, reasoning in zip(
            current["label"], reported, current["reasoning"]
        )
    ]
    rescored = current[["repo_id", "spec_id", "label"]].assign(
        confidence=[str(confidence) for confidence, _ in scored],
        gemini_confidence=reported,
        rescore_rule=[rule for _, rule in scored],
        reasoning=current["reasoning"],
    )
    review = rescored[rescored["rescore_rule"].str.startswith("review")]
    for path in (rescored_path, review_path):
        if path.is_symlink():
            print(f"ERROR: Refusing to write through symlink {path}", file=sys.stderr)
            return 1
    try:
        write_csv_atomically(rescored, rescored_path)
        write_csv_atomically(review, review_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(rescored["rescore_rule"].value_counts().to_string())
    print(f"\nRescored {len(rescored)} rows from {results_path} into {rescored_path}")
    print(f"{len(review)} rows to review listed in {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
