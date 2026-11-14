import os

import pandas as pd
import pytest

from specmine import rescore_confidence as rc

COLUMNS = ["repo_id", "spec_id", "label", "confidence", "reasoning"]


def write_results(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, lineterminator="\n")


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# Comment 5: the full 50-100 range is rescaled, not clamped at 75.
def test_place_in_band_uses_full_rubric_range():
    assert rc.place_in_band(50, rc.MID_BAND) == 65
    assert rc.place_in_band(100, rc.MID_BAND) == 84
    assert rc.place_in_band(55, rc.NO_EVIDENCE_BAND) != rc.place_in_band(
        65, rc.NO_EVIDENCE_BAND
    )
    assert rc.place_in_band(10, rc.LOW_BAND) == 50


# Comment 4: claimed categories need qualifying quoted evidence.
def test_unquoted_category_claims_are_flagged():
    reasoning = (
        "Tells: D, G. Human indicators: no. The document uses bold-label "
        "bullets and typical assistant prose."
    )
    confidence, rule = rc.rescore(1, 90, reasoning)
    assert confidence == rc.UNSUPPORTED_CONFIDENCE
    assert rule.startswith("review")


def test_quote_must_qualify_for_its_category():
    reasoning = (
        "Tells: G, H. Human indicators: no. Generic assistant prose such as "
        "'frobnicate the widget' and unsupported precision like 'under 200ms'."
    )
    assert rc.supported_categories(reasoning) == {"H"}
    assert rc.rescore(1, 90, reasoning)[1].startswith("review")


def test_supported_two_stylistic_categories_get_low_band():
    reasoning = (
        "Tells: D, H. Human indicators: no. Category D is shown by "
        "'- **Prototype:** Implement a subset'. Category H is shown by "
        "'p90 latency of 100 milliseconds'."
    )
    confidence, rule = rc.rescore(1, 75, reasoning)
    assert rule == "two stylistic categories"
    assert rc.LOW_BAND[0] <= confidence <= rc.LOW_BAND[1]


def test_human_indicator_needs_qualifying_quote():
    unquoted = "Tells: none. Human indicators: yes. The author writes informally."
    assert rc.rescore(0, 90, unquoted)[1] == "review: human indicator not quoted"
    wrong_kind = "Tells: none. Human indicators: yes. It mixes languages: '系統 SHALL'."
    assert rc.rescore(0, 90, wrong_kind)[1].startswith("review")
    placeholder = (
        "Tells: none. Human indicators: yes. A dated note: "
        "'TBD - created by archiving change foo'."
    )
    assert rc.rescore(0, 90, placeholder)[1].startswith("review")
    quoted = "Tells: none. Human indicators: yes. A typo: 'recieve the payload'."
    confidence, rule = rc.rescore(0, 90, quoted)
    assert rule == "no tells, human indicators"
    assert rc.HUMAN_BAND[0] <= confidence <= rc.HUMAN_BAND[1]


# Comment 9: requirement IDs and Gherkin keywords are not bold labels.
@pytest.mark.parametrize("quote", ["- **FR-001**: System MUST", "- **WHEN** user", "**SC-12**"])
def test_requirement_ids_are_not_bold_labels(quote):
    assert not rc.is_descriptive_bold_label(quote)
    reasoning = (
        f"Tells: D, H. Human indicators: no. Bold-label bullets such as '{quote}' "
        "and unsupported precision like '99.9% uptime'."
    )
    confidence, rule = rc.rescore(1, 90, reasoning)
    assert confidence == rc.UNSUPPORTED_CONFIDENCE
    assert "requirement IDs" in rule


def test_descriptive_bold_label_counts():
    assert rc.is_descriptive_bold_label("- **Why this priority**: LinkedIn")
    assert rc.cites_other_bold_label("Tells: D. Human indicators: no. '**Build**: Gradle'")


# Comment 10: two pervasive stylistic categories map to the mid band.
def test_two_pervasive_categories_get_mid_band():
    reasoning = (
        "Tells: D, G. Human indicators: no. Bold-label bullets are used "
        "throughout, such as '- **Language**: Concise'; generic assistant prose "
        "appears repeatedly, like 'robust and seamless'."
    )
    confidence, rule = rc.rescore(1, 75, reasoning)
    assert rule == "two pervasive categories"
    assert rc.MID_BAND[0] <= confidence <= rc.MID_BAND[1]


def test_residue_plus_another_gets_high_band():
    reasoning = (
        "Tells: B, H. Human indicators: no. Tool residue: 'Generated with Copilot'. "
        "Precision: 'under 200ms'."
    )
    assert rc.rescore(1, 90, reasoning)[1] == "direct residue plus another"


# Comments 2, 6, 7, 8: the raw results file is never rewritten; output is a
# separate file with both scores, and appended rows are picked up on rerun.
def test_main_writes_separate_file_and_keeps_raw_results(tmp_path):
    results = tmp_path / "ai_analysis_results.csv"
    write_results(
        results,
        [
            ["1", "a" * 16, "0", "60", "Tells: none. Human indicators: no."],
            ["2", "b" * 16, "", "", "unlabeled"],
        ],
    )
    raw_before = results.read_bytes()
    assert rc.main([str(results)]) == 0
    assert results.read_bytes() == raw_before

    rescored = read(tmp_path / "ai_analysis_results_rescored.csv")
    assert list(rescored.columns) == [
        "repo_id", "spec_id", "label", "confidence",
        "gemini_confidence", "rescore_rule", "reasoning",
    ]
    assert rescored.loc[0, "gemini_confidence"] == "60"
    assert rescored.loc[0, "confidence"] == str(rc.place_in_band(60, rc.NO_EVIDENCE_BAND))
    assert rescored.loc[1, "rescore_rule"] == "unlabeled"

    # The labeler appends a row; rescoring again includes it with raw score kept.
    with results.open("a") as stream:
        stream.write(f"3,{'c' * 16},1,95,\"Tells: D. Human indicators: no.\"\n")
    assert rc.main([str(results)]) == 0
    rescored = read(tmp_path / "ai_analysis_results_rescored.csv")
    assert len(rescored) == 3
    assert rescored.loc[2, "gemini_confidence"] == "95"
    review = read(tmp_path / "ai_analysis_results_review.csv")
    assert list(review["spec_id"]) == ["c" * 16]


# Comments 3, 6: a legacy backup restores Gemini's scores, keyed with repo_id.
def test_legacy_backup_restores_scores_and_keeps_appended_rows(tmp_path):
    results = tmp_path / "r.csv"
    backup = tmp_path / "r_gemini_confidence.csv"
    row = ["1", "a" * 16, "0", "88", "Tells: none. Human indicators: no."]
    write_results(backup, [row])
    write_results(
        results,
        [row[:3] + ["57"] + row[4:], ["1", "b" * 16, "0", "62", row[4]]],
    )
    assert rc.main([str(results)]) == 0
    rescored = read(tmp_path / "r_rescored.csv")
    assert list(rescored["gemini_confidence"]) == ["88", "62"]


def test_legacy_backup_with_stale_repo_id_is_rejected(tmp_path):
    results = tmp_path / "r.csv"
    write_results(
        tmp_path / "r_gemini_confidence.csv",
        [["1", "a" * 16, "0", "88", "x"]],
    )
    write_results(results, [["1", "a" * 16, "1", "57", "x"]])
    assert rc.main([str(results)]) == 1
    assert not (tmp_path / "r_rescored.csv").exists()


# Comment 1: refuse to write through a symlink; write atomically.
def test_refuses_symlinked_destination(tmp_path):
    results = tmp_path / "r.csv"
    write_results(results, [["1", "a" * 16, "0", "60", "Tells: none. Human indicators: no."]])
    target = tmp_path / "elsewhere.csv"
    target.write_text("keep me")
    (tmp_path / "r_review.csv").symlink_to(target)
    assert rc.main([str(results)]) == 1
    assert target.read_text() == "keep me"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "out.csv"
    path.write_text("old")
    rc.write_csv_atomically(pd.DataFrame({"a": [1]}), path)
    assert path.read_text() == "a\n1\n"
    assert os.listdir(tmp_path) == ["out.csv"]
