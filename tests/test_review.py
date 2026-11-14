import pandas as pd

from specmine import review


def test_provenance_direct_from_ai_coauthor_trailer():
    row = {
        "file_first_commit_message": (
            "Add spec\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
        ),
        "file_first_commit_author": "Dana",
        "spec_tool": "openspec",
    }
    evidence = review.provenance_evidence(row)
    assert evidence["level"] == "direct"
    assert evidence["signals"][0] == "first commit co-authored by Claude Opus 4.8"
    assert "path attributed to OpenSpec workflow" in evidence["signals"]


def test_provenance_ignores_human_coauthors_and_people_named_claude():
    row = {
        "file_first_commit_message": "Fix\n\nCo-authored-by: Sam Lee <sam@example.com>",
        "file_first_commit_author": "claude.hsieh",
        "file_first_commit_author_login": "ClaudeHsieh",
        "spec_tool": "adhoc_other",
    }
    assert review.provenance_evidence(row)["level"] == "none"


def test_provenance_tool_level_and_missing_columns():
    assert review.provenance_evidence({"spec_tool": "speckit"})["level"] == "tool"
    assert review.provenance_evidence({"file_last_commit_message": float("nan")}) == {
        "level": "none",
        "signals": [],
        "context": [],
    }


def test_provenance_table_has_one_row_per_spec():
    metadata = pd.DataFrame(
        [
            {"repo_id": 1, "file_url_sha16": "a", "spec_tool": "kiro"},
            {"repo_id": 2, "file_url_sha16": "b", "spec_tool": "root_bare"},
        ]
    )
    table = review.provenance_table(metadata)
    assert table["spec_id"].tolist() == ["a", "b"]
    assert table["evidence_level"].tolist() == ["tool", "none"]


MARKDOWN = """# Spec

## Requirements *(mandatory)*

- **Why this matters**: users need a robust flow
- **WHEN** the user logs in
- **FR-001**: System MUST respond in under 200ms
- plain bullet, TODO check with Sam
"""


def test_text_tells_counts_rubric_features_but_not_requirement_keywords():
    tells = review.text_tells(MARKDOWN).set_index("category")
    assert tells.loc["D", "hits"] == 1
    assert tells.loc["D", "examples"] == "**Why this matters**:"
    assert tells.loc["C", "hits"] == 1
    assert tells.loc["G", "examples"] == "robust"
    assert tells.loc["H", "examples"] == "under 200ms"
    assert tells.loc["HUMAN", "hits"] == 1
    assert "A" not in tells.index


def test_render_markdown_escapes_html_and_marks_spans():
    rendered = review.render_markdown("<script>x</script> leverage →\n")
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert rendered.count("<mark") == 2
    assert review.render_markdown("leverage", highlight=False).count("<mark") == 0


def test_metadata_card_blind_mode_hides_commits():
    row = {
        "repo_name": "owner/repo",
        "file_path": "spec.md",
        "file_url": "https://github.com/owner/repo/blob/main/spec.md",
        "file_first_commit_message": "Co-Authored-By: Claude <noreply@anthropic.com>",
    }
    assert "Co-Authored-By" in review.render_metadata_card(row)
    assert "Co-Authored-By" not in review.render_metadata_card(row, show_commits=False)
