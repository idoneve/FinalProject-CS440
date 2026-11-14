"""Helpers for reviewing specs by hand.

Two independent kinds of evidence are produced here:

* ``provenance_evidence`` reads repository and commit metadata only (never the
  spec text), so it can serve as a check on text-based labels.
* ``text_tells`` scans the Markdown for the surface features named in the
  labeling rubric so a reviewer can see them highlighted in context.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

# --------------------------------------------------------------------------
# Provenance evidence (metadata only)
# --------------------------------------------------------------------------

EVIDENCE_LEVELS = ("direct", "tool", "none")

_AGENT_NAMES = (
    r"claude|anthropic|copilot|cursor|caffeine|devin|codex|openai|chatgpt|gemini|"
    r"jules|aider|windsurf|codeium|opencode|openhands|kiro|trae|qwen|glm|z\.ai|"
    r"ampcode|factory-droid|replit|lovable|bolt\.new"
)
_AGENT_RE = re.compile(rf"(?i)(?<![a-z])(?:{_AGENT_NAMES})(?![a-z])")
_TRAILER_RE = re.compile(r"(?im)^\s*co-authored-by:\s*(.+?)\s*$")
_GENERATED_RE = re.compile(
    r"(?i)generated (?:with|by) \[?([a-z][\w .-]{1,30}?)\]?(?:\(|\s*$|[.\n])"
)
_EMAIL_RE = re.compile(r"\s*<[^>]*>")
# Commit identities used by coding agents. Matched exactly, because people are
# called Claude too.
_AGENT_IDENTITIES = {
    "claude", "claude code", "copilot", "github copilot", "copilot-swe-agent",
    "cursor", "cursor agent", "cursoragent", "devin", "devin ai",
    "devin-ai-integration", "google-labs-jules", "jules", "codex", "openai codex",
    "chatgpt-codex-connector", "openhands", "openhands-agent", "aider", "caffeine",
    "gemini", "gemini-cli", "kiro", "opencode", "amp", "replit-agent", "lovable-dev",
}

# Path-attributed tools whose normal workflow has a coding agent draft the spec.
AGENT_DRIVEN_TOOLS = {
    "openspec": "OpenSpec",
    "speckit": "Spec Kit",
    "kiro": "Kiro",
    "caffeine": "Caffeine",
    "trae": "Trae",
    "claude_code": "Claude Code",
    "codex": "Codex",
    "opencode": "opencode",
    "agent_os": "Agent OS",
    "conductor": "Conductor",
    "cursor": "Cursor",
    "bmad": "BMAD",
}

_COMMITS = (("first", "file_first_commit"), ("last", "file_last_commit"))


def _text(row: Mapping, key: str) -> str:
    value = row.get(key) if hasattr(row, "get") else None
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _is_agent_identity(name: str) -> bool:
    return name.strip().lower().removesuffix("[bot]") in _AGENT_IDENTITIES


def _strip_email(value: str) -> str:
    return _EMAIL_RE.sub("", value).strip()


def provenance_evidence(row: Mapping) -> dict:
    """Summarize AI-involvement evidence from repo/commit metadata for one spec.

    Returns ``{"level", "signals", "context"}`` where ``level`` is ``direct``
    (an AI agent is named as author or co-author of a commit touching the
    file), ``tool`` (the file lives in an agent-driven spec workflow), or
    ``none``. ``none`` means no evidence was recorded, not that a human wrote
    the file: most commits carry no trailer either way.
    """
    direct: list[str] = []
    tool: list[str] = []
    context: list[str] = []

    for which, prefix in _COMMITS:
        message = _text(row, f"{prefix}_message")
        for trailer in _TRAILER_RE.findall(message):
            if _AGENT_RE.search(trailer):
                direct.append(f"{which} commit co-authored by {_strip_email(trailer)}")
        for name in _GENERATED_RE.findall(message):
            if _AGENT_RE.search(name):
                direct.append(f"{which} commit says generated with {name.strip()}")
        for role in ("author", "committer"):
            names = {
                _text(row, f"{prefix}_{role}"),
                _text(row, f"{prefix}_{role}_login"),
            } - {""}
            agent_names = sorted(n for n in names if _is_agent_identity(n))
            if agent_names:
                direct.append(f"{which} commit {role} is {agent_names[0]}")
            elif _text(row, f"{prefix}_{role}_type") == "Bot" and names:
                context.append(f"{which} commit {role} is a bot ({sorted(names)[0]})")
        body = _TRAILER_RE.sub("", message)
        mentioned = sorted({m.lower() for m in _AGENT_RE.findall(body)})
        if mentioned:
            context.append(f"{which} commit message mentions {', '.join(mentioned)}")

    spec_tool = _text(row, "spec_tool")
    if spec_tool in AGENT_DRIVEN_TOOLS:
        tool.append(f"path attributed to {AGENT_DRIVEN_TOOLS[spec_tool]} workflow")
    if _text(row, "feature_branch"):
        tool.append("has a Spec Kit 'Feature Branch' header")
    path = _text(row, "file_path")
    hidden = re.match(r"\.(claude|cursor|kiro|trae|codex|specify|opencode|agent-os)\b", path)
    if hidden:
        tool.append(f"file lives under .{hidden.group(1)}/")

    direct = list(dict.fromkeys(direct))
    tool = list(dict.fromkeys(tool))
    context = list(dict.fromkeys(context))
    level = "direct" if direct else "tool" if tool else "none"
    return {"level": level, "signals": direct + tool, "context": context}


def provenance_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """One provenance row per spec in ``metadata``."""
    records = []
    for row in metadata.to_dict("records"):
        evidence = provenance_evidence(row)
        records.append(
            {
                "repo_id": row.get("repo_id"),
                "spec_id": row.get("file_url_sha16"),
                "evidence_level": evidence["level"],
                "signals": "; ".join(evidence["signals"]),
                "context": "; ".join(evidence["context"]),
            }
        )
    return pd.DataFrame(
        records,
        columns=["repo_id", "spec_id", "evidence_level", "signals", "context"],
    )


# --------------------------------------------------------------------------
# Text tells (rubric categories, for highlighting only)
# --------------------------------------------------------------------------

# code -> (name, highlight colour, what to ask yourself)
TELL_CATEGORIES = {
    "A": ("Chat residue", "#ffb3b3", "Is this text addressed to a requester?"),
    "B": ("Tool or prompt residue", "#ffcf99", "Attribution, pasted prompt or unfilled variable?"),
    "C": ("Template scaffolding", "#ffe999", "Annotation a careful author would delete?"),
    "D": ("Bold-label bullets", "#c9e7a8", "Are most prose bullets '**Label**: text'?"),
    "E": ("Emoji as structure", "#a8e7d8", "Emoji in headings or as status markers?"),
    "F": ("Boilerplate sections", "#a8d4f0", "Full boilerplate set for a small feature?"),
    "G": ("Assistant filler", "#c4b8f5", "Filler vocabulary with no personal voice?"),
    "H": ("Unsupported precision", "#f0b8e4", "A number with no source or rationale?"),
    "I": ("Typeset characters", "#d9d9d9", "Arrows, curly quotes or box drawing?"),
    "HUMAN": ("Human indicator", "#9be89b", "Typo, first person, named person, dated note, shorthand?"),
}

_KEYWORDS = {
    "when", "then", "given", "and", "but", "if", "while", "where", "shall",
    "must", "should", "当", "则", "且", "如果", "那么", "并且", "假设", "但是",
}
_EMOJI = "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2B06\u2705\u274C]"

_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("A", re.compile(
        r"(?i)\b(?:here(?: is|'s) (?:the|a|an|your)\b[^\n]{0,40}|sure[!,]|certainly[!,]|"
        r"let me know if[^\n]{0,40}|i(?:'ve| have) (?:created|updated|added|generated)\b|"
        r"as an ai\b|based on your (?:request|description|input)|feel free to|"
        r"reply [\"“][^\"”\n]{1,30}[\"”]|would you like me to)"
    )),
    ("B", re.compile(
        r"(?i)(?:generated (?:by|with) \[?(?:claude|copilot|chatgpt|cursor|gemini|kiro|"
        r"codex|ai)\b[^\n]{0,30}|\$ARGUMENTS\b|\*\*Input\*\*:\s*User (?:description|story)[^\n]{0,60}|"
        r"co-authored-by: (?:claude|copilot)[^\n]{0,40})"
    )),
    ("C", re.compile(
        r"(?i)(?:\*?\((?:mandatory|optional)\)\*?|\[NEEDS CLARIFICATION[^\]\n]{0,80}\]?|"
        r"ACTION REQUIRED[^\n]{0,40}|TBD - created by archiving change[^\n]{0,40}|"
        r"<!--[^\n]{0,80}-->)"
    )),
    ("E", re.compile(rf"(?m)^(?:#{{1,6}}\s+|\s*[-*+]\s+(?:\[.\]\s+)?){_EMOJI}")),
    ("F", re.compile(
        r"(?im)^#{1,4}\s+(?:\d+[.)]?\s+)?(?:overview|goals?|non-goals?|risks?(?: (?:and|&) "
        r"mitigations?)?|mitigations?|success (?:metrics|criteria)|future (?:enhancements|work)|"
        r"conclusion|summary|assumptions|out of scope)\b[^\n]*"
    )),
    ("G", re.compile(
        r"(?i)\b(?:comprehensive(?:ly)?|robust(?:ly)?|seamless(?:ly)?|leverag(?:e|es|ing)|"
        r"ensur(?:e|es|ing) that|streamlin(?:e|es|ed|ing)|cutting-edge|delve|"
        r"not (?:just|only) [^.\n]{1,40}?,? but)\b"
    )),
    ("H", re.compile(
        r"(?i)(?:\b\d{2,3}(?:\.\d+)?\s?%|(?:<|>|under|within|less than|at least)\s?\d+(?:\.\d+)?\s?"
        r"(?:ms|milliseconds?|seconds?|s\b|minutes?)|\bphase \d+\s*\((?:weeks?|days?)[^)\n]{0,20}\)|"
        r"\bweeks? \d+(?:\s?[-–]\s?\d+)?)"
    )),
    ("I", re.compile("[\u2190-\u21FF\u2500-\u257F\u201C\u201D\u2018\u2019]+")),
    ("HUMAN", re.compile(
        r"(?:\b(?:TODO|FIXME|XXX|HACK)\b[^\n]{0,60}|(?i:\b(?:i think|i'm|i am|i want|i need|"
        r"my |imo|iirc|tbh|idk|afaik|fwiw|btw|lol|hmm+|dunno|kinda|gonna|wanna)\b)|\?\?+)"
    )),
)
_BOLD_BULLET = re.compile(
    r"(?m)^[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?"
    r"(\*\*([^*\n]{1,60}?)(?:[:：]\*\*|\*\*[ \t]*[:：—–-]))"
)
_ANY_BULLET = re.compile(r"(?m)^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\S")
_EM_DASH = "\u2014"


def _is_descriptive_label(label: str) -> bool:
    label = label.strip(" :：").lower()
    if not label or label in _KEYWORDS:
        return False
    return not re.match(r"^[a-z]{1,5}-?\d+", label)


def find_spans(markdown: str) -> list[tuple[int, int, str]]:
    """Non-overlapping ``(start, end, category)`` highlights, in text order."""
    found: list[tuple[int, int, str]] = []
    for category, pattern in _PATTERNS:
        found.extend((m.start(), m.end(), category) for m in pattern.finditer(markdown))
    for match in _BOLD_BULLET.finditer(markdown):
        if _is_descriptive_label(match.group(2)):
            found.append((match.start(1), match.end(1), "D"))
    order = list(TELL_CATEGORIES)
    found.sort(key=lambda span: (span[0], order.index(span[2]), -span[1]))
    spans, cursor = [], 0
    for start, end, category in found:
        if start >= cursor and end > start:
            spans.append((start, end, category))
            cursor = end
    return spans


def text_tells(markdown: str, max_examples: int = 3) -> pd.DataFrame:
    """Count rubric surface features in one spec, with short verbatim examples."""
    grouped: dict[str, list[str]] = {code: [] for code in TELL_CATEGORIES}
    for category, pattern in _PATTERNS:
        grouped[category].extend(m.group(0).strip() for m in pattern.finditer(markdown))
    bullets = len(_ANY_BULLET.findall(markdown))
    grouped["D"] = [
        m.group(1) for m in _BOLD_BULLET.finditer(markdown)
        if _is_descriptive_label(m.group(2))
    ]
    notes = {code: "" for code in TELL_CATEGORIES}
    if bullets and grouped["D"]:
        notes["D"] = f"file has {bullets} list items"
    em_dashes = markdown.count(_EM_DASH)
    if em_dashes >= 5:
        notes["G"] = f"plus {em_dashes} em dash(es)"
    rows = []
    for code, (name, _, question) in TELL_CATEGORIES.items():
        hits = grouped[code]
        if not hits and not notes[code]:
            continue
        examples = list(dict.fromkeys(hit[:70] for hit in hits))[:max_examples]
        rows.append(
            {
                "category": code,
                "feature": name,
                "hits": len(hits),
                "note": notes[code],
                "examples": " | ".join(examples),
                "ask yourself": question,
            }
        )
    return pd.DataFrame(
        rows, columns=["category", "feature", "hits", "note", "examples", "ask yourself"]
    )


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

_LEVEL_STYLE = {
    "direct": ("#b42318", "Direct evidence of AI involvement"),
    "tool": ("#b54708", "Agent-driven tool workflow"),
    "none": ("#475467", "No provenance evidence recorded"),
}
_BOX = (
    "border:1px solid #d0d5dd;border-radius:6px;padding:10px 12px;margin:8px 0;"
    "background:#ffffff;color:#101828;font-size:13px;line-height:1.5"
)
_KEY_FEATURES = (
    "n_lines", "n_words", "n_headings", "n_list_items", "n_tables", "n_code_fences",
    "n_shall", "n_gherkin", "n_scenario_blocks", "n_delta_headers", "n_checkboxes",
    "n_checked", "n_needs_clarification", "n_todo", "has_mandatory_marker",
    "has_unfilled_placeholder", "is_tiny", "lang", "content_family",
)


def _esc(value) -> str:
    return html.escape(str(value))


def _link(url: str, label: str) -> str:
    if not url.startswith("https://"):
        return _esc(label)
    return f'<a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(label)}</a>'


def _date(value: str) -> str:
    return value[:10] if value else "unknown"


def _highlight_agents(text: str) -> str:
    escaped = _esc(text)
    return _AGENT_RE.sub(
        lambda m: f'<mark style="background:#ffb3b3;color:#101828">{m.group(0)}</mark>',
        escaped,
    )


def render_metadata_card(
    row: Mapping, evidence: dict | None = None, show_commits: bool = True
) -> str:
    """Repository, file and commit facts for one spec, tolerant of missing columns.

    Set ``show_commits=False`` and leave ``evidence`` unset for a blind review:
    commit messages and authors are what reveal AI involvement.
    """
    lines = []
    repo_name = _text(row, "repo_name")
    repo_bits = [
        f"<b>{_link(_text(row, 'repo_url'), repo_name or 'unknown repo')}</b>",
        f"repo_id {_esc(_text(row, 'repo_id'))}",
    ]
    for key, label in (("repo_stars", "stars"), ("repo_forks", "forks")):
        if _text(row, key):
            repo_bits.append(f"{_esc(_text(row, key))} {label}")
    for key in ("repo_language", "owner_type"):
        if _text(row, key):
            repo_bits.append(_esc(_text(row, key)))
    if _text(row, "repo_created_at"):
        repo_bits.append(f"created {_date(_text(row, 'repo_created_at'))}")
    lines.append(" · ".join(repo_bits))
    if _text(row, "repo_description"):
        lines.append(f"<i>{_esc(_text(row, 'repo_description'))}</i>")

    file_bits = [
        _link(_text(row, "file_url"), _text(row, "file_path") or "spec file"),
        f"spec_id {_esc(_text(row, 'file_url_sha16'))}",
    ]
    for key, label in (("spec_tool", "tool"), ("spec_role", "role")):
        if _text(row, key):
            file_bits.append(f"{label}: <b>{_esc(_text(row, key))}</b>")
    if _text(row, "file_first_commit_at"):
        file_bits.append(f"first committed {_date(_text(row, 'file_first_commit_at'))}")
    if _text(row, "file_total_commits"):
        file_bits.append(f"{_esc(_text(row, 'file_total_commits'))} commit(s) to this file")
    siblings = [
        key[4:] for key in ("has_tasks", "has_plan", "has_proposal", "has_design")
        if _text(row, key) in {"1", "1.0"}
    ]
    if siblings:
        file_bits.append("sibling docs: " + ", ".join(siblings))
    lines.append(" · ".join(file_bits))

    seen_messages = set()
    for which, prefix in _COMMITS if show_commits else ():
        message = _text(row, f"{prefix}_message").strip()
        if not message or message in seen_messages:
            continue
        seen_messages.add(message)
        author = _text(row, f"{prefix}_author") or "unknown author"
        login = _text(row, f"{prefix}_author_login")
        who = f"{author} (@{login})" if login and login != author else author
        label = "First and last commit" if (
            which == "first"
            and message == _text(row, "file_last_commit_message").strip()
        ) else f"{which.capitalize()} commit"
        lines.append(
            f"<b>{label}</b> {_date(_text(row, f'{prefix}_at'))} by "
            f"{_highlight_agents(who)}"
            f'<pre style="margin:4px 0 0 0;padding:6px 8px;background:#f2f4f7;color:#101828;'
            f'white-space:pre-wrap;max-height:160px;overflow:auto">'
            f"{_highlight_agents(message)}</pre>"
        )

    if evidence is not None:
        colour, title = _LEVEL_STYLE[evidence["level"]]
        detail = "".join(f"<li>{_esc(s)}</li>" for s in evidence["signals"])
        extra = "".join(
            f'<li style="color:#475467">{_esc(s)}</li>' for s in evidence["context"]
        )
        lines.append(
            f'<span style="background:{colour};color:#ffffff;padding:2px 8px;'
            f'border-radius:10px;font-weight:600">{title}</span>'
            + (f'<ul style="margin:4px 0 0 0">{detail}{extra}</ul>' if detail or extra else "")
        )
    return f'<div style="{_BOX}">' + "<br>".join(lines) + "</div>"


def render_feature_strip(features: Mapping) -> str:
    """Compact one-line view of the precomputed content features that are set."""
    bits = []
    for key in _KEY_FEATURES:
        value = _text(features, key)
        if value and value not in {"0", "0.0"}:
            bits.append(f"{_esc(key)}=<b>{_esc(value)}</b>")
    if not bits:
        return ""
    return f'<div style="{_BOX}">' + " · ".join(bits) + "</div>"


def render_legend(categories: Iterable[str] | None = None) -> str:
    codes = list(categories) if categories is not None else list(TELL_CATEGORIES)
    chips = "".join(
        f'<span style="background:{TELL_CATEGORIES[c][1]};color:#101828;padding:1px 6px;'
        f'margin-right:6px;border-radius:3px">{_esc(c)} {_esc(TELL_CATEGORIES[c][0])}</span>'
        for c in codes
    )
    return f'<div style="margin:6px 0;font-size:12px">{chips}</div>'


def render_markdown(markdown: str, highlight: bool = True) -> str:
    """Escaped, line-numbered Markdown with rubric features highlighted."""
    pieces, cursor = [], 0
    for start, end, category in find_spans(markdown) if highlight else []:
        pieces.append(_esc(markdown[cursor:start]))
        name, colour, _ = TELL_CATEGORIES[category]
        pieces.append(
            f'<mark title="{_esc(category)}: {_esc(name)}" '
            f'style="background:{colour};color:#101828">{_esc(markdown[start:end])}</mark>'
        )
        cursor = end
    pieces.append(_esc(markdown[cursor:]))
    body = "".join(pieces).split("\n")
    width = len(str(len(body)))
    numbered = "\n".join(
        f'<span style="color:#98a2b3;user-select:none">{index:>{width}} </span>{line}'
        for index, line in enumerate(body, start=1)
    )
    return (
        '<pre style="background:#ffffff;color:#101828;border:1px solid #d0d5dd;'
        'border-radius:6px;padding:10px 12px;white-space:pre-wrap;font-size:12px;'
        f'line-height:1.45">{numbered}</pre>'
    )


# --------------------------------------------------------------------------
# Command line: write the provenance table for a labels file
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from specmine.data import load_data

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=root / "data/labeled_data/real_labels.csv")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument(
        "--output", type=Path, default=root / "data/labeled_data/provenance_evidence.csv"
    )
    args = parser.parse_args(argv)

    labels = pd.read_csv(args.labels, dtype=str)
    metadata = load_data(data_dir=args.data_dir, local_data_only=True)
    selected = metadata.loc[metadata["file_url_sha16"].isin(labels["spec_id"])]
    selected = selected.drop_duplicates(subset=["file_url_sha16"])
    table = provenance_table(selected)
    order = {spec_id: index for index, spec_id in enumerate(labels["spec_id"])}
    table = table.sort_values("spec_id", key=lambda s: s.map(order)).reset_index(drop=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} rows to {args.output}")
    print(table["evidence_level"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
