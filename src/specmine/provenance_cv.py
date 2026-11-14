"""Run from the project root: python -m specmine.provenance_cv [output.csv]

Cross-validated text classifiers vs commit-provenance labels, with LLM judges on the same specs."""
import glob, re, sys, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from specmine.train import _build_estimators, _numeric_features
from specmine.data import MARKDOWN_COLUMN
warnings.filterwarnings("ignore")
L = "data/labeled_data/"
ev = pd.read_csv(L + "provenance_evidence.csv", dtype=str).fillna("")
meta = pd.read_parquet("data/data/spec_files.parquet", columns=["file_url_sha16", "spec_tool", "repo_name"])
feat = pd.read_parquet("data/data/spec_content_features.parquet")
files = {re.search(r"-([0-9a-f]{16})\.md$", p).group(1): p for p in glob.glob("data/spec_files/*.md")}
d = ev.merge(meta, left_on="spec_id", right_on="file_url_sha16").merge(feat, on="file_url_sha16")
d[MARKDOWN_COLUMN] = [open(files[s], encoding="utf-8", errors="replace").read() for s in d.spec_id]
assert d.repo_name.is_unique and len(d) == 335
num, cols = _numeric_features(d[list(feat.columns)])
X = pd.concat([num, d[[MARKDOWN_COLUMN]]], axis=1)
LENGTH = ["n_bytes", "n_lines", "n_words"]
FORMAT = ["n_shall", "n_gherkin", "n_scenario_blocks", "n_delta_headers", "has_ears", "has_gherkin",
          "has_requirements", "has_purpose", "has_mandatory_marker", "has_user_scenarios", "has_success_criteria"]
est = _build_estimators(42)
MODELS = {
    "LR, all features": ("logistic_regression", cols),
    "RF, all features": ("random_forest", cols),
    "LR, length only": ("logistic_regression", LENGTH),
    "LR, no length or format features": ("logistic_regression", [c for c in cols if c not in LENGTH + FORMAT]),
    "Naive Bayes, char n-grams": ("naive_bayes", [MARKDOWN_COLUMN]),
    "LR, char n-grams + n_words": ("markdown_plus_feature", [MARKDOWN_COLUMN, "n_words"]),
}
def judge(path):
    r = pd.read_csv(path, dtype=str, skipinitialspace=True)
    r.columns = r.columns.str.strip()
    r = r.apply(lambda s: s.str.strip()).drop_duplicates("spec_id")
    r = r[r.label.isin(["0", "1"]) & r.confidence.notna()]
    c = r.confidence.astype(float) / 100
    return pd.Series(np.where(r.label == "1", c, 1 - c), index=r.spec_id), r.set_index("spec_id").label.astype(int)
c1 = pd.read_csv(L + "claude_labels1.csv", dtype=str)
JUDGES = {"Claude (rubric prompt)": L + "claude_labels1.csv", "Gemini run 1": L + "gemini_labels1.csv",
          "Gemini run 2": L + "gemini_labels2.csv", "Gemini run 3": L + "gemini_labels3.csv"}
GENERIC = {"adhoc_other", "docs_embedded", "root_bare", "specs_generic", "dot_specs"}
dn = d.evidence_level.isin(["direct", "none"])
TASKS = {
    "A: direct evidence vs none": (dn, d.evidence_level.eq("direct")),
    "B: same, ordinary paths only": (dn & d.spec_tool.isin(GENERIC), d.evidence_level.eq("direct")),
}
rng = np.random.default_rng(0)
def boot_ci(y, s, n=2000):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])
rows = []
for task, (mask, ypos) in TASKS.items():
    Xt, y, ids = X[mask].reset_index(drop=True), ypos[mask].astype(int).to_numpy(), d.spec_id[mask].to_numpy()
    R = 10
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(Xt, y))
    for name, (key, fc) in MODELS.items():
        P = np.zeros((R, len(y)))
        for k, (tr, te) in enumerate(splits):
            m = clone(est[key]).fit(Xt.iloc[tr][fc], y[tr])
            P[k // 5, te] = m.predict_proba(Xt.iloc[te][fc])[:, 1]
        aucs = [roc_auc_score(y, p) for p in P]
        s = P.mean(0); lo, hi = boot_ci(y, s)
        bal = np.mean([balanced_accuracy_score(y, p >= 0.5) for p in P])
        rows.append(dict(task=task, n=len(y), n_ai=int(y.sum()), method=name, kind="classifier, 5-fold CV x10",
                         auc=np.mean(aucs), auc_lo=lo, auc_hi=hi, balanced_acc=bal))
    for name, path in JUDGES.items():
        score, lab = judge(path)
        keep = np.isin(ids, score.index)
        yy, ss = y[keep], score.reindex(ids[keep]).to_numpy()
        lo, hi = boot_ci(yy, ss)
        rows.append(dict(task=task, n=int(keep.sum()), n_ai=int(yy.sum()), method=name, kind="LLM judge, no training",
                         auc=roc_auc_score(yy, ss), auc_lo=lo, auc_hi=hi,
                         balanced_acc=balanced_accuracy_score(yy, lab.reindex(ids[keep]).to_numpy())))
r = pd.DataFrame(rows)
out = sys.argv[1] if len(sys.argv) > 1 else L + "provenance_cv_results.csv"
r.to_csv(out, index=False)
pd.set_option("display.width", 220)
for t, g in r.groupby("task", sort=False):
    print(f"\n{t}")
    g = g.assign(ci=g.apply(lambda x: f"{x.auc_lo:.2f}-{x.auc_hi:.2f}", axis=1))
    print(g[["method", "kind", "n", "n_ai", "auc", "ci", "balanced_acc"]].round(2).to_string(index=False))
