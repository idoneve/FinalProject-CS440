# CS440 - Intro to AI Final Project: AI vs Human Specification Agent

## Project Overview

Our team is participating in the **MSR 2027 Mining Challenge** using the **SpecMine** dataset
Specification can be found here: **<https://2027.msrconf.org/track/msr-2027-mining-challenge>**

Software developers are increasingly using AI agents to write technical specification files (`spec.md`). This project aims to build an AI classifier that analyzes specification files and predicts whether **AI was used at any point in creating a specific file**.

## Installation

From the `FinalProject-CS440` directory (the directory containing this README),
install the package and its dependencies into the Python environment you intend
to use:

```bash
cd path/to/FinalProject-CS440
python -m pip install -e ".[dev]"
```

The notebooks add this repository's `src/` folder to their import path, so they
do not rely on an editable install to find `specmine`. They print the Python
interpreter used by their kernel; if imports fail for missing dependencies,
install them into that interpreter with
`<python-from-notebook-output> -m pip install -r requirements.txt`. In VS Code,
choose the intended environment using **Select Kernel** in the notebook.

Import `load_data` from the installed `specmine` package:

```bash
python -c 'from specmine.data import load_data; df = load_data(sample_size=5, seed=0); print(df.head() if df is not None else "Dataset load failed")'
```

The default call reads the full parquet dataset. `sample_size` limits the
returned DataFrame but does not reduce the remote parquet download or its
memory use. Use `curated_only=True` to load the curated repositories and
download their spec files into `./data`:

```python
from specmine.data import load_data

df = load_data(curated_only=True, sample_size=500, seed=0)
```

The loader needs an internet connection to access the dataset. Use
`notebooks/label_specs.ipynb` to choose an unlabeled repository from the full
metadata parquet and display one random spec's raw Markdown and precomputed
content features. Only that selected spec is downloaded. Markdown is shown as
escaped text so remote images and embedded HTML are not rendered. The curated
downloader accepts only HTTPS URLs on `github.com` and
`raw.githubusercontent.com` and validates every redirect before following it.
Record one reviewed row in `examples/real_labels.csv` using
`repo_id,spec_id,label,confidence`. `repo_id` identifies the repository for
leakage prevention; `spec_id` identifies the exact spec and remains the model
feature join key. Use `1` if AI was used at any point, `0` if it was not, and
leave label and confidence blank if evidence is ambiguous. A confidence score
is a numeric value from 0 to 100 describing certainty in the assigned label.
The notebook rejects repositories shared by `real_labels.csv` and
`fake_labels.csv`, and excludes repositories already in either file from
selection. Fake-label spec IDs are resolved to repository IDs through the
metadata parquet. The displayed feature table is content-based; commit history
is not used for labeling or by the current training/evaluation workflow.

`examples/fake_labels.csv` is the test fixture used by the training and results
notebooks. Its deterministic confidence values are test data, not verified
assessments. Do not copy fake rows into the real-label file or interpret fake
confidence values as reviewed assessments. Use `notebooks/predict.ipynb` to
train and evaluate one selected baseline using matched, nonblank labels, then
refit on all labeled specs and
predict a random sample of 100 specs not listed in the labels CSV. Specs
without precomputed features are replaced from the unlabeled pool; if the full
requested sample is not available, the notebook stops rather than saving an
undersized result. The sample size and model are configurable. Each run
publishes predictions to a unique CSV filename after a complete temporary-file
write, so concurrent runs do not overwrite one another.

## Training Baselines

The training module compares Logistic Regression and Random Forest baselines
using numeric spec-content features and labels supplied by you. Commit history
is not used by the current training/evaluation workflow. It refuses to train
unless `--labels-csv` is provided. Use `1` when AI was used for any part of
creating a spec, `0` only when no AI was used, and a blank label for ambiguous
cases to exclude. Repositories are kept separate between training and holdout
data.

```bash
python -m specmine.train --labels-csv labels.csv --seed 42 \
  --model-dir models/baselines
```

The training CSV requires `spec_id,label,confidence`: `spec_id` matches
`file_url_sha16`, `label` is `0`, `1`, or blank, and `confidence` is a numeric
score from 0 through 100. The manual-label CSV may additionally include
`repo_id` for repository-level exclusion checks; training continues to join
labels to features by `spec_id`. Every nonblank label must have a confidence
score.
Blank labels are excluded from training. The training code passes
`confidence / 100` as a sample weight, so a score of 0 contributes no weight
and 100 contributes full weight. Holdout metrics remain unweighted. Omit
`--sample-size` to use all matched labeled examples; if supplied, it caps the
matched rows after ID matching. Training reads the full metadata parquet to
ensure labeled IDs are not lost to random sampling, then loads features only
for labeled specs. Omit `--model-dir` to skip saving models. Select one or both
estimators with `--models logistic_regression random_forest`.

The results notebook reports predicted-class confidence as the larger of the
two class probabilities, expressed from 0 to 100, alongside the probability
that a spec is AI-assisted.

The training notebook defaults to `examples/fake_labels.csv` for testing.
Manual labels should be added to `examples/real_labels.csv` in the labeling
notebook; switch `LABELS_CSV` when training with that file. All IDs appearing
in the selected CSV (including blank-label rows) are excluded from the
extrapolation sample. Repeated rows for the same ID and label collapse to one
row; conflicting labels for an ID are treated as blank/ambiguous and reported
with a warning. Prediction CSVs use a unique filename based on the
`notebooks/unlabeled_predictions.csv` output name by default.

## Evaluating a Classifier

The released SpecMine schema does not include AI-authorship labels. Supply a
CSV of manually labeled examples with `spec_id` values matching
`file_url_sha16`, a binary `label` column (`1` if AI was used at any point in
creating the spec, `0` if no AI was used), and a `confidence` score from 0 to
100. Examples without a label match are excluded and counted. Leave ambiguous
examples' label and confidence blank; training excludes them.

```csv
spec_id,label,confidence
0123456789abcdef,1,90
fedcba9876543210,0,75
```

Save a fitted sklearn pipeline with `joblib`, then evaluate a split:

```bash
python -m specmine.evaluate --model-path models/specmine.joblib \
  --labels-csv labels.csv --split test \
  --output-json metrics.json --output-csv predictions.csv
```

Without `--labels-csv`, the command emits predictions only and does not compute
supervised metrics. `--split` accepts `train`, `test`, or `all`; use
`--make-val --split val` to deterministically carve a validation split out of
train. CSV metadata is escaped when it begins with a spreadsheet formula
prefix. Curated downloads use `--data-dir` (default `./data`).

Run the local tests with:

```bash
pytest -v
```

Network smoke tests are skipped by default so the local test command works
offline. To run them explicitly with an internet connection:

```bash
SPECMINE_RUN_NETWORK_TESTS=1 pytest -v -m network
```

## Dataset Attribution & Citations

This project participates in the **MSR 2027 Mining Challenge** and utilizes the `spec_files` schema provided by the **SpecMine** research corpus (2026).

If you use this work or pipeline, please cite both the foundational paper and the Hugging Face data repository (see the full BibTeX entries in the referenced documentation).

### 1. Foundational Research Paper

- **Reference:** *SpecMine: A Large-Scale Corpus of Spec-Driven Development Artifacts* (MSR '27, IEEE) [arXiv:2608.25202 [cs.SE]](https://arxiv.org).

### 2. Data Repository

- **Reference:** *SpecMine Dataset Parquet Mirror* (Hugging Face datasets/ShyAgarwal/specmine) [Dataset](https://huggingface.co/datasets/ShyAgarwal/specmine).
