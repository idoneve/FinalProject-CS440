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

Choose the same Python environment for installation and the notebook kernel.
In VS Code, use **Select Kernel** in each notebook. To confirm which interpreter
the kernel is using, run `import sys; print(sys.executable)` in a code cell.
Installing the package in that environment as shown above is recommended;
some notebooks also add the repository's `src/` folder to their import path.
If a notebook reports a missing dependency, install the requirements into the
selected environment with
`<python-interpreter> -m pip install -r requirements.txt`.

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
`notebooks/data_init.ipynb` to reuse the metadata and content-feature parquet
files already in `notebooks/data/`, or download them there if either is missing.
It then downloads or reuses a configurable random sample of raw spec Markdown
(300 files by default), with one unique spec per repository, and chooses an
unreviewed repository from that downloaded sample to display its Markdown,
exact feature row, and metadata. Repeated spec IDs and repositories are excluded
from the sample; failed downloads are replaced with candidates from other
repositories. Change `SPEC_SAMPLE_SIZE` or `SEED` near the top of the notebook.
The sample manifest is saved as `notebooks/data/spec_sample.csv`.

The Python API can also be told to use the local cache without a network
fallback:

```python
from specmine.data import load_data, load_feature_data

metadata = load_data(local_data_only=True)
features = load_feature_data(local_data_only=True)
```

By default, local-only loading looks under this repository's
`notebooks/data/data/`; pass `data_dir=...` to use a different cache
directory. If a parquet file is missing, local-only mode raises
`FileNotFoundError` and never falls back to the network. The default remains
`local_data_only=False`, preserving remote loading. Local-only mode cannot be
combined with `curated_only=True`, which downloads raw spec files.

Use `notebooks/data_init.ipynb` for data initialization, sampling, and review.
Use `notebooks/show_specs.ipynb` to look up IDs and inspect all repository/spec
properties, matched features, and Markdown. Missing parquet files and requested
spec Markdown are downloaded into the data directory. Markdown is shown as
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
notebooks. It includes both `spec_id` and the corresponding `repo_id` for
repository-level leakage checks. Its deterministic confidence values are test
data, not verified assessments. Do not copy fake rows into the real-label file
or interpret fake confidence values as reviewed assessments. Use
`notebooks/predict.ipynb` to
train and evaluate one selected baseline using matched, nonblank labels, then
refit on all labeled specs and
predict a random sample of 100 specs not listed in the labels CSV. Specs
without precomputed features are replaced from the unlabeled pool; if the full
requested sample is not available, the notebook stops rather than saving an
undersized result. The sample size and model are configurable. Each run
publishes predictions to a unique CSV filename after a complete temporary-file
write, so concurrent runs do not overwrite one another.

## Notebook Quick Start

Open notebooks from the repository and run their code cells from top to
bottom. The data notebooks need an internet connection to fetch the parquet
files and GitHub spec Markdown; once cached, the labeling, prediction, and
results workflows read the parquet files locally.

| Notebook | Purpose and how to use it |
| --- | --- |
| `notebooks/data_init.ipynb` | Initializes or reuses the metadata and feature parquet cache, downloads/reuses a random sample of one spec per repository, writes `notebooks/data/spec_sample.csv`, and displays one unreviewed repository from that sample. Set `SPEC_SAMPLE_SIZE` and `SEED` near the top; the default sample size is 300 and `SEED = None` chooses a new sample each run. |
| `notebooks/show_specs.ipynb` | Looks up all distinct specs for one or more `REPO_IDS`, or specific `SPEC_IDS` (list or comma-separated string). It downloads missing parquet or Markdown files, then displays all repository/spec properties, matched features, and Markdown. |
| `notebooks/predict.ipynb` | Trains and evaluates baselines, refits a selected model using all nonblank labels, then predicts an unlabeled sample. Set `LABELS_CSV`, `MODEL_NAME`, and `PREDICTION_SAMPLE_SIZE` near the top to change the inputs. It saves models under `models/` and a uniquely named prediction CSV under `notebooks/`. |
| `notebooks/results.ipynb` | Builds the dataset and repository-disjoint split summaries, fits fresh confidence-weighted baselines, evaluates them, and saves figures under `notebooks/figures/`. By default it uses the fake-label fixture and reports agreement with those supplied labels, not verified authorship accuracy. |

For labeling, ambiguous evidence should be recorded with blank
`label` and `confidence` fields. Nonblank labels must be `0` or `1`, and their
confidence must be from 0 to 100. Do not copy rows from `fake_labels.csv` into
`real_labels.csv`; the fake labels and their confidence values are test data.
The `data_init.ipynb` workflow selects its review repository from the downloaded
sample and reports an error if no eligible downloaded spec is available; increase
`SPEC_SAMPLE_SIZE` if the sample contains no repository that is still available
to review.

## Project Structure

```text
FinalProject-CS440/
├── README.md                    # Installation, workflows, and project guide
├── examples/
│   ├── fake_labels.csv          # Test labels; not verified authorship data
│   └── real_labels.csv          # Human-reviewed labels
├── models/                      # Saved scikit-learn model pipelines
├── notebooks/
│   ├── data_init.ipynb          # Combined sample and review workflow
│   ├── show_specs.ipynb         # Inspect specs by repo or spec ID
│   ├── predict.ipynb            # Train, evaluate, and predict
│   ├── results.ipynb            # Evaluation summaries and plots
│   ├── data/                    # Local parquet cache, Markdown, and manifest
│   └── figures/                 # Generated result plots
├── src/specmine/
│   ├── data.py                  # Dataset loading, caching, and downloads
│   ├── features.py              # Feature preparation
│   ├── labeling.py              # Leakage-safe review selection and matching
│   ├── labels.py                # Label CSV parsing and validation
│   ├── train.py                 # Training and repository-disjoint splitting
│   └── evaluate.py              # Model evaluation and prediction outputs
├── tests/                       # Offline unit tests and opt-in network tests
├── DATA_DICTIONARY.md           # SpecMine dataset and feature reference
├── schema.sql                   # Database schema reference
├── pyproject.toml               # Package metadata and development dependencies
├── requirements.txt             # Runtime dependency list
├── LICENSE
└── LICENSE-DATA
```

The parquet cache, downloaded Markdown, sample manifest, and generated plots
are local workflow data. Run an initializer again to refresh or create the
cache; other notebooks may create prediction CSVs and model files when run.

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
  --local-data-only \
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
  --labels-csv labels.csv --split test --local-data-only \
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
