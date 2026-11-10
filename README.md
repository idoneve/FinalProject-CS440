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

The default call reads `spec_files.parquet` from the local cache when present;
otherwise it downloads that parquet from Hugging Face into
`notebooks/data/data/`. `sample_size` limits the returned DataFrame but does not
reduce the first download or the memory needed to read the full metadata
parquet. `data_dir=...` changes the cache root.

Use `curated_only=True` to load the curated repositories. The filtered metadata
is saved as `curated_spec_files.parquet` and reused on later calls, so the full
metadata parquet is not scanned again. Raw Markdown is downloaded to the cache
root and reused; `sample_size` limits the selected rows/files. With no
`sample_size`, all curated rows are selected. To load curated metadata without
downloading Markdown, set `download_raw_specs=False` and pass the metadata to
`download_random_spec_sample` or `download_spec_files`:

```python
from specmine.data import download_random_spec_sample, load_data

metadata = load_data(
    curated_only=True,
    data_dir="notebooks/data",
    download_raw_specs=False,
)
sample = download_random_spec_sample(
    metadata,
    sample_size=3,
    seed=0,
    data_dir="notebooks/data",
)
```

`notebooks/data_init.ipynb` uses `CURATED_ONLY = True` by default to select the
curated repositories and samples up to `SPEC_SAMPLE_SIZE` (3 by default), with
at most one spec per repository. Set it to `False` to sample from the full
metadata table. Parquets and Markdown files are cache-first: existing files are
reused and missing files are downloaded into `notebooks/data/`. The notebook
chooses an unreviewed repository from the sample and displays its Markdown,
exact feature row, and metadata. Repeated spec IDs and repositories are
excluded; failed downloads are replaced with candidates from other
repositories. The first run and each newly selected, uncached spec may require
network access; rerunning the same seeded sample reuses its local files. Change
`SPEC_SAMPLE_SIZE` or `SEED` near the top of the notebook. The sample manifest
is saved as `notebooks/data/spec_sample.csv`.

Normal `load_data` and `load_feature_data` calls also use the local cache first
and download only a missing required parquet from Hugging Face. For explicitly
strict offline reads, use `local_data_only=True`:

```python
from specmine.data import load_data, load_feature_data

metadata = load_data(local_data_only=True)
features = load_feature_data(local_data_only=True)
```

Local-only loading looks under this repository's `notebooks/data/data/`; pass
`data_dir=...` to use a different cache root. If a parquet file is missing,
local-only mode raises `FileNotFoundError` and never falls back to the network.
`curated_only=True` is cache-first rather than a strict no-network option: it
may fetch missing curated metadata on its first run, then reuse the local
subset. `local_data_only=True` cannot be combined with `curated_only=True`.

Use `notebooks/data_init.ipynb` for cache initialization, sampling, and review.
Use `notebooks/show_specs.ipynb` to look up IDs and inspect all repository/spec
properties, matched features, and Markdown. Only missing parquet files and
requested Markdown are downloaded into the data directory. Markdown is shown
as escaped text so remote images and embedded HTML are not rendered. The raw
spec downloader accepts only HTTPS URLs on `github.com` and
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
`notebooks/predict.ipynb` to train and evaluate a selected baseline using
matched, nonblank labels, refit on all available labeled specs, and predict a
random sample of 100 specs not listed in the labels CSV. Specs without the
selected model's required features or Markdown are replaced from the unlabeled
pool; if the full requested sample is not available, the notebook stops rather
than saving an undersized result. The sample size and model are configurable.
Each run publishes predictions to a unique CSV filename after a complete
temporary-file write, so concurrent runs do not overwrite one another.

## Notebook Quick Start

Open notebooks from the repository and run their code cells from top to
bottom. All notebooks use the same cache-first loaders. An internet connection
is needed only when a required parquet or selected Markdown file is missing;
cached files are reused from `notebooks/data`.

| Notebook | Purpose and how to use it |
| --- | --- |
| `notebooks/data_init.ipynb` | Reuses or fills the cache, samples up to three specs from distinct repositories, writes `notebooks/data/spec_sample.csv`, and displays one unreviewed repository with its Markdown, properties, and features. Set `CURATED_ONLY`, `SPEC_SAMPLE_SIZE`, and `SEED` near the top. |
| `notebooks/show_specs.ipynb` | Looks up all distinct specs for one or more `REPO_IDS`, or specific `SPEC_IDS` (list or comma-separated string). It reuses cached files and downloads only missing metadata, features, or requested Markdown. |
| `notebooks/predict.ipynb` | Trains and evaluates baselines, refits a selected model, then predicts an unlabeled sample. Set `LABELS_CSV`, `MODEL_NAME`, `TEXT_FEATURE`, `PREDICTION_SAMPLE_SIZE`, and `LOCAL_DATA_ONLY` near the top. It saves models under `models/` and a uniquely named prediction CSV under `notebooks/`. |
| `notebooks/results.ipynb` | Compares all five models on the same repository-disjoint holdout, saves tables and plots in the notebook, and writes figures under `notebooks/figures/`. Set `TEXT_FEATURE` or `LOCAL_DATA_ONLY` near the top. Its default fake labels measure agreement with test data, not verified authorship accuracy. |

For labeling, ambiguous evidence should be recorded with blank
`label` and `confidence` fields. Nonblank labels must be `0` or `1`, and their
confidence must be from 0 to 100. Do not copy rows from `fake_labels.csv` into
`real_labels.csv`; the fake labels and their confidence values are test data.
The `data_init.ipynb` workflow selects its review repository from the downloaded
sample and reports an error if no eligible downloaded spec is available; increase
`SPEC_SAMPLE_SIZE` if the sample contains no repository that is still available
to review.

## Gemini-Assisted Spec Labeling

The optional labeler analyzes the Markdown files already cached in
`notebooks/data`; it does not download additional specs or metadata. Install the
labeling extra:

```bash
python -m pip install -e ".[labeling]"
```

Set `GEMINI_API_KEY` in your environment using your usual secure secret
mechanism; do not put the key in source files, shell history, or commits. In
bash or zsh, you can enter it without echoing it to the terminal:

```bash
printf "Gemini API key: "
IFS= read -rs GEMINI_API_KEY
printf '\n'
export GEMINI_API_KEY
```

The labeler sends each selected file's complete contents to Gemini, so only run
it on files you are permitted to share and that contain no sensitive data or
source code. Each request can consume API quota or incur charges.

Run the following command to analyze all cached Markdown files:

```bash
python notebooks/ai-labeler.py
```

To try only particular files first, repeat `--spec-id` with their 16-character
SpecMine IDs:

```bash
python notebooks/ai-labeler.py --spec-id 0123456789abcdef \
  --spec-id fedcba9876543210
```

Results are appended to `notebooks/data/ai_analysis_results.csv` with columns
`repo_id,spec_id,label,confidence,reasoning`. Existing valid spec results are
reused on subsequent runs, including when multiple specs belong to the same
repository. Errors are reported and produce a nonzero exit status; unsuccessful
specs can be retried by running the command again. Formula-like reasoning is
escaped for spreadsheet safety. The default model is `gemini-3.5-flash-lite`;
use `--model` to select another available model and `--delay-seconds` to adjust
the four-second delay between requests. The `--data-dir` and `--output` options
override the default cache and CSV paths.

These are model-generated estimates, not verified evidence of authorship.
Confidence is the model's stated confidence in its binary label, not a
calibrated probability. Review the results before using them as training data.

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
│   ├── data/                    # Parquet/curated metadata cache, Markdown, manifest
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

The parquet cache, curated metadata subset, downloaded Markdown, sample
manifest, and generated plots are local workflow data. Cache-first loaders
reuse existing files and download only missing required files. Downloaded
Markdown filenames use `repo_name-repo_id-spec_id.md` (with `/` in the
repository name replaced by `_`) so the repository identity is retained in the
filename. Remove a cached file when you intentionally want to fetch it again;
other notebooks may create prediction CSVs and model files when run.

## Training Baselines

The training module provides five baselines:

| Model name | Inputs and estimator |
| --- | --- |
| `logistic_regression` | All numeric spec-content features, with Logistic Regression. |
| `random_forest` | All numeric spec-content features, with Random Forest. |
| `naive_bayes` | Full Markdown represented by character TF-IDF, with Multinomial Naive Bayes; no numeric features. |
| `uniform_guess` | Independent, seeded 50/50 class guesses; it ignores feature values. |
| `markdown_plus_feature` | Full Markdown represented by character TF-IDF plus exactly one selected numeric feature, with Logistic Regression. |

Both text models use case-sensitive character 3- to 5-gram TF-IDF with at most
50,000 vocabulary terms. The vectorizer and numeric preprocessing are fitted
only on the training partition. `markdown_plus_feature` defaults to `n_words`;
choose another numeric column with `--text-feature`, the `text_feature`
argument to `train_models`, or `TEXT_FEATURE` in the notebooks. The existing
numeric baselines remain the CLI and Python API defaults. Commit history is
not used by the current training/evaluation workflow.

Training requires `--labels-csv`. Use `1` when AI was used for any part of
creating a spec, `0` only when no AI was used, and a blank label for ambiguous
cases to exclude. Repositories are kept separate between training and holdout
data.

```bash
python -m specmine.train --labels-csv labels.csv --seed 42 \
  --local-data-only \
  --model-dir models/baselines
```

For a comparison including the new models:

```bash
python -m specmine.train --labels-csv labels.csv --seed 42 \
  --models logistic_regression random_forest naive_bayes uniform_guess markdown_plus_feature \
  --text-feature n_words --model-dir models/baselines
```

Selecting a text model loads Markdown for the matched labeled specs, reusing
the cache and downloading missing files from approved GitHub URLs. Unavailable
or unreadable Markdown is reported by spec ID and excluded from **all**
selected models before their shared repository-disjoint split. Set
`--local-data-only` to prevent network access; the same exclusions apply to
uncached files, and trainable models require a split with both classes.
Numeric-only and uniform-only selections do not download Markdown. When
calling the Python API directly, request
`prepare_training_data(..., include_markdown=True)` before training a text
model.

The training CSV requires `spec_id,label,confidence`: `spec_id` matches
`file_url_sha16`, `label` is `0`, `1`, or blank, and `confidence` is a numeric
score from 0 through 100. The manual-label CSV may additionally include
`repo_id` for repository-level exclusion checks; training continues to join
labels to features by `spec_id`. Every nonblank label must have a confidence
score.
Blank labels are excluded from training. The training code passes
`confidence / 100` as a sample weight to the trainable estimators, so a score
of 0 contributes no weight and 100 contributes full weight. The uniform
guesser ignores those weights. Holdout metrics remain unweighted. Omit
`--sample-size` to use all matched labeled examples; if supplied, it caps the
matched rows after ID matching. Training reads the full metadata parquet to
ensure labeled IDs are not lost to random sampling, then loads features only
for labeled specs. Omit `--model-dir` to skip saving models.

The results notebook reports predicted-class confidence as the larger of the
two class probabilities, expressed from 0 to 100, alongside the probability
that a spec is AI-assisted. The uniform guesser reports a probability of 0.5
for each class and makes seeded random class guesses; its class predictions do
not use the probability threshold.

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
supervised metrics. For a text model without labels, provide `--sample-size`
to bound the number of requested Markdown files. Missing Markdown is reported
and excluded, and the JSON report includes `n_unavailable_markdown`.
`--local-data-only` prevents Markdown downloads as well as parquet downloads.
`--split` accepts `train`, `test`, or `all`; use
`--make-val --split val` to deterministically carve a validation split out of
train. CSV metadata is escaped when it begins with a spreadsheet formula
prefix. Curated downloads use `--data-dir` (default `notebooks/data`).

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
