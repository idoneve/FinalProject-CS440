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

The example notebook adds this repository's `src/` folder to its import path,
so it does not rely on an editable install to find `specmine`. It prints the
Python interpreter used by its kernel; if imports fail for missing dependencies,
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

Alternatively, open `notebooks/example.ipynb` after installing the package.
The loader needs an internet connection to access the dataset.

## Training Baselines

The training module compares Logistic Regression and Random Forest baselines
using numeric features and labels supplied by you. It does not generate proxy
labels and refuses to train unless `--labels-csv` is provided. Use `1` when AI
was used for any part of creating a spec, `0` only when no AI was used, and a
blank label for ambiguous cases to exclude. Repositories are kept separate
between training and holdout data.

```bash
python -m specmine.train --labels-csv labels.csv --seed 42 \
  --model-dir models/baselines
```

The CSV uses the same schema as evaluation: `spec_id` matches
`file_url_sha16`, and `label` is `0`, `1`, or blank. Omit `--sample-size` to use
all matched labeled examples; if supplied, it caps the matched rows after ID
matching. Training reads the full metadata parquet to ensure labeled IDs are
not lost to random sampling, then loads features only for labeled specs. Omit
`--model-dir` to skip saving models. Select one or both estimators with
`--models logistic_regression random_forest`.

For pipeline smoke-testing only, `examples/synthetic_labels.csv` contains
fabricated labels for sample specs. They are **not authorship ground truth**;
metrics from that file are not evidence of AI-authorship accuracy. The trainer
always describes results as metrics against the supplied labels. Only verified
labels support conclusions about authorship performance.

The notebook requires an explicit `SPECMINE_LABELS_CSV` environment variable;
set it to either your real labels or the synthetic smoke-test file before
running its training cell.

## Evaluating a Classifier

The released SpecMine schema does not include AI-authorship labels. Supply a
CSV of manually labeled examples with `spec_id` values matching
`file_url_sha16` and a binary `label` column (`1` if AI was used at any point in
creating the spec, `0` if no AI was used). Examples without a label match are
excluded and counted. Leave ambiguous examples blank; training excludes them.

```csv
spec_id,label
0123456789abcdef,1
fedcba9876543210,0
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
train. Curated downloads use `--data-dir` (default `./data`).

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
