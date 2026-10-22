# CS440 - Intro to AI Final Project: AI vs Human Specification Agent

## Project Overview

Our team is participating in the **MSR 2027 Mining Challenge** using the **SpecMine** dataset
Specification can be found here: **<https://2027.msrconf.org/track/msr-2027-mining-challenge>**

Software developers are increasingly using AI agents to write technical specification files (`spec.md`). This project aims to build a AI classifier that analyzes the language inside these specification files and predicts whether a specific file **was or was not drafted by an AI agent**.

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
