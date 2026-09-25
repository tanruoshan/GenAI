# Semantic Repair of Damage-Occluded Historical Newspaper Text

GenAI coursework project (6 ECTS).

## Research question

Given pretrained generative and masked-language-model GenAI methods, to what extent can damaged
word spans in historical newspaper text be reconstructed so that the semantic content and key
facts of the original sentence are preserved, and how does reconstruction quality change as the
size of the damaged region grows?

## Status

Day 1: dataset loading and the synthetic damage-injection pipeline. Repair methods and evaluation
are not part of this deliverable yet.

## Notebooks

1. `notebooks/01_dataset_and_damage_pipeline.ipynb`: data pool, calibration, and the damage
   pipeline (this deliverable).
2. `notebooks/02_repair_and_experiments.ipynb`: repair methods and experiments (later).
3. `notebooks/03_analysis.ipynb`: results analysis and report figures (later).

## Data

This project uses [BLN600](https://orda.shef.ac.uk/) (Booth, Thomas & Gaizauskas, 2024), a
password-protected, CC BY-NC-ND licensed corpus. It is not included in this repository. To run
the notebooks, unzip `BLN600.zip` into `data/raw/BLN600/`; the folder is git-ignored.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

Select the `.venv` interpreter as the Jupyter kernel in VS Code to run the notebooks.
