# Learnability of Metal-Complex Properties across 3d–5f Series

Code and derived data for the manuscript:

> Mapping the Learnability of Metal-Complex Properties: A Five-Series (3d–5f)
> Machine Learning Study and Audited Inverse Design

The repository contains the analysis pipeline, graph-neural-network benchmarks,
quantum-chemical recomputation scripts, and figure-generation code used in the
study. Raw public datasets are not redistributed; they are obtained from their
original sources and referenced in the manuscript.

## What is included

| Path | Contents |
|---|---|
| `code/` | Composition-model pipeline: data extraction, random forest and MLP training, decomposition, robustness, uncertainty, five-series transfer, and differentiable inverse design |
| `gnn/` | Graph construction, SchNet/DimeNet++/CC-Schnet training, SOAP and frozen-feature baselines, and information decomposition |
| `data_prep/` | Polarizability, node-level, edge-level, RDKit and composition-baseline preparation |
| `qc_recompute/` | ORCA input generation and parsing for label reproduction and multi-reference anchors |
| `figures/` | Figure-generation scripts and the small derived tables needed to redraw the manuscript figures |
| `docs/` | Data layout, environment variables, and step-by-step reproduction notes |

## Data

The three public datasets are:

- **LnQM** — lanthanoid complexes with quantum-chemical labels.
- **tmQM** — transition-metal complexes (3d, 4d and 5d series).
- **AcQM** — actinide complexes (5f series).

Download them from the sources cited in the manuscript and place them under a
single data root. The code reads paths from environment variables:

```bash
export METAL_ML_DATA=/path/to/data
export METAL_ML_RESULTS=/path/to/results
export LNQM_H5=/path/to/lnqm.h5
export GRAPH_DIR=/path/to/graph/data
export SPOOKYNET_DIR=/path/to/SpookyNet   # optional, for the SpookyNet baseline
```

The expected directory layout and file schemas are described in
`docs/data_layout.md`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The graph models use PyTorch Geometric. The SpookyNet and MACE-OMol baselines
require their own upstream packages and pretrained weights; see
`docs/reproduction.md`.

## Quick start

```bash
# 1. Extract and align the property tables
python code/extract_properties.py
python code/extract_tmqm_45d.py
python code/align_acqm.py

# 2. Fit the composition models and run the analyses
python code/train_composition_models.py --models both
python code/decomposition_analysis.py
python code/robustness.py
python code/uncertainty.py
python code/diagnostics.py
python code/model_families.py

# 3. Five-series transfer and design experiments
python code/five_series.py
python code/transfer.py
python code/design_benchmark.py
python code/design_rebacktest.py

# 4. Redraw the figures (derived data are included)
python figures/fig1/plot_fig1.py
python figures/fig2_10/plot_fig2.py
```

The full sequence, including graph-model and quantum-chemical steps, is in
`docs/reproduction.md`.

## Figure data

The small derived tables required to redraw the manuscript figures are included
under `figures/fig1/data_raw`, `figures/main/data_raw` and
`figures/fig2_10/data_raw`. These are processed values, not raw quantum-chemical
outputs. Figure scripts write their outputs next to the script.

## Outputs

All analysis scripts write to the directory given by `METAL_ML_RESULTS`. The
default is `results/` at the repository root. The figure scripts write to their
own directories.

## Citation

If you use this code, please cite the manuscript above. A journal reference and
DOI will be added after publication.

## License

MIT License. See `LICENSE`.

## Maintainer

Jianqi Chen (@jqChen1566), Jinan University.
