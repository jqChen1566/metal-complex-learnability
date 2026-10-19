# metal-complex-learnability Code Manual

Version: 1.0  
Date: 2026-10-19  
Repository: https://github.com/jqChen1566/metal-complex-learnability  
License: MIT

## 1. Purpose

This manual describes how to install, configure, run, and extend the code accompanying the manuscript:

> Mapping the Learnability of Metal-Complex Properties: A Five-Series (3d–5f) Machine Learning Study and Audited Inverse Design

The repository is intended for readers who wish to reproduce the manuscript figures and tables, retrain the models, or build on the analysis. Raw public datasets are not redistributed. LnQM, tmQM, and AcQM must be obtained from their original sources and placed in the layout described in Section 4.

## 2. Repository layout

```
metal-complex-learnability/
  README.md
  LICENSE
  requirements.txt
  code/                 composition models, decomposition, robustness, transfer, inverse design
  gnn/                  graph construction, SchNet/DimeNet++/CC-Schnet, SOAP, frozen features
  data_prep/            polarizability, node-level, edge-level, RDKit, composition baselines
  qc_recompute/         ORCA input generation, output parsing, multi-reference anchors
  figures/              manuscript figure scripts and included derived data
  docs/                 data layout, reproduction notes, and this manual
```

## 3. Environment

### 3.1 Python

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies: numpy, scipy, scikit-learn, torch, torch-geometric, h5py, ase, dscribe, rdkit, matplotlib, joblib, and pandas.

### 3.2 External programs

- ORCA 6.1.1 for the independent recomputation and multi-reference anchors in `qc_recompute/`.
- SpookyNet: optional, used by `gnn/spookynet_features.py`; obtain the source tree and pretrained weights separately.
- MACE-OMol: optional, used by `gnn/mace_omol_features.py`; install the upstream package and weights separately.
- GPU: optional. The composition MLP and the graph models run on CPU; a GPU reduces training time.

## 4. Data layout

### 4.1 Environment variables

The code locates data and outputs through environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `METAL_ML_DATA` | `data` | Property tables, partitions, and datasets |
| `METAL_ML_RESULTS` | `results` | Analysis outputs |
| `METAL_ML_PARTITIONS` | `$METAL_ML_DATA/partitions` | Partition files |
| `LNQM_H5` | `$METAL_ML_DATA/lnqm.h5` | Raw LnQM HDF5 file |
| `GRAPH_DIR` | `$METAL_ML_DATA/graphs` | Graph files and graph-model outputs |
| `SPOOKYNET_DIR` | unset | SpookyNet source directory |
| `METAL_ML_RECOMP` | `$METAL_ML_RESULTS/recompute` | Quantum-chemical recomputation workspace |
| `MULTIREF_DIR` | `$METAL_ML_RECOMP/multi_reference` | Multi-reference anchor outputs |

Example:

```bash
export METAL_ML_DATA=/path/to/data
export METAL_ML_RESULTS=/path/to/results
export LNQM_H5=/path/to/data/lnqm.h5
export GRAPH_DIR=/path/to/data/graphs
```

### 4.2 Property tables

```
data/
  lnqm.h5
  lnqm_properties_train.json
  lnqm_properties_val.json
  lnqm_properties_test.json
  lnqm_properties_ext.json
  tmqm_properties.json
  tmqm_properties_4d.json
  tmqm_properties_5d.json
  tmqm_properties_5f.json
  tmqm_y.csv
  partitions/
    lnqm_scaffold_12444.json
    tmqm_scaffold_33086.json
```

Each property JSON record contains atomic numbers, coordinates, total charge, unpaired electrons, target values, and the original dataset index. The train, validation, and test files define the fixed split; the JSON files in `partitions/` are the authoritative row indices.

### 4.3 Graph files

```
data/graphs/
  graphs_lnqm.pt
  graphs_3d.pt
  graphs_4d.pt
  graphs_5d.pt
  graphs_5f.pt
```

Each graph file is a tensor dictionary with `z`, `pos`, `y_homo`, `train_rows`, `val_rows`, and `test_rows`. The data-preparation scripts may add `y_pol`, `y_hirsh_q`, `y_hirsh_s`, `bo_pairs`, and `bo_vals`. Coordinates are in angstrom; energies are in electronvolts.

## 5. Quick start

```bash
# Property extraction and alignment
python code/extract_properties.py
python code/extract_tmqm_45d.py
python code/align_acqm.py

# Composition models and main analyses
python code/train_composition_models.py --models both
python code/decomposition_analysis.py
python code/robustness.py
python code/uncertainty.py
python code/diagnostics.py
python code/model_families.py

# Five-series analysis, transfer, and inverse design
python code/five_series.py
python code/transfer.py
python code/design_benchmark.py
python code/design_rebacktest.py

# Figures (the included data_raw directories allow direct execution)
python figures/fig1/plot_fig1.py
python figures/fig2_10/plot_fig2.py
```

## 6. Step-by-step workflow

### 6.1 Property extraction and alignment

`code/extract_properties.py` extracts the fifteen targets from the LnQM HDF5 file, writes `lnqm_properties_ext.json`, and checks index and atom-count alignment.

`code/extract_tmqm_45d.py` extracts the tmQM 4d and 5d single-metal subsets and writes their property tables and scaffold partitions.

`code/align_acqm.py` aligns the AcQM HDF5 file with the property tables and writes the 5f table and partition.

### 6.2 Composition models

`code/train_composition_models.py` trains a random forest and a multilayer perceptron on the 31-dimensional composition descriptor. Hyperparameters are selected by scaffold-grouped cross-validation within the training partition; the test partition is used only for final reporting.

`code/decomposition_analysis.py` produces Gini importance, permutation importance, charge ablation, and leave-one-element results.

`code/robustness.py` runs random-forest seed stability and the MLP hyperparameter grid.

`code/uncertainty.py` evaluates the random-forest tree spread and the MLP ensemble spread and reports coverage and Spearman correlation.

`code/diagnostics.py` stratifies errors by charge, metal, and atom count, and performs paired leave-one-element versus cross-validation tests.

`code/model_families.py` runs histogram gradient boosting and SOAP kernel ridge regression as model-family controls.

### 6.3 Five-series analysis and transfer

`code/five_series.py` trains composition models on the five series and writes the series baselines and leave-one-metal results.

`code/gini_ci.py`, `code/ablation_series.py`, and `code/partition_sensitivity.py` provide the charge-importance confidence intervals, the ablation spectrum, and the partition-seed sensitivity.

`code/transfer.py` runs the 5×5 zero-shot transfer matrix, the fine-tuning curves, and the source-size controls.

### 6.4 Graph models

```bash
python gnn/build_graphs.py
python gnn/build_series.py
python gnn/train_gnn.py --model schnet --epochs 300
python gnn/train_gnn.py --model dimenet --epochs 300
python gnn/train_gnn.py --model ccschnet --epochs 300
python gnn/soap_krr.py --cache full
python gnn/spookynet_features.py --cache full
python gnn/mace_omol_features.py --cache full
python gnn/decomposition.py
```

`build_graphs.py` and `build_series.py` create the PyG graph datasets. `train_gnn.py` supports SchNet, DimeNet++, CC-Schnet, and node-level and edge-level tasks. `soap_krr.py` trains SOAP descriptors with kernel ridge regression. `spookynet_features.py` and `mace_omol_features.py` use frozen features with a ridge readout. `decomposition.py` analyzes the source of the geometric gain.

### 6.5 Inverse design

```bash
python code/inverse_design.py
python code/design_benchmark.py
python code/design_rebacktest.py
```

`inverse_design.py` trains the differentiable surrogate and implements projected gradient, random sampling, a genetic algorithm, and greedy search. `design_benchmark.py` runs the 50-run budget-matched benchmark. `design_rebacktest.py` re-evaluates candidates with an independent random forest and reports the paired improvement over the starting structure.

### 6.6 Quantum-chemical recomputation

```bash
python qc_recompute/sample_lnqm.py
python qc_recompute/gen_orca_inputs.py
# run the generated inputs in ORCA
python qc_recompute/analyze_orca.py
python qc_recompute/parse_multiref.py
```

`sample_lnqm.py` selects 100 LnQM complexes. `gen_orca_inputs.py` generates single-point inputs for the default and tightened convergence protocols. `analyze_orca.py` parses the outputs and compares them with the stored labels. `parse_multiref.py` parses the AVAS-CASCI and NEVPT2 anchor results.

### 6.7 Figures

```bash
python figures/fig1/plot_fig1.py
python figures/fig2_10/plot_fig2.py
python figures/fig2_10/plot_fig3.py
python figures/fig2_10/plot_fig4.py
python figures/fig2_10/plot_fig5.py
python figures/fig2_10/plot_fig6.py
python figures/fig2_10/plot_fig7.py
python figures/fig2_10/plot_fig8.py
python figures/fig2_10/plot_fig9.py
python figures/fig2_10/plot_fig10.py
python figures/fig2_10/plot_toc.py
python figures/main/plot_main_figure.py
```

Each figure script reads the derived data in its local `data_raw/` directory and writes outputs next to the script. `figures/fig2_10/make_tiff.py` converts PNG files to 600 dpi TIFF with LZW compression and an embedded sRGB profile.

## 7. Script reference

### 7.1 code/

| Script | Purpose |
|---|---|
| `common.py` | Paths, units, and the 31-dimensional composition features |
| `extract_properties.py` | LnQM fifteen-target extraction |
| `train_composition_models.py` | Random forest and MLP main pipeline |
| `decomposition_analysis.py` | Gini, permutation, ablation, leave-one-element |
| `robustness.py` | Seed stability and MLP grid |
| `uncertainty.py` | Uncertainty calibration |
| `diagnostics.py` | Error stratification and paired tests |
| `model_families.py` | Gradient boosting and kernel ridge regression |
| `five_series.py` | Five-series baselines and leave-one-metal |
| `extract_tmqm_45d.py` | tmQM 4d/5d extraction |
| `align_acqm.py` | AcQM alignment and partition |
| `transfer.py` | Zero-shot transfer and fine-tuning |
| `gini_ci.py` | Charge-importance confidence intervals |
| `ablation_series.py` | Five-series charge ablation |
| `partition_sensitivity.py` | Partition-seed sensitivity |
| `inverse_design.py` | Differentiable inverse-design framework |
| `design_benchmark.py` | Design benchmark |
| `design_rebacktest.py` | Independent re-evaluation |
| `overview_figures.py` | Overview figure generation |
| `benchmark_mlp_device.py` | CPU/GPU training benchmark |

### 7.2 gnn/

| Script | Purpose |
|---|---|
| `build_graphs.py` | Build LnQM graph data |
| `build_series.py` | Build 3d/4d/5d/5f graph data |
| `train_gnn.py` | SchNet, DimeNet++, and CC-Schnet training |
| `soap_krr.py` | SOAP descriptors with kernel ridge regression |
| `spookynet_features.py` | SpookyNet frozen features with ridge |
| `mace_omol_features.py` | MACE-OMol frozen features with ridge |
| `decomposition.py` | Geometric-gain decomposition |

### 7.3 data_prep/

| Script | Purpose |
|---|---|
| `add_polarizability.py` | Add polarizability labels to LnQM graphs |
| `add_tmqm_polarizability.py` | Add polarizability labels to tmQM graphs |
| `rf_baselines.py` | 120-dimensional composition random-forest baseline |
| `atom_additive_baseline.py` | Atom-additive polarizability baseline |
| `add_node_labels.py` | Add Hirshfeld charge and spin node labels |
| `rdkit_baseline.py` | Morgan fingerprint and physicochemical descriptor baseline |
| `add_edge_labels.py` | Add Mayer bond-order edge labels |

### 7.4 qc_recompute/

| Script | Purpose |
|---|---|
| `sample_lnqm.py` | Select 100 complexes for recomputation |
| `gen_orca_inputs.py` | Generate two ORCA single-point input sets |
| `analyze_orca.py` | Parse ORCA output and compare with stored labels |
| `parse_multiref.py` | Parse multi-reference anchor results |

### 7.5 figures/

| Script | Purpose |
|---|---|
| `fig1/plot_fig1.py` | Manuscript Fig. 1 |
| `main/plot_main_figure.py` | Model × series × property heatmap set |
| `fig2_10/plot_fig2.py` | Fig. 2 |
| `fig2_10/plot_fig3.py` | Fig. 3 |
| `fig2_10/plot_fig4.py` | Fig. 4 |
| `fig2_10/plot_fig5.py` | Fig. 5 |
| `fig2_10/plot_fig6.py` | Fig. 6 |
| `fig2_10/plot_fig7.py` | Fig. 7 |
| `fig2_10/plot_fig8.py` | Fig. 8 |
| `fig2_10/plot_fig9.py` | Fig. 9 |
| `fig2_10/plot_fig10.py` | Fig. 10 |
| `fig2_10/plot_toc.py` | Table-of-contents graphic |
| `fig2_10/make_tiff.py` | PNG to TIFF conversion |

## 8. Output directories

Directories under `METAL_ML_RESULTS` retain the analysis-stage labels used in the manuscript:

| Directory | Contents |
|---|---|
| `E02/` | Composition models, seeds, hyperparameters, calibration, diagnostics |
| `E02b/` | Determinant matrix, ablation, leave-one-element |
| `E03/` | Five-series baselines, transfer, partition sensitivity |
| `E04/` | Design benchmark and re-evaluation |

Figure scripts write PNG, PDF, and SVG files; `make_tiff.py` produces TIFF files.

## 9. Reproduction conventions

- Split: scaffold-grouped partition; all models within a series share the same split.
- Standardization: means and standard deviations are computed on the training partition only.
- Random seeds: default 42; seed-stability and partition-sensitivity scripts state the additional seeds.
- Units: energy targets are reported in electronvolts with 27.2114 eV per hartree; dipole moments in debye, polarizabilities in atomic units, and charges in elementary charge.
- Test partition: used only for final reporting, never for model selection.
- Graph models: training budget, hidden width, layer count, and early-stopping rules are exposed as command-line arguments; the manuscript configurations are listed in the Supporting Information.

## 10. Troubleshooting

### 10.1 Property files not found

Check `METAL_ML_DATA`, `LNQM_H5`, and `METAL_ML_PARTITIONS`. Run scripts from the repository root or export absolute paths.

### 10.2 CUDA is unavailable

The code falls back to CPU automatically. A GPU affects speed only, not the algorithm. To force CPU, set `CUDA_VISIBLE_DEVICES` to an empty value.

### 10.3 torch-geometric import fails

Confirm that `torch-geometric` matches the installed PyTorch version. Graph construction in this repository uses pure torch and does not require `torch_cluster` or `pyg-lib`.

### 10.4 SpookyNet or MACE-OMol import fails

These baselines require their upstream packages and pretrained weights. Set `SPOOKYNET_DIR` for SpookyNet; configure the MACE-OMol weights as described by the upstream package. Skip the corresponding scripts if these baselines are not needed.

### 10.5 ORCA inputs do not run

`qc_recompute/gen_orca_inputs.py` writes self-contained inputs with coordinates and `%pal nprocs 1`. Run them with ORCA 6.1.1; parallel settings and convergence keywords may differ in other versions.

### 10.6 Figure font or width differs

The figure scripts use Arial and a matching mathtext configuration. If Arial is unavailable, matplotlib falls back to another font and the figure width may change. Regenerate and inspect the figures in the target environment before submission.

## 11. Citation and license

If you use this code, please cite the manuscript:

> Mapping the Learnability of Metal-Complex Properties: A Five-Series (3d–5f) Machine Learning Study and Audited Inverse Design

Journal information and the DOI will be added after publication. The code is released under the MIT License; see `LICENSE` in the repository.

## 12. Revision history

| Date | Version | Change |
|---|---|---|
| 2026-10-19 | 1.0 | First release: repository layout, data layout, workflow, script reference, outputs, and troubleshooting |
