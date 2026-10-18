# Reproduction notes

The commands below reproduce the analysis in the order used in the study. Run
them from the repository root with the environment variables from
`docs/data_layout.md` set.

## 1. Property extraction and alignment

```bash
python code/extract_properties.py
python code/extract_tmqm_45d.py
python code/align_acqm.py
```

These scripts read the public datasets and write the property JSON files and
the scaffold partitions. The 4f, 3d, 4d, 5d and 5f partitions are generated
once and reused by every model.

## 2. Composition models and analyses

```bash
python code/train_composition_models.py --models both
python code/decomposition_analysis.py
python code/robustness.py
python code/uncertainty.py
python code/diagnostics.py
python code/model_families.py
python code/five_series.py
python code/gini_ci.py
python code/ablation_series.py
python code/partition_sensitivity.py
python code/transfer.py
```

The random forest and MLP are trained on the 31-descriptor composition vector.
Energy targets are converted from hartree to electronvolts with the constant in
`code/common.py`. Hyperparameters and seeds are documented in the manuscript
and in the Supporting Information tables.

## 3. Graph models and descriptor baselines

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

`spookynet_features.py` requires the SpookyNet source tree and pretrained weights.
`mace_omol_features.py` requires the MACE-OMol package and weights. The scripts
accept a `--subset` argument for quick self-tests.

## 4. Differentiable inverse design

```bash
python code/inverse_design.py
python code/design_benchmark.py
python code/design_rebacktest.py
```

The design loop uses the trained composition MLP as a differentiable
surrogate. `design_benchmark.py` runs the budget-matched benchmark; `design_rebacktest.py` performs the
independent re-evaluation with a held-out random forest and the paired
start-structure baseline.

## 5. Quantum-chemical recomputation and anchors

```bash
python qc_recompute/sample_lnqm.py
python qc_recompute/gen_orca_inputs.py
# run the generated ORCA inputs externally
python qc_recompute/analyze_orca.py
python qc_recompute/parse_multiref.py
```

`gen_orca_inputs.py` writes ORCA inputs for the stored-label recomputation.
`analyze_orca.py` parses the outputs and produces the paired deviation tables.
`parse_multiref.py` parses the AVAS-CASCI/NEVPT2 anchor outputs.

## 6. Figures

The derived data needed for the figures are included under `figures/`.

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
```

`figures/main/plot_main_figure.py` regenerates the model × series × property
heatmaps. `figures/fig2_10/make_tiff.py` converts the PNG outputs to the
submission TIFF format.

## Notes on data and randomness

- All models use the same scaffold partition within a series.
- Random seeds are explicit in the scripts; the default is 42.
- Standardization statistics are computed on the training partition only.
- The test partition is used once, for the final reported errors.
