# Data layout

All paths are relative to `METAL_ML_DATA` unless noted otherwise. The code does
not bundle raw datasets. Download LnQM, tmQM and AcQM from their public sources
and arrange the files as described below.

## Property tables

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
  acqm_properties_occ50.json
  tmqm_y.csv
  partitions/
    lnqm_scaffold_12444.json
    tmqm_scaffold_33086.json
```

The property JSON files contain one record per complex with atomic numbers,
coordinates, total charge, unpaired electrons, target values and the original
dataset index. The training, validation and test files contain the scaffold
split used throughout the paper; the partition JSON files are the authoritative
row indices.

For AcQM, the raw h5 frontier orbitals were parsed with an occupancy rule that
is invalid for ORCA SmearTemp calculations (fractional occupations). The file
`data_prep/acqm_occ50_labels.npz` contains the corrected labels, with HOMO
defined as the highest orbital with occupation >= 0.5 and LUMO as the lowest
with occupation < 0.5. `code/align_acqm.py` applies this table and writes
`acqm_properties_occ50.json`; 147 of the 2,394 structures change. Do not use the
raw h5 HOMO/LUMO values for the 5f series.

## Graph files

```
data/graphs/
  graphs_lnqm.pt
  graphs_3d.pt
  graphs_4d.pt
  graphs_5d.pt
  graphs_5f.pt
```

Each graph file is a dictionary of tensors with the fields `z`, `pos`, `y_homo`,
`train_rows`, `val_rows` and `test_rows`. Optional fields added by the data
preparation scripts are `y_pol`, `y_hirsh_q`, `y_hirsh_s`, `bo_pairs` and
`bo_vals`. Coordinates are in angstrom; energies are in electronvolts.

## Results

Scripts write to `METAL_ML_RESULTS` (default `results/`). Each experiment uses
its own subdirectory, for example:

```
results/
  E02/      composition models and diagnostics
  E02b/     decomposition analysis
  E03/      five-series transfer and robustness
  E04/      inverse design
```

The E02/E03/E04 directory names are stage labels retained for traceability with
the manuscript's analysis sequence.

The figure scripts read their inputs from the `data_raw` directories included
with this repository and write outputs next to the script.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `METAL_ML_DATA` | `data` | Property tables and partitions |
| `METAL_ML_RESULTS` | `results` | Analysis outputs |
| `LNQM_H5` | `$METAL_ML_DATA/lnqm.h5` | Raw LnQM HDF5 file |
| `GRAPH_DIR` | `$METAL_ML_DATA/graphs` | Graph files and model outputs |
| `SPOOKYNET_DIR` | unset | SpookyNet source tree, only for the SpookyNet baseline |
