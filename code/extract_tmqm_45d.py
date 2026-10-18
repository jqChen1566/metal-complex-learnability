"""E03a: extraction and partitioning of tmQM 4d/5d single-metal subsets.

Parsing follows the tmQM control construction (metal type parameterized).
The 3d output is regression-checked against the reference tmqm_properties.json
(row count + CSD set + sampled fields).
Outputs:
  data/tmqm_properties_4d.json, data/tmqm_properties_5d.json
  data/partitions/tmqm4d_scaffold_<n>.json, tmqm5d_scaffold_<n>.json
  (same rule, seed 42)
"""
import sys
import re
import gzip
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from common import DATA, PART_DIR

RAW = DATA / 'tmqm_raw2024'
PHASE1_TMQM3D = DATA / 'tmqm_properties.json'

TM = {
    '3d': {'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30},
    '4d': {'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48},
    '5d': {'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80},
}
LN = {'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
      'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71}
ALL_METALS = {**TM['3d'], **TM['4d'], **TM['5d'], **LN}
ELEMENT_Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
             'K': 19, 'Ca': 20, **ALL_METALS,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
             'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86}


def parse_all():
    structures = []
    for xyz_file in ['tmQM_X1.xyz.gz', 'tmQM_X2.xyz.gz', 'tmQM_X3.xyz.gz']:
        n_file = 0
        with gzip.open(RAW / xyz_file, 'rt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    natoms = int(line)
                except ValueError:
                    continue
                comment = f.readline().strip()
                elements = [f.readline().split()[0] for _ in range(natoms)]
                csd = re.search(r'CSD_code\s*=\s*(\S+)', comment)
                q_m = re.search(r'q\s*=\s*([-\d.]+)', comment)
                s_m = re.search(r'S\s*=\s*([-\d./]+)', comment)
                elem_z = [ELEMENT_Z.get(el, 0) for el in elements]
                total_q = float(q_m.group(1)) if q_m else 0.0
                unpaired = 0
                if s_m:
                    raw = s_m.group(1)
                    if '/' in raw:
                        a, b = raw.split('/')
                        spin_s = float(a) / float(b)
                    else:
                        spin_s = float(raw)
                    unpaired = int(round(2 * spin_s))
                mt = mz = me = None
                for el in elements:
                    hit = False
                    for k, dd in TM.items():
                        if el in dd:
                            mt, mz, me = k, dd[el], el
                            hit = True
                            break
                    if hit:
                        break
                    if el in LN:
                        mt, mz, me = '4f', LN[el], el
                        break
                n_metals = sum(1 for el in elements if el in ALL_METALS)
                structures.append(dict(csd=csd.group(1) if csd else '', natoms=natoms,
                                       atomic_numbers=elem_z, total_charge=total_q,
                                       unpaired_e=unpaired, metal_type=mt, metal_z=mz,
                                       metal_el=me, n_metals=n_metals))
                n_file += 1
        print(f'  {xyz_file}: {n_file}', flush=True)
    return structures


def build_subset(structures, want, by_csd):
    out = []
    for s in structures:
        if s['metal_type'] != want or s['n_metals'] != 1:
            continue
        r = by_csd.get(s['csd'])
        if r is None or pd.isna(r['HOMO_Energy']) or pd.isna(r['LUMO_Energy']):
            continue
        out.append(dict(csd=s['csd'], atomic_numbers=s['atomic_numbers'],
                        total_charge=int(s['total_charge']), unpaired_e=int(s['unpaired_e']),
                        homo=float(r['HOMO_Energy']), lumo=float(r['LUMO_Energy']),
                        gap=float(r['HL_Gap']), dipole_mag=float(r['Dipole_M']),
                        metal_q=float(r['Metal_q']), metal_z=s['metal_z'], metal_el=s['metal_el'],
                        smiles=str(r['SMILES']) if pd.notna(r['SMILES']) else ''))
    return out


def scaffold_key(d, metal_zs):
    non_m = tuple(sorted(int(z) for z in d['atomic_numbers'] if z not in metal_zs))
    return (non_m, len(non_m), d['total_charge'])


def make_partition(rows, metal_zs, out_path):
    # encode scaffolds by insertion order
    uniq = {}
    for d in rows:
        k = scaffold_key(d, metal_zs)
        if k not in uniq:
            uniq[k] = len(uniq)
    g_int = np.array([uniq[scaffold_key(d, metal_zs)] for d in rows])
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, rest_idx = next(gss1.split(np.arange(len(rows)), groups=g_int))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    va_rel, te_rel = next(gss2.split(rest_idx, groups=g_int[rest_idx]))
    va_idx, te_idx = rest_idx[va_rel], rest_idx[te_rel]
    # assert no cross-split scaffolds
    assert len(set(g_int[tr_idx]) & set(g_int[te_idx])) == 0
    assert len(set(g_int[tr_idx]) & set(g_int[va_idx])) == 0
    part = {'n_structures_loaded': len(rows), 'n_scaffolds': len(uniq),
            'n_train': len(tr_idx), 'n_val': len(va_idx), 'n_test': len(te_idx),
            'train_idx': tr_idx.tolist(), 'val_idx': va_idx.tolist(),
            'test_idx': te_idx.tolist(), 'split_seed': 42,
            'method': 'GroupShuffleSplit scaffold-level 80/10/10 '
                      '(scaffold key = non-TM composition + counts + charge)'}
    json.dump(part, open(out_path, 'w'))
    print(f'  partition -> {out_path.name}: {len(tr_idx)}/{len(va_idx)}/{len(te_idx)}', flush=True)
    return part


def main():
    t0 = time.time()
    PART_DIR.mkdir(parents=True, exist_ok=True)
    print('parsing xyz.gz ...', flush=True)
    structures = parse_all()
    print(f'total parsed {len(structures)} | {time.time()-t0:.0f}s', flush=True)

    df = pd.read_csv(RAW / 'tmQM_y.csv', sep=';')
    by_csd = {r['CSD_code']: r for _, r in df.iterrows()}

    # ---- 3d regression check ----
    s3d = build_subset(structures, '3d', by_csd)
    phase1 = json.load(open(PHASE1_TMQM3D))
    csd_a = set(s['csd'] for s in s3d)
    csd_b = set(d['csd'] for d in phase1)
    print(f'== 3d regression: new {len(s3d)} vs reference {len(phase1)} | csd intersection '
          f'{len(csd_a & csd_b)} | new-only {len(csd_a - csd_b)} | ref-only {len(csd_b - csd_a)}',
          flush=True)
    assert len(s3d) == len(phase1) == len(csd_a & csd_b), '3d regression mismatch'
    # sampled fields, exact match
    p1_by_csd = {d['csd']: d for d in phase1}
    n_chk = 0
    for s in s3d[:200]:
        p = p1_by_csd[s['csd']]
        assert s['homo'] == p['homo'] and s['lumo'] == p['lumo'] and s['gap'] == p['gap']
        assert s['atomic_numbers'] == p['atomic_numbers']
        n_chk += 1
    print(f'  sampled fields identical {n_chk}/200 PASS', flush=True)

    # ---- 4d/5d outputs ----
    for want, zs in [('4d', set(TM['4d'].values())), ('5d', set(TM['5d'].values()))]:
        rows = build_subset(structures, want, by_csd)
        out = DATA / f'tmqm_properties_{want}.json'
        json.dump(rows, open(out, 'w'))
        n_metal = {}
        for d in rows:
            n_metal[d['metal_el']] = n_metal.get(d['metal_el'], 0) + 1
        print(f'== {want}: {len(rows)} structures -> {out.name}', flush=True)
        print(f'  per-metal: {n_metal}', flush=True)
        natoms = np.array([len(d['atomic_numbers']) for d in rows])
        print(f'  natoms mean {natoms.mean():.1f} min {natoms.min()} max {natoms.max()}', flush=True)
        part = make_partition(rows, zs, PART_DIR / f'tmqm{want}_scaffold_{len(rows)}.json')

    print(f'DONE E03a | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
