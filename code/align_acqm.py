"""E03b: AcQM alignment and main data partition.

Alignment key: h5 `uid` == `name` in homo_lumo.json, exact match, same order;
the json is an ordered subset of the h5 file (2394/2531 parsed successfully).
Units: h5 energies in Hartree (evaluated x27.2114); h5 dipole in a.u. x2.5417
-> D. Outputs: data/acqm_properties_occ50.json (LnQM-compatible fields) plus
partitions/acqm_scaffold_*.json. LoEO check: held-out element counts are
compared item by item against split_info.

Important: the h5 frontier orbitals were parsed with an occupancy rule that is
invalid for ORCA SmearTemp calculations (fractional occupations). The corrected
labels in data_prep/acqm_occ50_labels.npz define HOMO as the highest orbital
with occupation >= 0.5 and LUMO as the lowest with occupation < 0.5. They are
applied here; 147 of the 2,394 structures change. Do not use the raw h5
frontier orbitals for the 5f series.
"""
import sys
import json
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import h5py
from sklearn.model_selection import GroupShuffleSplit
from common import DATA, PART_DIR, scaffold_key, AU_TO_DEBYE, EV_PER_HA, METAL_RANGES

H5 = DATA / 'acqm.h5'
JSON_HL = DATA / 'acqm' / 'acqm_homo_lumo.json'
SPLIT_INFO = DATA / 'acqm' / 'split_info.json'
OUT = DATA / 'acqm_properties_occ50.json'
LABELS = Path(__file__).resolve().parent.parent / 'data_prep' / 'acqm_occ50_labels.npz'

AN_SYMBOL = {89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am',
             96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr'}


def main():
    t0 = time.time()
    f = h5py.File(H5, 'r')
    d, s = f['data'], f['slices']
    n = len(d['uid'])
    uids = [x.decode() if isinstance(x, bytes) else str(x) for x in d['uid'][()]]
    hl = json.load(open(JSON_HL))
    keep = set(r['name'] for r in hl)
    idx_keep = [i for i, u in enumerate(uids) if u in keep]
    print(f'h5 {n} | json {len(hl)} | matched {len(idx_keep)} | {time.time()-t0:.0f}s', flush=True)
    assert len(idx_keep) == len(hl), 'alignment count mismatch'

    b_coord = s['coord'][()]
    rows = []
    for j, i in enumerate(idx_keep):
        a0, a1 = int(b_coord[i]), int(b_coord[i + 1])
        zs = d['numbers'][a0:a1].ravel().astype(int).tolist()
        hu = float(d['homo_spin_up'][i].item()); hd = float(d['homo_spin_down'][i].item())
        lu = float(d['lumo_spin_up'][i].item()); ld = float(d['lumo_spin_down'][i].item())
        homo = max(hu, hd); lumo = min(lu, ld)
        ch = int(d['charge'][i].item())
        if ch > 2 ** 63:
            ch -= 2 ** 64
        rows.append(dict(index=j, uid=uids[i], atomic_numbers=zs,
                         total_charge=ch, unpaired_e=int(d['unpaired_e'][i].item()),
                         homo_up=hu, homo_dn=hd, lumo_up=lu, lumo_dn=ld,
                         homo=homo, lumo=lumo, gap=lumo - homo,
                         dipole_mag=float(d['dipole'][i].item()) * AU_TO_DEBYE))

    labels = np.load(LABELS)
    assert np.array_equal(np.asarray(labels['index'], dtype=int),
                          np.arange(len(rows))), 'occ50 label index mismatch'
    changed = 0
    for j, r in enumerate(rows):
        hu = float(labels['homo_up'][j])
        hd = float(labels['homo_dn'][j])
        lu = float(labels['lumo_up'][j])
        ld = float(labels['lumo_dn'][j])
        if (abs(hu - r['homo_up']) > 1e-12 or abs(hd - r['homo_dn']) > 1e-12
                or abs(lu - r['lumo_up']) > 1e-12
                or abs(ld - r['lumo_dn']) > 1e-12):
            changed += 1
        r['homo_up'], r['homo_dn'] = hu, hd
        r['lumo_up'], r['lumo_dn'] = lu, ld
        r['homo'] = max(hu, hd)
        r['lumo'] = min(lu, ld)
        r['gap'] = r['lumo'] - r['homo']
    print(f'occ50 labels applied: {changed} structures changed', flush=True)
    assert changed == 147, 'unexpected number of corrected 5f structures'
    json.dump(rows, open(OUT, 'w'))
    print(f'written {len(rows)} -> {OUT.name} | {time.time()-t0:.0f}s', flush=True)

    # ---- check 1: alignment and convention (the HOMO in the extracted json
    #      corresponds to the h5 homo_dn channel; training values use the
    #      max(up,dn) convention, i.e. the true HOMO across spins. Here we
    #      compare json vs h5_dn.) ----
    hl_by = {r['name']: r for r in hl}
    devs = np.array([abs(r['homo_dn'] * EV_PER_HA - hl_by[r['uid']]['HOMO']) for r in rows])
    n_ok = int((devs < 0.01).sum())
    exc_idx = np.where(devs >= 0.01)[0]
    print(f'check 1: json HOMO ~= h5.homo_dn verified {n_ok}/{len(devs)} <0.01 eV '
          f'(max {devs.max():.2f} eV) | exceptions {len(exc_idx)}', flush=True)
    if len(exc_idx):
        exc_unp = Counter(rows[i]['unpaired_e'] for i in exc_idx)
        print(f'  exception set unpaired distribution: {dict(sorted(exc_unp.items()))}', flush=True)
    assert n_ok / len(devs) > 0.90, 'alignment/convention check failed'

    # ---- check 2: dipole conversion sanity ----
    dips = np.array([r['dipole_mag'] for r in rows])
    print(f'check 2: dipole(D) mean {dips.mean():.2f} | min {dips.min():.2f} | max {dips.max():.2f}',
          flush=True)
    assert 1 <= dips.mean() <= 30

    # ---- main partition (same rule, seed 42; metal = actinides 89-103) ----
    metal = METAL_RANGES['an']
    uniq = {}
    for r in rows:
        k = scaffold_key(r, metal)
        if k not in uniq:
            uniq[k] = len(uniq)
    g = np.array([uniq[scaffold_key(r, metal)] for r in rows])
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, rest = next(gss1.split(np.arange(len(rows)), groups=g))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    va_r, te_r = next(gss2.split(rest, groups=g[rest]))
    va, te = rest[va_r], rest[te_r]
    assert len(set(g[tr]) & set(g[te])) == 0 and len(set(g[tr]) & set(g[va])) == 0
    part = dict(n_structures_loaded=len(rows), n_scaffolds=len(uniq), n_train=len(tr),
                n_val=len(va), n_test=len(te), train_idx=tr.tolist(), val_idx=va.tolist(),
                test_idx=te.tolist(), split_seed=42,
                method='GroupShuffleSplit scaffold-level 80/10/10 '
                       '(scaffold key = non-An composition + counts + charge)')
    pp = PART_DIR / f'acqm_scaffold_{len(rows)}.json'
    json.dump(part, open(pp, 'w'))
    print(f'partition -> {pp.name}: {len(tr)}/{len(va)}/{len(te)} | {time.time()-t0:.0f}s', flush=True)

    # ---- check 3: LoEO sizes vs split_info ----
    cnt = Counter()
    for r in rows:
        el = next((z for z in r['atomic_numbers'] if 89 <= z <= 103), None)
        cnt[AN_SYMBOL.get(el, str(el))] += 1
    si = json.load(open(SPLIT_INFO))
    print('check 3: leave-one-element counts vs split_info test counts', flush=True)
    ok = True
    for el_sym in sorted(cnt):
        fold = si.get(f'loeo_holdout_{el_sym}')
        ref = fold['test'] if fold else -1
        flag = '' if cnt[el_sym] == ref else '  <-- MISMATCH'
        if cnt[el_sym] != ref:
            ok = False
        print(f'  {el_sym:3s} ours {cnt[el_sym]:4d} vs split_info {ref:4d}{flag}', flush=True)
    assert ok, 'LoEO sizes disagree with split_info; check alignment'
    print(f'DONE E03b | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
