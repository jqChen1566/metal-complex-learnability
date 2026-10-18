"""E01-ext: multi-property extraction from lnqm.h5.

Alignment scheme:
  - per-atom/slice fields: structure i spans slices[key][i] .. slices[key][i+1];
  - per-structure fields (exc etc.): index e_idx = int(slices['energy'][i]).
Writes data/lnqm_properties_ext.json (index = h5 structure order), after
alignment checks and range sanity checks.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import h5py
from common import H5_LNQM, LNQM_EXT, LNQM_TRAIN


def main():
    f = h5py.File(H5_LNQM, 'r')
    d, s = f['data'], f['slices']
    n = len(d['uid'])
    print('n structures =', n, flush=True)

    def slice_sum(key):
        b = s[key][()]
        v = d[key][()].ravel()
        out = np.zeros(n)
        for i in range(n):
            out[i] = v[int(b[i]):int(b[i + 1])].sum()
        return out

    print('summing per-atom fields...', flush=True)
    pol = slice_sum('polarizabilities')
    ceh = slice_sum('ceh')
    eeq = slice_sum('eeq')
    qg2 = slice_sum('q_gfn2')
    low = slice_sum('loewdin_charges')
    mul = slice_sum('mulliken_charges')
    mpop = slice_sum('mayer_pop')

    per = {k: d[k][()].ravel() for k in ['exc', 'ex', 'ec', 'ecnl', 'eemb']}
    bcn, vcn = s['cn'][()], d['cn'][()]
    brc = s['rot_const'][()]
    vrc = d['rot_const'][()].ravel()

    rows = []
    for i in range(n):
        e_idx = int(s['energy'][i])
        seg = vcn[int(bcn[i]):int(bcn[i + 1])]
        rseg = vrc[int(brc[i]):int(brc[i + 1])]
        rv = rseg[~np.isnan(rseg)]
        rows.append(dict(
            index=i,
            exc=float(per['exc'][e_idx]), ex=float(per['ex'][e_idx]),
            ec=float(per['ec'][e_idx]), ecnl=float(per['ecnl'][e_idx]),
            eemb=float(per['eemb'][e_idx]),
            pol_sum=float(pol[i]), ceh_sum=float(ceh[i]), eeq_sum=float(eeq[i]),
            q_gfn2_sum=float(qg2[i]), loewdin_sum=float(low[i]),
            mulliken_sum=float(mul[i]), mayer_pop_sum=float(mpop[i]),
            cn_mean=float(seg.mean()), cn_max=float(seg.max()), cn_std=float(seg.std()),
            rot_mean=float(rv.mean()) if len(rv) else float('nan')))

    json.dump(rows, open(LNQM_EXT, 'w'))
    print('written ->', LNQM_EXT, len(rows), 'rows', flush=True)

    # ---- alignment check (must pass) ----
    print('== alignment check ==', flush=True)
    tr = json.load(open(LNQM_TRAIN))
    for r in tr[:20]:
        idx = r['index']
        h5v = float(d['homo_spin_up'][int(s['energy'][idx])].item())
        assert abs(h5v - r['homo_up']) < 1e-6, (idx, h5v, r['homo_up'])
        n_h5 = int(s['coord'][idx + 1]) - int(s['coord'][idx])
        assert n_h5 == len(r['atomic_numbers']), idx
    print('index <-> homo/atoms consistency: PASS (20/20)', flush=True)

    # ---- range sanity check ----
    for k in ['exc', 'ex', 'ec', 'ecnl', 'pol_sum', 'eeq_sum', 'cn_mean', 'rot_mean']:
        v = np.array([r[k] for r in rows])
        print(f'{k:12s} min {np.nanmin(v):10.4f} max {np.nanmax(v):10.4f} mean {np.nanmean(v):10.4f}')


if __name__ == '__main__':
    main()
