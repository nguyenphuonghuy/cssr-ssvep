#!/usr/bin/env python
"""End-to-end time of OACCA (5 or 1 subbands, project protocol "nq") on the ZCU104 PS.
Run in ~/ssvep_zcu104_pkg/doi_thu (needs doi_thu_oacca.py, du_lieu_ben/benchmark_S1.npz). Same clock as zcu104_doithu_ktnr.py:
the clock starts when the raw trial (9 x t) is in memory and stops after the decision AND the adaptation update of that trial
(the update must finish before the next trial). Pre-stimulus mean removal, causal 50 Hz comb notch + Chebyshev I subbands
(filter coefficients designed once at start-up), OACCA step of doi_thu_oacca.OACCA. Trials in file order 1..240, state carried over.
Times of trials 21..240 are reported (first 20 discarded). The 240 decisions are compared with the HPC run (--ref, decisions "dec").
lam_oacca(T, seed, nb) is also used by zcu104_nangluong.py (energy session).
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
from scipy.signal import lfilter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FS, ON, LAT = 250, 125, 0.14


def lam_oacca(T, seed, nb):
    import doi_thu_oacca as O
    O.NB = nb; O.FB = np.array([k ** -1.25 + 0.25 for k in range(1, nb + 1)])
    z = np.load(os.path.join(HERE, "du_lieu_ben", "benchmark_S1.npz"))
    freqs, phases = np.asarray(z["freqs"], float), np.asarray(z["phases"], float)
    ns = int(round(T * FS)); st = ON + int(round(LAT * FS))
    (bn, an), sos = O.bo_loc(FS)
    Y, Qy = O.tham_chieu(freqs, phases, ns, FS)
    m = O.OACCA(z["X"].shape[1], Y, Qy)

    def quyet(xr):
        xb = xr[:, : st + ns] - xr[:, :ON].mean(axis=1, keepdims=True)
        xb = lfilter(bn, an, xb, axis=1)
        sig = np.empty((nb, xr.shape[0], ns))
        for k in range(nb):
            sig[k] = sosfilt(sos[k], xb, axis=1)[:, st: st + ns]
        return m.buoc(sig)[2]
    return quyet, z["X"], z["y"]


def q(v):
    v = np.asarray(v, float)
    return dict(tb_ms=1e3 * float(v.mean()), trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)),
                max_ms=1e3 * float(v.max()), n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, required=True); ap.add_argument("--luong", type=int, required=True)
    ap.add_argument("--so-dai", type=int, default=5, choices=[1, 5]); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--ref", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("output exists, not overwriting")
    np.random.seed(a.seed)
    quyet, X, y = lam_oacca(a.T, a.seed, a.so_dai)
    tt, d = [], []
    for i in range(len(X)):
        t0 = time.perf_counter(); d.append(int(quyet(X[i]))); tt.append(time.perf_counter() - t0)
    d = np.asarray(d)
    r = dict(pp="oacca%d" % a.so_dai, T=a.T, luong=a.luong, seed=a.seed, tong=q(tt[20:]), tung_luot_ms=[round(1e3 * v, 4) for v in tt],
             do_chinh_xac_sau=float((d[40:] == y[40:]).mean()))
    if a.ref:
        ref = json.load(open(a.ref))["tung_nguoi"][0]; assert ref["nguoi"] == 1
        dr = np.asarray(ref["T"]["%.1f" % a.T]["oacca"]["dec"])
        r["so_quyet_dinh_hpc"] = dict(n=int(len(dr)), so_khac=int((dr != d).sum()), chi_so_khac=np.where(dr != d)[0][:20].tolist())
    import scipy
    r.update(numpy=np.__version__, scipy=scipy.__version__, loadavg_sau=open("/proc/loadavg").read().strip(),
             bien_moi_truong={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
             script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
             doi_thu_oacca_sha256=hashlib.sha256(open(os.path.join(HERE, "doi_thu_oacca.py"), "rb").read()).hexdigest())
    open(a.out, "w").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
