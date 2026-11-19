#!/usr/bin/env python
"""CI KTN-R - FBCCA (FBCCA dung cong thuc, fbcca_dung.py) cho ban thao B1. Bao kem. Thay b1_fbcca_doi_chung.py (dung FBCCA sai cong thuc).
Dung: python b1_fbcca_dung_ci.py <json ra> <thu muc ket-qua-b1moi> <fbcca json> [<fbcca nak json>]
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RA, CU, FB = sys.argv[1:4]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
F = json.load(open(FB, encoding="utf-8"))["bo"]
if len(sys.argv) > 4:
    F.update(json.load(open(sys.argv[4], encoding="utf-8"))["bo"])
MAU = {"benchmark": "benchmark_ktnr-buoc1_%s_seed20261107_*.json", "beta": "beta_ktnr-buoc1_%s_seed20261107_*.json",
       "eld": "eldbeta_ktnr-buoc1_%s_seed20261107_*.json", "dong": "dong_ktnr-xn_%s_seed20261107_*.json", "nak": "nak_ktnr-buoc2_%s_seed20261107_*.json"}


def ci(a):
    a = 100 * np.asarray(a, float)
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "BCa"
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


out = dict(seed=SEED, fbcca=FB, ghi_chu="bao kem", bo={})
for ds in F:
    out["bo"][ds] = {}
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        fs_ = [f for f in glob.glob(os.path.join(CU, "ktnr", MAU[ds] % tt)) if "phan-tich" not in f]
        assert len(fs_) == 1, (ds, fs_)
        P = json.load(open(fs_[0], encoding="utf-8"))["tung_nguoi"]
        n = len(P); assert [p["fold"] for p in P] == list(range(1, n + 1))
        Q = F[ds][T]["tung_nguoi"]; assert [q["nguoi"] for q in Q] == list(range(1, n + 1)), ds
        k = np.array([p["ktnr"] for p in P]); fb = np.array([q["sau"] for q in Q])
        c = ci(k - fb)
        out["bo"][ds][T] = dict(ktnr=float(100 * k.mean()), fbcca=float(100 * fb.mean()), KTNR_tru_FBCCA=c, mac_fbcca=F[ds][T]["mac_5dai"])
        print(ds, T, "KTN-R %.1f FBCCA %.1f | %+.2f [%+.2f; %+.2f] | MAC %d" % (100 * k.mean(), 100 * fb.mean(), c["tb"], c["thap"], c["cao"], F[ds][T]["mac_5dai"]))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
