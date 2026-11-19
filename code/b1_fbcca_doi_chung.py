#!/usr/bin/env python
"""FBCCA (chen2015filter) lam doi chung khong huan luyen thay kh_deu va ct_R trong ban thao B1 (tac gia quyet 2026-11-18). Bao kem.
Dung: python b1_fbcca_doi_chung.py <json ra> <thu muc ket-qua-b1moi>
FBCCA 5 dai, 5 hai, loc nhan qua, cham luot 41..het: khoa 'fbcca5_sau' trong rcs/<bo>_rcs-tai-hien_*.json (rcs_tai_hien.py dong 198).
MAC moi quyet dinh: 5 x mac['fbcca_mot_dai'] cua cung tep (B.macs('fbcca', C, ns)); bank loc ngoai tran.
KTN-R: ktnr/<bo>_ktnr-buoc1_<T>_*.json (Benchmark, BETA, eldBETA) va ktnr/dong_ktnr-xn_<T>_*.json (Dong), khoa 'ktnr', fold 1..n.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RA, CU = sys.argv[1:3]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
BO = {"benchmark": (35, "benchmark_rcs-tai-hien_seed20261107_20261114.json", "benchmark_ktnr-buoc1_%s_seed20261107_*.json"),
      "beta": (70, "beta_rcs-tai-hien_seed20261107_20261114.json", "beta_ktnr-buoc1_%s_seed20261107_*.json"),
      "eld": (100, "eldbeta_rcs-tai-hien_seed20261107_20261115.json", "eldbeta_ktnr-buoc1_%s_seed20261107_*.json"),
      "dong": (59, "dong_rcs-tai-hien_seed20261107_20261117.json", "dong_ktnr-xn_%s_seed20261107_*.json")}


def ci(a):
    a = 100 * np.asarray(a, float)
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "BCa"
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


out = dict(seed=SEED, ghi_chu="bao kem", bo={})
for ds, (n, frcs, fk) in BO.items():
    R = json.load(open(os.path.join(CU, "rcs", frcs), encoding="utf-8"))
    out["bo"][ds] = {}
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        fs_ = glob.glob(os.path.join(CU, "ktnr", fk % tt)); fs_ = [f for f in fs_ if "phan-tich" not in f]
        assert len(fs_) == 1, fs_
        P = json.load(open(fs_[0], encoding="utf-8"))["tung_nguoi"]
        assert [p["fold"] for p in P] == list(range(1, n + 1)), ds
        k = np.array([p["ktnr"] for p in P], float)
        fb = np.array([p["fbcca5_sau"] for p in R["ket_qua"][T]["tung_nguoi"]], float)
        assert len(fb) == n and np.isfinite(fb).all() and (fb >= 0).all() and (fb <= 1).all()
        mac = R["ket_qua"][T]["mac"]
        m5 = 5 * int(mac["fbcca_mot_dai"])
        e = dict(ktnr=float(100 * k.mean()), fbcca=float(100 * fb.mean()), KTNR_tru_FBCCA=ci(k - fb), mac_fbcca_5dai=m5)
        out["bo"][ds][T] = e
        c = e["KTNR_tru_FBCCA"]
        print(ds, T, "KTN-R %.1f FBCCA %.1f | KTNR-FBCCA %+.2f [%+.2f; %+.2f] | MAC FBCCA %d" % (e["ktnr"], e["fbcca"], c["tb"], c["thap"], c["cao"], m5))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
