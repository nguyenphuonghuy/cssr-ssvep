#!/usr/bin/env python
"""CI bo sung cho dan y B1 ban 2 (bang 2 va bang 4). Chi bao kem (mo ta).
Dung: python b1_ci_bo_sung.py <json ra> <thu muc ket-qua-b1moi>
Cung ham ci() voi ktnr_buoc1_phan_tich.py (BCa 10.000, seed 20261107, percentile du phong). Chenh ghep cap theo nguoi.
Nguon tung nguoi (cung cach nap voi ma phan tich da dung):
  KTN-R, mang tinh, mang tinh + R: ktnr/<bo>_ktnr-buoc1_<T>_seed20261107_*.json (khoa 'ktnr', 'tinh', 'tinh_r'; fold 1..n).
  kh_deu: Benchmark, BETA: <bo>_b1moi_buoc0b ['ket_qua'][T]['tung_nguoi'][i]['40']['kh_deu'] (nhu b1moi_lai_beta_phan_tich.py);
          eldBETA: eldbeta_b1moi_kh ['ket_qua'][T]['tung_nguoi'][i]['40']['kh_deu'] (nhu b1moi_lai_eld_phan_tich.py).
  RCS day du: rcs/<bo>_rcs-tai-hien ['ket_qua'][T]['tung_nguoi'][i]['day_du_lo']['sau'] (nhu b1moi_lai_beta_phan_tich.py).
  Ensemble-DNN: ed/<bien the>_<bo>_<T>/foldNNN.json ['sau'].
Kiem: so nguoi, thu tu fold, gia tri trong [0, 1]; in TB de doi chieu voi bang da co.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RA, CU = sys.argv[1:3]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
SO = {"benchmark": 35, "beta": 70, "eld": 100}
TEN = {"benchmark": "benchmark", "beta": "beta", "eld": "eldbeta"}


def ci(a):
    a = 100 * np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()), phuong_phap="hang")
    pp = "BCa"
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError("khong huu han")
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


def hop_le(v, n, ten):
    v = np.asarray(v, float)
    assert len(v) == n and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), ten
    return v


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
out = dict(seed=SEED, ghi_chu="bao kem", bo={})
for ds in ("benchmark", "beta", "eld"):
    n = SO[ds]; out["bo"][ds] = {}
    kh_f = os.path.join(CU, "%s_b1moi_buoc0b_seed20261107_20261114.json" % ds) if ds != "eld" else os.path.join(CU, "eldbeta_b1moi_kh_seed20261107_20261114.json")
    KH = json.load(open(kh_f, encoding="utf-8"))
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        fs_ = glob.glob(os.path.join(CU, "ktnr", "%s_ktnr-buoc1_%s_seed%d_*.json" % (TEN[ds], tt, SEED)))
        assert len(fs_) == 1, fs_
        P = json.load(open(fs_[0], encoding="utf-8"))["tung_nguoi"]
        assert [p["fold"] for p in P] == list(range(1, n + 1))
        x = lambda k: hop_le([p[k] for p in P], n, k)
        ktnr = x("ktnr")
        kh = hop_le([p["40"]["kh_deu"] for p in KH["ket_qua"][T]["tung_nguoi"]], n, "kh_deu")
        e = dict(tb=dict(ktnr=100 * ktnr.mean(), tinh=100 * x("tinh").mean(), tinh_r=100 * x("tinh_r").mean(), kh_deu=100 * kh.mean()),
                 KTNR_tru_tinh=ci(ktnr - x("tinh")), KTNR_tru_tinhR=ci(ktnr - x("tinh_r")), KTNR_tru_khdeu=ci(ktnr - kh))
        if ds in ("benchmark", "beta"):
            R1 = json.load(open(os.path.join(CU, "rcs", "%s_rcs-tai-hien_seed20261107_20261114.json" % ds), encoding="utf-8"))
            rcs = hop_le([p["day_du_lo"]["sau"] for p in R1["ket_qua"][T]["tung_nguoi"]], n, "rcs")
            e["tb"]["rcs_day_du"] = 100 * rcs.mean(); e["KTNR_tru_RCS_day_du"] = ci(ktnr - rcs)
            for bt in ("22", "du"):
                fl = sorted(glob.glob(os.path.join(CU, "ed", "%s_%s_%s" % (bt, ds, tt), "fold*.json")))
                J = [json.load(open(q, encoding="utf-8")) for q in fl]
                assert [j["fold"] for j in J] == list(range(1, n + 1)) and all(j["ds"] == ds and abs(j["T"] - float(T)) < 1e-9 and j["bien_the"] == bt for j in J), (bt, ds, T)
                ed = hop_le([j["sau"] for j in J], n, "ed" + bt)
                e["tb"]["ED_" + bt] = 100 * ed.mean(); e["KTNR_tru_ED_" + bt] = ci(ktnr - ed)
        e["tb"] = {k: float(v) for k, v in e["tb"].items()}
        out["bo"][ds][T] = e
        print(ds, T, {k: round(v, 1) for k, v in e["tb"].items()})
        print("    ", " | ".join("%s %s" % (k, f(v)) for k, v in e.items() if isinstance(v, dict) and "thap" in v))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
