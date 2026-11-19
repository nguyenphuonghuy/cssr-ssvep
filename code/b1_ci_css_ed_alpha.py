#!/usr/bin/env python
"""CI bao kem cho dong gop 2 cua B1 (2026-11-18): (a) CSS (KTN) - Ensemble-DNN, hai ban, Benchmark va BETA; (b) do nhay theo alpha cua CSS-R
Dung: python b1_ci_css_ed_alpha.py <json ra> <thu muc ket-qua-b1moi>
Nap du lieu va ham ci() giong b1_ci_bo_sung.py (BCa 10.000, seed 20261107, percentile du phong). Chenh ghep cap theo nguoi.
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


f = lambda c: "%+.2f [%+.2f; %+.2f] %s" % (c["tb"], c["thap"], c["cao"], c["phuong_phap"])
out = dict(seed=SEED, ghi_chu="bao kem", bo={})
for ds in ("benchmark", "beta", "eld"):
    n = SO[ds]; out["bo"][ds] = {}
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        fs_ = glob.glob(os.path.join(CU, "ktnr", "%s_ktnr-buoc1_%s_seed%d_*.json" % (TEN[ds], tt, SEED)))
        assert len(fs_) == 1, fs_
        P = json.load(open(fs_[0], encoding="utf-8"))["tung_nguoi"]
        assert [p["fold"] for p in P] == list(range(1, n + 1))
        x = lambda k: hop_le([p[k] for p in P], n, k)
        ktn, ktnr, a025, a1 = x("ktn"), x("ktnr"), x("ktnr_a025"), x("ktnr_a1")
        e = dict(tb=dict(ktn=100 * ktn.mean(), ktnr=100 * ktnr.mean(), ktnr_a025=100 * a025.mean(), ktnr_a1=100 * a1.mean()),
                 KTNR_a05_tru_a025=ci(ktnr - a025), KTNR_a05_tru_a1=ci(ktnr - a1))
        if ds in ("benchmark", "beta"):
            for bt in ("22", "du"):
                fl = sorted(glob.glob(os.path.join(CU, "ed", "%s_%s_%s" % (bt, ds, tt), "fold*.json")))
                J = [json.load(open(q, encoding="utf-8")) for q in fl]
                assert [j["fold"] for j in J] == list(range(1, n + 1)) and all(j["ds"] == ds and abs(j["T"] - float(T)) < 1e-9 and j["bien_the"] == bt for j in J), (bt, ds, T)
                ed = hop_le([j["sau"] for j in J], n, "ed" + bt)
                e["tb"]["ED_" + bt] = 100 * ed.mean(); e["KTN_tru_ED_" + bt] = ci(ktn - ed)
        e["tb"] = {k: float(v) for k, v in e["tb"].items()}
        out["bo"][ds][T] = e
        print(ds, T, {k: round(v, 2) for k, v in e["tb"].items()})
        print("    ", " | ".join("%s %s" % (k, f(v)) for k, v in e.items() if isinstance(v, dict) and "thap" in v))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
