#!/usr/bin/env python
"""Phan tich D7 (tach co che KTN): KTN - ngau5 va cac phep so kem theo.
Moi bo (Benchmark, BETA, eldBETA) x cua so (0,6; 1,0 s): chenh ghep cap theo nguoi, CI 95 % BCa 10.000 lan, seed 20261107, luot 41..het.
  Bao kem: KTN - kem5, KTN - tatca, KTN - ens5 (gop du doan kieu Ensemble-DNN), ngau5 - mang tinh.
  Kiem nhat quan: d7_ktn phai trung SF (sau) tung nguoi.
Dung: python b1moi_d7_phan_tich.py <json ra> "benchmark|T10|T06" "beta|T10|T06" "eld|T10|T06"
"""
import json, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107


def ci(a):
    a = 100 * np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()))
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "BCa"
    except Exception:  # BCa suy bien thi dung percentile, co ghi lai
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    if not (np.isfinite(r.low) and np.isfinite(r.high)):
        raise SystemExit("CI khong huu han")
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
import os
CAN = {"benchmark": ("benchmark", 35), "beta": ("beta", 70), "eld": ("eld", 100)}           # buoc du dung ba bo, du nguoi
if os.path.exists(sys.argv[1]):
    raise SystemExit("tep ket qua da co: khong ghi de")
ds_vao = [a_.split("|")[0] for a_ in sys.argv[2:]]
if sorted(ds_vao) != sorted(CAN):
    raise SystemExit("phai co dung ba bo benchmark, beta, eld")
out = dict(seed=SEED, bo={}); h2s = []
for arg in sys.argv[2:]:
    ten, f10, f06 = arg.split("|")                                          # dung | de duong dan Windows co dau hai cham
    out["bo"][ten] = {}
    for T, fn in (("1.0", f10), ("0.6", f06)):
        j = json.load(open(fn, encoding="utf-8")); per = j["tung_nguoi"]
        assert j["d7"] and j["r_d7"] == 20 and j["seed"] == SEED and j["n0"] == 40 and abs(j["T"] - float(T)) < 1e-9
        assert j["ds"] == CAN[ten][0] and len(per) == CAN[ten][1] and [p["fold"] for p in per] == list(range(1, CAN[ten][1] + 1)), (ten, T)
        g = lambda k: np.array([p["B"][k] for p in per])
        sf = np.array([p["B"]["SF"]["sau"] for p in per]); cf = np.array([p["B"]["CF"]["sau"] for p in per])
        assert np.allclose(g("d7_ktn"), sf), "d7_ktn khac SF"
        e = dict(n=len(per), tb={k: float(100 * g(k).mean()) for k in ("d7_ktn", "d7_ngau5", "d7_kem5", "d7_tatca", "d7_ens5")}, mang_tinh=float(100 * cf.mean()),
                 H2s=ci(g("d7_ktn") - g("d7_ngau5")), tru_kem5=ci(g("d7_ktn") - g("d7_kem5")), tru_tatca=ci(g("d7_ktn") - g("d7_tatca")),
                 tru_ens5=ci(g("d7_ktn") - g("d7_ens5")), ngau5_tru_tinh=ci(g("d7_ngau5") - cf), tatca_tru_tinh=ci(g("d7_tatca") - cf))
        out["bo"][ten][T] = e
        if T == "1.0":
            h2s.append(e["H2s"])
        print(ten, T, "n", e["n"], {k[3:]: round(v, 1) for k, v in e["tb"].items()}, "tinh %.1f" % e["mang_tinh"])
        print("    H2s KTN-ngau5", f(e["H2s"]), "| -kem5", f(e["tru_kem5"]), "| -tatca", f(e["tru_tatca"]), "| -ens5", f(e["tru_ens5"]), "| ngau5-tinh", f(e["ngau5_tru_tinh"]))
assert len(h2s) == 3
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
