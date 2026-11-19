#!/usr/bin/env python
"""Phan tich XAC NHAN tren BETA cho ung vien SF kieu B (mo-dun loc khong gian = TB cua 5 mo-dun nguon duoc chon khong nhan, giu co dinh).
Chenh ghep cap theo nguoi (70, thu tu S1..S70), luot 41..160, CI 95 % BCa 10.000 lan, seed 20261107.
  Y1: SF_B - kh_deu.
  Y2: SF_B - RCS trong tran manh nhat (bien the nhip lo co TB cao nhat trong {tran_lo, tran1R_lo, tranK_lo, tranKR_lo} o tung cua so).
Bao kem: SF_B - mang tinh (CF), - ct_R, - RCS day du; ca phien; kiem hoi quy Benchmark (SF_B phai trung job 18691).
Dung: python b1moi_lai_beta_phan_tich.py <thu muc lai_beta> <thu muc ket-qua-b1moi> <json ra>
"""
import json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
D, CU, RA = sys.argv[1:4]


def ci(a):
    a = 100 * np.asarray(a, float)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high))


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
k = json.load(open(os.path.join(CU, "beta_b1moi_buoc0b_seed20261107_20261114.json"), encoding="utf-8"))
c = json.load(open(os.path.join(CU, "ct", "beta_b1moi_ct_seed20261107_20261114.json"), encoding="utf-8"))
r1 = json.load(open(os.path.join(CU, "rcs", "beta_rcs-tai-hien_seed20261107_20261114.json"), encoding="utf-8"))
r2 = json.load(open(os.path.join(CU, "rcs", "beta_rcs-rieng_seed20261107_20261114.json"), encoding="utf-8"))
assert r1["nguoi"] == list(range(1, 71)) and r2["nguoi"] == list(range(1, 71)) and k["n_subjects"] == 70
assert k["ds"] == "beta" and c["ds"] == "beta" and c["n_subjects"] == 70 and r1["ds"] == "beta" and r2["ds"] == "beta" and k["seed"] == SEED and c["seed"] == SEED
assert all(len(k["ket_qua"][T]["tung_nguoi"]) == 70 and len(c["ket_qua"][T]["tung_nguoi"]) == 70 for T in ("0.6", "1.0"))
out = dict(seed=SEED, T={}); vp = []
for T, fn in (("0.6", "T06"), ("1.0", "T10")):
    j = json.load(open(os.path.join(D, "beta_b1moi_lai-xn_%s_seed20261107_20261114.json" % fn), encoding="utf-8"))
    per = j["tung_nguoi"]
    assert abs(j["T"] - float(T)) < 1e-9 and j["n0"] == 40 and j["k_src"] == 5 and j["buoc_nguon"] == 100 and j["seed"] == SEED
    assert j["ds"] == "beta" and j["xac_nhan"] and len(per) == 70 and [p["fold"] for p in per] == list(range(1, 71))
    sf = np.array([p["B"]["SF"]["sau"] for p in per]); cf = np.array([p["B"]["CF"]["sau"] for p in per])
    kh = np.array([p["40"]["kh_deu"] for p in k["ket_qua"][T]["tung_nguoi"]]); ctr = np.array([p["40"]["ct_R"] for p in c["ket_qua"][T]["tung_nguoi"]])
    rc = {v: np.array([p[v]["sau"] for p in r1["ket_qua"][T]["tung_nguoi"]]) for v in ("tran_lo", "tran1R_lo", "day_du_lo")}
    rc.update({v: np.array([p[v]["sau"] for p in r2["ket_qua"][T]["tung_nguoi"]]) for v in ("tranK_lo", "tranKR_lo")})
    manh = max(("tran_lo", "tran1R_lo", "tranK_lo", "tranKR_lo"), key=lambda v: rc[v].mean())
    assert all(np.isfinite(x).all() and len(x) == 70 for x in [sf, cf, kh, ctr] + list(rc.values()))
    e = dict(tb=dict(SF_B=float(100 * sf.mean()), SF_B_phien=float(100 * np.mean([p["B"]["SF"]["phien"] for p in per])), mang_tinh=float(100 * cf.mean()),
                     kh_deu=float(100 * kh.mean()), ct_R=float(100 * ctr.mean()), **{v: float(100 * x.mean()) for v, x in rc.items()}),
             q_nguon_tb=float(100 * np.mean([p["B"]["q_nguon_tb"] for p in per])), q_nguon_chon=float(100 * np.mean([p["B"]["q_nguon_chon"] for p in per])),
             rcs_trong_tran_manh_nhat=manh, Y1=ci(sf - kh), Y2=ci(sf - rc[manh]), tru_mang_tinh=ci(sf - cf), tru_ct_R=ci(sf - ctr), tru_rcs_day_du=ci(sf - rc["day_du_lo"]))
    out["T"][T] = e
    print(T, {a: round(b, 1) for a, b in e["tb"].items()}, "q nguon TB/chon %.1f/%.1f" % (e["q_nguon_tb"], e["q_nguon_chon"]))
    print("    Y1 SF_B-kh_deu", f(e["Y1"]), "| Y2 SF_B-%s" % manh, f(e["Y2"]), "| -mang tinh", f(e["tru_mang_tinh"]), "| -ct_R", f(e["tru_ct_R"]), "| -RCS day du", f(e["tru_rcs_day_du"]))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
