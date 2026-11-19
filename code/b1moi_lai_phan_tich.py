#!/usr/bin/env python
"""Phan tich phep thu 2 x 2.
Chenh ghep cap theo nguoi (35), CI 95 % BCa 10.000 lan, seed 20261107. Xet rieng tung kieu mo-dun (A, B), 1,0 s:
  L1: SF - CF (luot 41-240). L2: SU - CU tren luot 41-120 (TB hai doan 40 luot dau).
  L3: SU - doi chung toan mang co TB cao hon trong {all_seq, all_seq_e} (41-240).
Dung: python b1moi_lai_phan_tich.py <json T10> <json T06> <json ra>
"""
import json, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107


def ci(a):
    a = 100 * np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()))
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high))


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
out = dict(seed=SEED, T={})
for ten, fn in (("1.0", sys.argv[1]), ("0.6", sys.argv[2])):
    j = json.load(open(fn, encoding="utf-8")); per = j["tung_nguoi"]
    assert len(per) == 35 and [p["fold"] for p in per] == list(range(1, 36)) and j["n0"] == 40
    s = lambda p: np.array([q["sau"] for q in p]); dau = lambda p: np.array([np.mean(q["doan"][:2]) for q in p])
    ctl = max(("all_seq", "all_seq_e"), key=lambda k: s([p[k] for p in per]).mean())
    e = dict(tb={"tinh": float(100 * s([p["tinh"] for p in per]).mean()), "all_seq": float(100 * s([p["all_seq"] for p in per]).mean()),
                 "all_seq_e": float(100 * s([p["all_seq_e"] for p in per]).mean())}, doi_chung_L3=ctl, kieu={})
    for k in ("A", "B"):
        o = {x: [p[k][x] for p in per] for x in ("CF", "CU", "SF", "SU")}
        d = dict(tb={x: float(100 * s(o[x]).mean()) for x in o}, tb_41_120={x: float(100 * dau(o[x]).mean()) for x in o},
                 doan_SU=[float(100 * np.mean([q["doan"][m] for q in o["SU"]])) for m in range(5)],
                 doan_CU=[float(100 * np.mean([q["doan"][m] for q in o["CU"]])) for m in range(5)],
                 q_nguon_tb=float(100 * np.mean([p[k]["q_nguon_tb"] for p in per])), q_nguon_chon=float(100 * np.mean([p[k]["q_nguon_chon"] for p in per])),
                 L1=ci(s(o["SF"]) - s(o["CF"])), L2=ci(dau(o["SU"]) - dau(o["CU"])), L3=ci(s(o["SU"]) - s([p[ctl] for p in per])),
                 CU_tru_CF=ci(s(o["CU"]) - s(o["CF"])), SU_tru_SF=ci(s(o["SU"]) - s(o["SF"])), SU_tru_CU=ci(s(o["SU"]) - s(o["CU"])))
        d["dat"] = dict(L1=bool(d["L1"]["thap"] > 0), L2=bool(d["L2"]["thap"] > 0), L3=bool(d["L3"]["thap"] > -2))
        if ten == "1.0":
            d["de_nghi_gia_thuyet_chinh"] = bool(d["dat"]["L2"] and d["dat"]["L3"] and d["tb"]["SU"] >= 78.7)
        e["kieu"][k] = d
        print(ten, k, {x: round(v, 1) for x, v in d["tb"].items()}, "q nguon TB/chon %.1f/%.1f" % (d["q_nguon_tb"], d["q_nguon_chon"]))
        print("    L1 SF-CF", f(d["L1"]), "| L2 SU-CU(41-120)", f(d["L2"]), "| L3 SU-%s" % ctl, f(d["L3"]), "| CU-CF", f(d["CU_tru_CF"]), "| SU-SF", f(d["SU_tru_SF"]), d["dat"], d.get("de_nghi_gia_thuyet_chinh"))
    print(ten, {x: round(v, 1) for x, v in e["tb"].items()})
    out["T"][ten] = e
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
