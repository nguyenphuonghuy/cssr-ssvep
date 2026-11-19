#!/usr/bin/env python
"""Phan tich KIEM DOC LAP tren eldBETA (9 muc tieu, 100 nguoi, 63 luot) cho ung vien SF kieu B.
Chenh ghep cap theo nguoi (S1..S100), luot 41..63, CI 95 % BCa 10.000 lan, seed 20261107.
  Z1: SF_B - kh_deu. Z2: SF_B - mang tinh (CF, cung mang nen; dong gop rieng cua chon nguon).
Bao kem: - ct_R, - FBCCA, ca phien. (Chua tai hien RCS tren eldBETA: khong co doi chung RCS.)
Dung: python b1moi_lai_eld_phan_tich.py <thu muc lai_eld> <thu muc ket-qua-b1moi> <json ra>
"""
import json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
D, CU, RA = sys.argv[1:4]


def ci(a):
    a = 100 * np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()))
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high))


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
k = json.load(open(os.path.join(CU, "eldbeta_b1moi_kh_seed20261107_20261114.json"), encoding="utf-8"))
c = json.load(open(os.path.join(CU, "ct", "eldbeta_b1moi_ct_seed20261107_20261114.json"), encoding="utf-8"))
assert k["n_subjects"] == 100 and c["n_subjects"] == 100 and k["seed"] == SEED and c["seed"] == SEED and k["n0"] == [40] and c["n0"] == [40]
out = dict(seed=SEED, T={})
for T, fn in (("0.6", "T06"), ("1.0", "T10")):
    j = json.load(open(os.path.join(D, "eldbeta_b1moi_lai-xn_%s_seed20261107_20261114.json" % fn), encoding="utf-8"))
    per = j["tung_nguoi"]
    assert j["ds"] == "eld" and j["xac_nhan"] and abs(j["T"] - float(T)) < 1e-9 and j["n0"] == 40 and j["k_src"] == 5 and j["buoc_nguon"] == 100 and j["seed"] == SEED
    assert len(per) == 100 and [p["fold"] for p in per] == list(range(1, 101))
    pk = [p["40"] for p in k["ket_qua"][T]["tung_nguoi"]]; pc = [p["40"] for p in c["ket_qua"][T]["tung_nguoi"]]
    assert len(pk) == 100 and len(pc) == 100
    sf = np.array([p["B"]["SF"]["sau"] for p in per]); cf = np.array([p["B"]["CF"]["sau"] for p in per])
    kh = np.array([p["kh_deu"] for p in pk]); ctr = np.array([p["ct_R"] for p in pc]); fb = np.array([p["fbcca"] for p in pk])
    assert all(np.isfinite(x).all() for x in (sf, cf, kh, ctr, fb))
    e = dict(tb=dict(SF_B=float(100 * sf.mean()), SF_B_phien=float(100 * np.mean([p["B"]["SF"]["phien"] for p in per])), mang_tinh=float(100 * cf.mean()),
                     kh_deu=float(100 * kh.mean()), ct_R=float(100 * ctr.mean()), fbcca=float(100 * fb.mean())),
             q_nguon_tb=float(100 * np.mean([p["B"]["q_nguon_tb"] for p in per])), q_nguon_chon=float(100 * np.mean([p["B"]["q_nguon_chon"] for p in per])),
             Z1=ci(sf - kh), Z2=ci(sf - cf), tru_ct_R=ci(sf - ctr), tru_fbcca=ci(sf - fb), mang_tinh_tru_kh_deu=ci(cf - kh))
    out["T"][T] = e
    print(T, {a: round(b, 1) for a, b in e["tb"].items()}, "q nguon TB/chon %.1f/%.1f" % (e["q_nguon_tb"], e["q_nguon_chon"]))
    print("    Z1 SF_B-kh_deu", f(e["Z1"]), "| Z2 SF_B-mang tinh", f(e["Z2"]), "| -ct_R", f(e["tru_ct_R"]), "| -FBCCA", f(e["tru_fbcca"]), "| mang tinh-kh_deu", f(e["mang_tinh_tru_kh_deu"]))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
