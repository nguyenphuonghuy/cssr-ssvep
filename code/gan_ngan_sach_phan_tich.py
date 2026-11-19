#!/usr/bin/env python
"""Analysis: CSS-R versus single-band CCA, OACCA (5 and 1 subbands) and FBCCA, paired per subject,
95 % BCa CI with 10 000 resamples, seed 20261107 (same ci() as ktnr_chon_nguon_dc_phan_tich.py).
Two scorings: trials 41..end ("sau", the paper's main scoring) and the whole session ("phien"; CSS-R's first 40 trials are decided by
the base network, from results/accuracy/hoanvi).
Usage (from the package root): python code/gan_ngan_sach_phan_tich.py <out.json>
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ktnr_chon_nguon_dc_phan_tich import ci, SEED   # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "accuracy")
BO = {"benchmark": "benchmark", "beta": "beta", "eld": "eldbeta", "dong": "dong"}


def cssr(ds, T):
    d = json.load(open(os.path.join(R, "hoanvi", "%s_ktnr-hoanvi_T%s_seed20261107_20261118.json" % (BO[ds], {"1.0": "10", "0.6": "06"}[T])), encoding="utf-8"))
    sau, phien = [], []
    for k, p in enumerate(d["tung_nguoi"]):
        assert p["fold"] == k + 1
        g = p["goc"]; n = len(g["chi_so_cham"])
        sau.append(g["ktnr"]); phien.append((40 * g["khoi_tao_mang_nen"] + n * g["ktnr"]) / (40 + n))
    return 100 * np.array(sau), 100 * np.array(phien)


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        sys.exit("output exists, not overwriting")
    c1 = json.load(open(os.path.join(R, "gan_ngan_sach", "cca1", "bon-bo_cca1_nq_seed20261107_20261120.json")))
    kq = {}
    for ds in ("benchmark", "beta", "eld", "dong"):
        for T in ("1.0", "0.6"):
            k_sau, k_ph = cssr(ds, T)
            pp = {}
            g = c1["bo"][ds][T]["tung_nguoi"]
            pp["cca1"] = (100 * np.array([p["sau"] for p in g]), 100 * np.array([p["phien"] for p in g]))
            pp["fbcca"] = (100 * np.array([p["fbcca_sau"] for p in g]), 100 * np.array([p["fbcca_phien"] for p in g]))
            for nb in (5, 1):
                o = json.load(open(os.path.join(R, "gan_ngan_sach", "oacca", "%s_oacca%d_nq_seed20261107_20261120.json" % (ds, nb))))
                assert [p["nguoi"] for p in o["tung_nguoi"]] == list(range(1, len(k_sau) + 1))
                pp["oacca%d" % nb] = (100 * np.array([p["T"][T]["oacca"]["sau"] for p in o["tung_nguoi"]]),
                                      100 * np.array([p["T"][T]["oacca"]["phien"] for p in o["tung_nguoi"]]))
                pp["mac_oacca%d" % nb] = o["mac"][T]
            e = dict(n=len(k_sau), cssr=dict(sau=float(k_sau.mean()), phien=float(k_ph.mean())), mac_cca1=c1["bo"][ds][T]["mac_1dai"])
            for m in ("cca1", "fbcca", "oacca5", "oacca1"):
                s, p = pp[m]; assert len(s) == len(k_sau)
                e[m] = dict(sau=float(s.mean()), phien=float(p.mean()), cssr_tru_sau=ci(k_sau - s), cssr_tru_phien=ci(k_ph - p))
            e["mac_oacca5"], e["mac_oacca1"] = pp["mac_oacca5"], pp["mac_oacca1"]
            kq["%s_%s" % (ds, T)] = e
            print("%-10s %s CSS-R %.1f/%.1f" % (ds, T, e["cssr"]["sau"], e["cssr"]["phien"]))
            for m in ("cca1", "fbcca", "oacca5", "oacca1"):
                a, b = e[m]["cssr_tru_sau"], e[m]["cssr_tru_phien"]
                print("   %-7s %.1f/%.1f  sau %+.2f [%.2f, %.2f]  phien %+.2f [%.2f, %.2f]" % (m, e[m]["sau"], e[m]["phien"], a["tb"], a["thap"], a["cao"],
                                                                                         b["tb"], b["thap"], b["cao"]))
    json.dump(dict(seed=SEED, script=os.path.basename(__file__), ket_qua=kq), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
