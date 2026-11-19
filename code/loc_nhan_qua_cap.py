#!/usr/bin/env python
"""Tong hop ket qua cua loc_nhan_qua_phan_ra.py: bang trung binh theo cau hinh, chon d* (theo TB Benchmark va BETA, cham tu luot 41)
cho tung phuong phap va bien the, ghep cap theo nguoi voi CSS-R (CI 95 % BCa 10 000 lan, seed 20261107).
CSS-R tung nguoi: ket qua da co cua bai (truong "ktnr", luot 41..het), 1,0 s.
Dung: python loc_nhan_qua_cap.py <out.json> <thu muc ket qua cssr> <tep json loc_nhan_qua ...>
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
CSSR = {"benchmark": "benchmark_ktnr-buoc1_T10_seed20261107_20261115.json", "beta": "beta_ktnr-buoc1_T10_seed20261107_20261115.json",
        "eld": "eldbeta_ktnr-buoc1_T10_seed20261107_20261115.json", "dong": "dong_ktnr-xn_T10_seed20261107_20261117.json",
        "nak": "nak_ktnr-buoc2_T10_seed20261107_20261116.json"}


def ci(a):
    a = np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()), pp="hang")
    pp = "BCa"
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), pp=pp)


def main():
    out, rc, tep = sys.argv[1], sys.argv[2], sys.argv[3:]
    if os.path.exists(out):
        sys.exit("tep ket qua da co: khong ghi de")
    cssr = {}
    for ds, f in CSSR.items():
        p = sorted(json.load(open(os.path.join(rc, f), encoding="utf-8"))["tung_nguoi"], key=lambda e: e["fold"])
        cssr[ds] = np.array([e["ktnr"] for e in p])
    hang = []
    for f in tep:
        if f.endswith("_config.json"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        ks = ["fbcca", "cca1"] if d["pp"] == "fbcca" else ["oacca"]
        for k in ks:
            ten = k if k != "oacca" else d["pp"]
            v = np.array([e[k]["sau"] for e in sorted(d["tung_nguoi"], key=lambda e: e["nguoi"])])
            h = dict(ds=d["ds"], pp=ten, F=d["F"], p=d["p_s"], d=d["d_s"], bien_the=d["bien_the"], n=len(v), tb_sau=100 * float(v.mean()),
                     tb_phien=100 * float(np.mean([e[k]["phien"] for e in d["tung_nguoi"]])), tep=os.path.basename(f))
            if d["ds"] in cssr and len(v) == len(cssr[d["ds"]]) and d["nguoi"] == list(range(1, len(v) + 1)):
                h["cssr_tru"] = ci(100 * (cssr[d["ds"]] - v))
            hang.append(h)
    # chon d*: chi F nq, p = 0, theo TB Benchmark va BETA
    chon = {}
    for pp in sorted({h["pp"] for h in hang}):
        for bt in ("giu_T", "giu_qd"):
            m = {}
            for h in hang:
                if h["pp"] == pp and h["F"] == "nq" and h["p"] == 0 and h["ds"] in ("benchmark", "beta") and (h["bien_the"] == bt or h["d"] == 0):
                    m.setdefault(h["d"], {})[h["ds"]] = h["tb_sau"]
            m = {d: np.mean(list(v.values())) for d, v in m.items() if len(v) == 2}
            if m:
                ds_ = max(m, key=lambda d: (round(m[d], 9), -d))
                chon["%s_%s" % (pp, bt)] = dict(d_sao=ds_, tb_phat_trien=m[ds_], luoi={str(k): v for k, v in sorted(m.items())})
    json.dump(dict(seed=SEED, hang=hang, d_sao=chon), open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for h in sorted(hang, key=lambda h: (h["ds"], h["pp"], h["F"], h["p"], h["bien_the"], h["d"])):
        c = h.get("cssr_tru")
        print("%-9s %-6s %-5s p%-4s d%-4s %-6s %6.2f" % (h["ds"], h["pp"], h["F"], h["p"], h["d"], h["bien_the"], h["tb_sau"]),
              ("CSS-R tru %+6.2f [%6.2f, %6.2f]" % (c["tb"], c["thap"], c["cao"])) if c else "")
    for k, v in chon.items():
        print("d*", k, v["d_sao"], "%.2f" % v["tb_phat_trien"], v["luoi"])


if __name__ == "__main__":
    main()
