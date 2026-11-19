#!/usr/bin/env python
"""Phan tich doi chung luat chon nguon.
Doc cac JSON cua ktnr_chon_nguon_dc.py, tinh cho tung bo: TB CSS-R, CSS cua tung luat (%), chenh ghep cap do_tin - luat khac
(CI 95 % BCa 10 000 lan, seed 20261107; BCa loi hoac khong huu han -> percentile; chenh hang -> CI = TB), so nguon trung, so fold lech hoi quy.
Cach dung: ktnr_chon_nguon_dc_phan_tich.py <out.json> <tep1.json> [<tep2.json> ...]
"""
import json, sys, os
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107


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
    out = sys.argv[1]
    if os.path.exists(out):
        sys.exit("da co: khong ghi de")
    kq = {}
    for f in sys.argv[2:]:
        d = json.load(open(f, encoding="utf-8")); P = d["tung_nguoi"]; L = d["luat"]
        v = {lu: {m: 100 * np.array([r[lu][m] for r in P]) for m in ("ktn", "ktnr")} for lu in L}
        e = dict(n=len(P), so_fold_lech_hoi_quy=d["so_fold_lech_hoi_quy"], hoi_quy_tung_fold=all(r.get("hoi_quy") for r in P),
                 tb={lu: {m: float(v[lu][m].mean()) for m in ("ktn", "ktnr")} for lu in L},
                 chenh_cssr={f"do_tin_tru_{lu}": ci(v["do_tin"]["ktnr"] - v[lu]["ktnr"]) for lu in L if lu != "do_tin"},
                 chenh_css={f"do_tin_tru_{lu}": ci(v["do_tin"]["ktn"] - v[lu]["ktn"]) for lu in L if lu != "do_tin"},
                 trung_tap_tb={k: float(np.mean([r["trung_tap"][k] for r in P])) for k in P[0]["trung_tap"]},
                 acc_5_nguon_chon_tb={lu: 100 * float(np.mean([r[lu]["acc_sau_tb_5_nguon"] for r in P])) for lu in L},
                 mo_dun_nguon={k: 100 * float(np.mean([r["mo_dun_nguon_sau"][k] for r in P])) for k in ("tb", "top5_tb", "thap5_tb")})
        kq[d["ds"]] = e
        print(d["ds"], "lech", e["so_fold_lech_hoi_quy"], {lu: round(e["tb"][lu]["ktnr"], 2) for lu in L})
        for k, c in e["chenh_cssr"].items():
            print("   ", k, "%+.2f [%.2f, %.2f] %s" % (c["tb"], c["thap"], c["cao"], c["pp"]))
    json.dump(dict(seed=SEED, tep=[os.path.basename(f) for f in sys.argv[2:]], bo=kq), open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
