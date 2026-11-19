#!/usr/bin/env python
"""Ghep cap theo nguoi CSS-R loc pha khong (pk) voi CSS-R loc nhan qua (nq) cua bai va voi OACCA5, OACCA1, FBCCA, CCA1 pk cua loc_nhan_qua_phan_ra.py.
CI 95 % BCa 10 000 lan, seed 20261107 (ham ci cua loc_nhan_qua_cap.py). Cham tu luot 41, 1,0 s.
Dung: python loc_pk_cssr_cap.py <out.json> <thu muc ktnr nq> <thu muc phan_ra> <thu muc cssr_pk>
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loc_nhan_qua_cap import ci, CSSR  # noqa: E402


def doc_cssr(p):
    t = sorted(json.load(open(p, encoding="utf-8"))["tung_nguoi"], key=lambda e: e["fold"])
    return {k: 100 * np.array([e[k] for e in t]) for k in ("ktnr", "ktn", "tinh")}, [e["fold"] for e in t]


def doc_pp(d, ds, pp, k):
    f = glob.glob(os.path.join(d, "%s_%s_pk-p0-d0-giu_T_seed20261107_*.json" % (ds, pp)))
    f = [x for x in f if not x.endswith("_config.json")]
    assert len(f) == 1, (ds, pp, f)
    t = sorted(json.load(open(f[0], encoding="utf-8"))["tung_nguoi"], key=lambda e: e["nguoi"])
    return 100 * np.array([e[k]["sau"] for e in t]), [e["nguoi"] for e in t]


def main():
    out, d_nq, d_pa, d_pk = sys.argv[1:5]
    if os.path.exists(out):
        sys.exit("tep ket qua da co: khong ghi de")
    kq = {}
    for ds in ("benchmark", "beta"):
        f = glob.glob(os.path.join(d_pk, "%s_ktnr-buoc1-pk_T10_seed20261107_*.json" % ds))
        f = [x for x in f if not x.endswith("_tien-xu-ly.json")]
        if not f:
            print(ds, "chua co ket qua pk"); continue
        pk, n1 = doc_cssr(f[0]); nq, n2 = doc_cssr(os.path.join(d_nq, CSSR[ds]))
        assert n1 == n2 == list(range(1, len(n1) + 1))
        r = dict(n=len(n1), tep_pk=os.path.basename(f[0]), tb={})
        for k in ("ktnr", "ktn", "tinh"):
            r["tb"]["%s_pk" % k] = float(pk[k].mean()); r["tb"]["%s_nq" % k] = float(nq[k].mean())
        r["cssr_pk_tru_nq"] = ci(pk["ktnr"] - nq["ktnr"])
        r["nen_pk_tru_nq"] = ci(pk["tinh"] - nq["tinh"])
        r["css_pk_tru_nq"] = ci(pk["ktn"] - nq["ktn"])
        r["loi_pk"] = ci(pk["ktnr"] - pk["tinh"]); r["loi_nq"] = ci(nq["ktnr"] - nq["tinh"])
        for pp, k in (("oacca5", "oacca"), ("oacca1", "oacca"), ("fbcca", "fbcca"), ("fbcca", "cca1")):
            v, nn = doc_pp(d_pa, ds, pp, k); assert nn == n1
            ten = pp if k != "cca1" else "cca1"
            r["tb"][ten + "_pk"] = float(v.mean())
            r["cssr_pk_tru_" + ten] = ci(pk["ktnr"] - v)
            r["cssr_nq_tru_" + ten] = ci(nq["ktnr"] - v)
        kq[ds] = r
        print("==", ds, "n", r["n"])
        print("  TB:", ", ".join("%s %.2f" % (a, b) for a, b in r["tb"].items()))
        for a, b in r.items():
            if isinstance(b, dict) and "thap" in b:
                print("  %-22s %+6.2f [%6.2f, %6.2f] %s" % (a, b["tb"], b["thap"], b["cao"], b["pp"]))
    json.dump(dict(seed=20261107, ket_qua=kq), open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
