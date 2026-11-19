#!/usr/bin/env python
"""Kiem hoi quy cho loc_nhan_qua_phan_ra.py: cau hinh nhu bai (F nq, loc tu mau 0, d = 0, T = 1,0 s) phai trung tung nguoi voi ket qua da co.
  FBCCA: sau, phien cua fbcca-dung_seed20261107_20261118.json (Benchmark, BETA, eld, Dong) va fbcca-dung_nak (Nakanishi).
  CCA mot dai: sau, phien cua bon-bo_cca1_nq_seed20261107_20261120.json.
  OACCA nam dai, mot dai: quyet dinh tung luot (dec) cua <ds>_oacca{5,1}_nq_seed20261107_20261120.json.
Dung: python loc_nhan_qua_kiem.py <thu muc tep tham chieu> <so nguoi moi bo>   (ma thoat 5 neu lech)
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc_nhan_qua_phan_ra as L  # noqa: E402

R = sys.argv[1]; K = int(sys.argv[2])
j = lambda f: json.load(open(os.path.join(R, f), encoding="utf-8"))
for d in L.B.DS:
    L.B.DS[d]["root"] = os.environ.get("B1_ROOT_" + d.upper(), L.B.DS[d]["root"])
L.DE.dang_ky_B(L.B); L.DG.dang_ky(L.B, None, "dong"); L.DN.dang_ky(L.B, None, "nak")
fb = j("fbcca-dung_seed20261107_20261118.json")["bo"]; fbn = j("fbcca-dung_nak_seed20261107_20261118.json")["bo"]
c1 = j("bon-bo_cca1_nq_seed20261107_20261120.json")["bo"]
lech = 0; t0 = time.time()


def cf(ds):
    c = L.B.DS[ds]; fs = c["fs"]
    return dict(F="nq", lat=int(round(c["lat"] * fs)), d=0, n=int(round(1.0 * fs)), p_mau=0)


def so(a, b, ten):
    global lech
    ok = abs(a - b) < 1e-12
    lech += int(not ok)
    if not ok:
        print("LECH", ten, a, b, flush=True)


for ds in ("benchmark", "beta", "eld", "dong", "nak"):
    ref_fb = (fbn if ds == "nak" else fb)[ds]["1.0"]["tung_nguoi"]
    for s in range(1, K + 1):
        e = L.fbcca_nguoi(ds, s, cf(ds))
        r = ref_fb[s - 1]; assert r["nguoi"] == s
        so(e["fbcca"]["sau"], r["sau"], (ds, s, "fbcca sau")); so(e["fbcca"]["phien"], r["phien"], (ds, s, "fbcca phien"))
        if ds in c1:
            r1 = c1[ds]["1.0"]["tung_nguoi"][s - 1]; assert r1["nguoi"] == s
            so(e["cca1"]["sau"], r1["sau"], (ds, s, "cca1 sau")); so(e["cca1"]["phien"], r1["phien"], (ds, s, "cca1 phien"))
        if ds != "nak":
            for nb in (5, 1):
                ro = j("%s_oacca%d_nq_seed20261107_20261120.json" % (ds, nb))["tung_nguoi"]
                ro = [x for x in ro if x["nguoi"] == s][0]["T"]["1.0"]["oacca"]["dec"]
                eo = L.oacca_nguoi(ds, s, nb, cf(ds))["oacca"]["dec"]
                ok = eo == ro; lech += int(not ok)
                print(ds, s, "oacca%d" % nb, "trung" if ok else "LECH %d luot" % int(np.sum(np.array(eo) != np.array(ro))), flush=True)
        print(ds, s, "fbcca %.4f cca1 %.4f" % (e["fbcca"]["sau"], e["cca1"]["sau"]), "%.0f s" % (time.time() - t0), flush=True)
print("KIEM HOI QUY", "DAT" if lech == 0 else "LECH %d" % lech)
sys.exit(0 if lech == 0 else 5)
