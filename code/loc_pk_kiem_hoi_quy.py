#!/usr/bin/env python
"""Kiem hoi quy truoc khi chay CSS-R voi loc pha khong tren bo dem. Ma thoat 5 neu lech.
  a: ket qua ktnr_buoc1 fold 1 Benchmark 1,0 s chay qua chay_loc.py --che-do nq phai trung ket qua da luu (ktnr, ktn, tinh, nguon chon).
  b: logit fold 1 Benchmark cua global_dnn_huanluyen.py qua chay_loc.py --che-do nq so voi b1v7_lo2/T10/fold001_logit-tinh.npz
     (cung ham huan luyen GT.huan_luyen, cung seed): bao lech lon nhat va ti le argmax trung; dat khi tuong quan logit >= 0,999 va chenh do chinh xac <= 1 diem.
  c: tien xu ly pk cua chay_loc.py trung cat(F = "pk") cua loc_nhan_qua_phan_ra.py voi day loc ba dai cua global_dnn (Benchmark nguoi 1).
Dung: python loc_pk_kiem_hoi_quy.py <json a moi> <json a goc> <thu muc b moi> <thu muc b goc>
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
lech = 0
a_moi = json.load(open(sys.argv[1]))["tung_nguoi"][0]
a_goc = [e for e in json.load(open(sys.argv[2]))["tung_nguoi"] if e["fold"] == 1][0]
for k in ("ktnr", "ktn", "tinh", "nguon_chon"):
    ok = a_moi[k] == a_goc[k]; lech += int(not ok)
    print("a", k, a_moi[k], a_goc[k], "trung" if ok else "LECH", flush=True)
lm = np.load(os.path.join(sys.argv[3], "fold001_logit-tinh.npz")); lg = np.load(os.path.join(sys.argv[4], "fold001_logit-tinh.npz"))
d = float(np.abs(lm["logit"] - lg["logit"]).max()); tr = float((lm["logit"].argmax(1) == lg["logit"].argmax(1)).mean())
ac = lambda z: float((z["logit"].argmax(1) == z["y"]).mean())
r = float(np.corrcoef(lm["logit"].ravel(), lg["logit"].ravel())[0, 1]); da = abs(ac(lm) - ac(lg))
okb = r >= 0.999 and da <= 0.01; lech += int(not okb)   # huan luyen GPU khong lap lai tung bit
print("b lech logit lon nhat %.3g, argmax trung %.4f, tuong quan %.5f, chenh do chinh xac %.4f" % (d, tr, r, da), "dat" if okb else "LECH", flush=True)
import b1_diem as B, global_dnn as GD   # noqa: E401,E402
import loc_nhan_qua_phan_ra as L          # noqa: E402
import chay_loc                            # noqa: E402
for dd in B.DS:
    B.DS[dd]["root"] = os.environ.get("B1_ROOT_" + dd.upper(), B.DS[dd]["root"])
X, y, fr = B.load_subject("benchmark", 1, B.CH9)
c = B.DS["benchmark"]; fs = c["fs"]; ns = 250; lat = int(round(c["lat"] * fs))
ba, sos = GD.bo_loc(fs)
ref = L.cat(X, "pk", c["onset"], lat, 0, ns, fs, sos, None, 0).astype(np.float32)
moi = chay_loc.tao_pk(GD)(X, "benchmark", "causal", ns, fs)
dc = float(np.abs(moi - ref).max()); okc = dc < 1e-5; lech += int(not okc)
print("c lech pk lon nhat %.3g" % dc, "dat" if okc else "LECH", flush=True)
print("KIEM HOI QUY", "DAT" if lech == 0 else "LECH %d" % lech)
sys.exit(0 if lech == 0 else 5)
