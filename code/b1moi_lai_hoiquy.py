#!/usr/bin/env python
"""CONG HOI QUY (2026-11-14): so ket qua SF kieu B cua che do xac nhan tren Benchmark voi job 18691, TUNG FOLD. Ma thoat 4 neu khong khop.
Khop = du 35 fold dung thu tu; nguon duoc chon trung o >= 33/35 fold; |chenh do chinh xac SF_B tung fold| TB <= 0,005 va |chenh TB| <= 0,002
(dung sai nho cho sai khac so hoc GPU; job 18691: TB 0,801143).
Dung: python b1moi_lai_hoiquy.py <json hoi quy> <json job 18691>
"""
import json, sys
import numpy as np
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
pa, pb = a["tung_nguoi"], b["tung_nguoi"]
assert a["ds"] == "benchmark" and a["xac_nhan"] and abs(a["T"] - b["T"]) < 1e-9 and a["n0"] == b["n0"] and a["k_src"] == b["k_src"] and a["buoc_nguon"] == b["buoc_nguon"]
ok = len(pa) == 35 and len(pb) == 35 and [p["fold"] for p in pa] == list(range(1, 36)) and [p["fold"] for p in pb] == list(range(1, 36))
if ok:
    x = np.array([p["B"]["SF"]["sau"] for p in pa]); y = np.array([p["B"]["SF"]["sau"] for p in pb])
    trung = sum(p["B"]["nguon_chon"] == q["B"]["nguon_chon"] for p, q in zip(pa, pb))
    print("hoi quy: TB moi %.6f, TB job 18691 %.6f, |chenh| TB tung fold %.6f, nguon trung %d/35" % (x.mean(), y.mean(), np.abs(x - y).mean(), trung))
    ok = trung >= 33 and np.abs(x - y).mean() <= 0.005 and abs(x.mean() - y.mean()) <= 0.002
print("CONG HOI QUY:", "DAT" if ok else "KHONG DAT")
sys.exit(0 if ok else 4)
