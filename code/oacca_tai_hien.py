#!/usr/bin/env python
"""Reproduction check of the OACCA port against the original paper (Wong et al., IEEE TBME 2022, Sec. IV-A, text after Fig. 9):
averaged over subjects, time windows (0.6..1.5 s) and 10 trial permutations, accuracy CCA 0.61 -> OACCA 0.84 on Dataset I (Benchmark)
and CCA 0.57 -> OACCA 0.72 on Dataset II (BETA). The port is run with --che-do goc (zero-phase filtering, latency 0.12/0.13 s, 2 s segment).
Note: the "CCA" of doi_thu_oacca is the 5-subband filter-bank CCA used inside OACCA (sum_k w_k r1_k), as in the released code.
Usage: python oacca_tai_hien.py <out json> <json> [<json> ...]   (part files by subject range)
"""
import json, os, sys
import numpy as np

GOC = {"benchmark": dict(cca=0.61, oacca=0.84), "beta": dict(cca=0.57, oacca=0.72)}
out = sys.argv[1]
if os.path.exists(out):
    sys.exit("output exists, not overwriting")
per = {}
for f in sys.argv[2:]:
    d = json.load(open(f))
    assert d["che_do"] == "goc" and d["so_lan_xao"] == 10
    for e in d["tung_nguoi"]:
        per.setdefault(d["ds"], {})[e["nguoi"]] = e
kq = {}
for ds, P in per.items():
    ng = sorted(P)
    Ts = sorted(P[ng[0]]["T"], key=float)
    A = {m: np.array([[np.mean(P[s]["T"][T][m]) for T in Ts] for s in ng]) for m in ("cca", "oacca")}   # nguoi x TW (TB 10 lan xao)
    kq[ds] = dict(so_nguoi=len(ng), du_nguoi=ng == list(range(1, len(ng) + 1)), cua_so=Ts,
                  tb_chung={m: float(A[m].mean()) for m in A}, bai_goc=GOC.get(ds),
                  lech_so_voi_bai_goc={m: float(A[m].mean() - GOC[ds][m]) for m in A} if ds in GOC else None,
                  theo_cua_so={T: {m: float(A[m][:, j].mean()) for m in A} for j, T in enumerate(Ts)})
    g = kq[ds]
    print(ds, "n=%d" % g["so_nguoi"], "CCA %.3f (goc %.2f)  OACCA %.3f (goc %.2f)" % (g["tb_chung"]["cca"], GOC[ds]["cca"], g["tb_chung"]["oacca"], GOC[ds]["oacca"]))
    print("   1.0 s: CCA %.3f OACCA %.3f" % (g["theo_cua_so"]["1.0"]["cca"], g["theo_cua_so"]["1.0"]["oacca"]))
json.dump(dict(nguon_bai_goc="Wong et al. 2022, IEEE TBME 69(6):2018-2028, DOI 10.1109/TBME.2021.3133594, muc IV-A sau Hinh 9", ket_qua=kq), open(out, "w"), indent=1)
