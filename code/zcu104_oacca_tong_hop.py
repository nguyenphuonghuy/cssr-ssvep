#!/usr/bin/env python
"""Timing summary of the OACCA session on the ZCU104 (zcu104_oacca_sweep.sh). Usage: python zcu104_oacca_tong_hop.py <session dir> <out json>
Per configuration: mean over the 7 processes of the per-process mean time (trials 21..240), t 95 % CI, largest p95 and max.
Same-session CSS-R reference (1.0 s, 4 threads): time ratio OACCA / CSS-R, and the "faster" criterion of the joint session
(largest per-process mean of CSS-R < smallest per-process mean of the other method).
Energy is summarized separately with zcu104_nangluong_tong_hop.py on the same session folder.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

ph, out = sys.argv[1], sys.argv[2]
if os.path.exists(out):
    sys.exit("output exists, not overwriting")
nhom = {}
for f in sorted(glob.glob(os.path.join(ph, "gio", "*.json"))):
    r = json.loads(open(f).readline()); ten = os.path.basename(f)[:-5].rsplit("_r", 1)[0]
    nhom.setdefault(ten, []).append(r["tong"])


def tom(v):
    tb = np.array([x["tb_ms"] for x in v]); m = float(tb.mean())
    h = float(stats.t.ppf(0.975, len(tb) - 1) * tb.std(ddof=1) / np.sqrt(len(tb)))
    return dict(tb_ms=m, ci95=[m - h, m + h], tb_tien_trinh_min=float(tb.min()), tb_tien_trinh_max=float(tb.max()),
                p95_max_ms=float(max(x["p95_ms"] for x in v)), max_ms=float(max(x["max_ms"] for x in v)), so_tien_trinh=len(tb))


kq = {k: tom(v) for k, v in nhom.items()}
k = kq["ktnr_T1.0_L4"]
for t in ("oacca5_T1.0_L4", "oacca1_T1.0_L4"):
    kq[t]["ti_le_so_voi_cssr"] = kq[t]["tb_ms"] / k["tb_ms"]
    kq[t]["cssr_nhanh_hon"] = bool(k["tb_tien_trinh_max"] < kq[t]["tb_tien_trinh_min"])
json.dump(dict(phien=os.path.basename(os.path.normpath(ph)), script=os.path.basename(__file__), ket_qua=kq), open(out, "w"), indent=1)
for t in sorted(kq):
    print("%-16s %8.2f ms [%.2f, %.2f] p95max %.2f max %.2f %s" % (t, kq[t]["tb_ms"], *kq[t]["ci95"], kq[t]["p95_max_ms"], kq[t]["max_ms"],
                                                                   ("x%.1f" % kq[t]["ti_le_so_voi_cssr"]) if "ti_le_so_voi_cssr" in kq[t] else ""))
