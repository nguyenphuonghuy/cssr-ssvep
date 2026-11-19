#!/usr/bin/env python
"""Summary of the ZCU104 energy session (zcu104_nangluong_sweep.sh). Usage: python zcu104_nangluong_tong_hop.py <session dir> <out json>
For each run: idle power = mean over [nghi_dau + 2 s, nghi_cuoi]; run power = mean over the inner 85 % of [t_dau, t_cuoi]
(10 % trimmed at the start, 5 % at the end); incremental energy per decision E = (P_run - P_idle) x mean decision time.
Reported for the INA226 (board 12 V input) and the IRPS5401 0x43 / 0x44 regulator inputs; mean and t 95 % CI over the 3 repetitions.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

ph, out = sys.argv[1], sys.argv[2]
if os.path.exists(out):
    sys.exit("output exists, not overwriting")
M = np.loadtxt(os.path.join(ph, "mau_cong_suat.txt"), comments="#")
t, P = M[:, 0], M[:, 1:] / 1e6                                               # W
ten_kenh = ["ina226", "irps_0043", "irps_0044"]
moc = {}
for ln in open(os.path.join(ph, "moc.txt")):
    k, ten, v = ln.split()
    moc.setdefault(ten, {})[k] = float(v)
dt = np.diff(t)


def tb(a, b):
    m = (t >= a) & (t <= b)
    assert m.sum() >= 20, (a, b, m.sum())
    return np.nanmean(P[m], axis=0), int(m.sum())


chay = {}
for f in sorted(glob.glob(os.path.join(ph, "chay", "*.json"))):
    r = json.loads(open(f).readline()); ten = os.path.basename(f)[:-5]
    d = r["t_cuoi"] - r["t_dau"]
    p0, n0 = tb(moc[ten]["nghi_dau"] + 2, moc[ten]["nghi_cuoi"])
    p1, n1 = tb(r["t_dau"] + 0.10 * d, r["t_cuoi"] - 0.05 * d)
    e = (p1 - p0) * r["tb_ms"]                                                # W x ms = mJ
    nhom = ten.rsplit("_r", 1)[0]
    chay.setdefault(nhom, []).append(dict(ten=ten, tb_ms=r["tb_ms"], so_quyet_dinh=r["so_quyet_dinh"], giay=d,
                                         P_nghi_W=dict(zip(ten_kenh, p0.tolist())), P_chay_W=dict(zip(ten_kenh, p1.tolist())),
                                         E_mJ=dict(zip(ten_kenh, e.tolist())), so_mau_nghi=n0, so_mau_chay=n1))


def ci(v):
    v = np.asarray(v, float); m = float(v.mean())
    if len(v) < 2:
        return dict(tb=m, n=len(v))
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return dict(tb=m, ci95=[m - h, m + h], sd=float(v.std(ddof=1)), n=len(v))


tong = {}
for nhom, rs in chay.items():
    tong[nhom] = dict(tb_ms=ci([r["tb_ms"] for r in rs]),
                      **{"E_mJ_" + k: ci([r["E_mJ"][k] for r in rs]) for k in ten_kenh},
                      **{"dP_W_" + k: ci([r["P_chay_W"][k] - r["P_nghi_W"][k] for r in rs]) for k in ten_kenh},
                      P_nghi_W_ina226=ci([r["P_nghi_W"]["ina226"] for r in rs]))
res = dict(phien=os.path.basename(os.path.normpath(ph)), tan_so_lay_mau_hz=float(1 / np.median(dt)), khoang_lay_mau_max_s=float(dt.max()),
           script=os.path.basename(__file__), tong_hop=tong, tung_lan=chay)
json.dump(res, open(out, "w"), indent=1)
for nhom in sorted(tong):
    g = tong[nhom]
    print("%-16s t=%8.2f ms  dP=%6.3f W  E=%9.3f mJ (INA226)" % (nhom, g["tb_ms"]["tb"], g["dP_W_ina226"]["tb"], g["E_mJ_ina226"]["tb"]))
