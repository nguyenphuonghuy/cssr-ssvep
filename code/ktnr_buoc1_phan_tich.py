#!/usr/bin/env python
"""Phan tich KTN-R buoc 1: KTN-R - KTN va cac phep so kem theo.
Dung: python ktnr_buoc1_phan_tich.py <json ra> <thu muc ket-qua-b1moi> <thu muc ket qua ktnr buoc 1>
  Cong hoi quy: 'ktn' tinh lai phai TRUNG SF (sau) tung nguoi cua lan chay KTN (sai khac > 1e-9 o bat ky nguoi nao -> thoat ma 4).
  CI: chenh ghep cap theo nguoi, BCa 10.000 lan, seed 20261107; BCa nem loi HOAC can khong huu han -> percentile (ghi phuong phap); chenh hang -> CI = TB.
  Bao kem: KTN-R - ct_R (eldBETA), D-R (ngau5, tatca, oz), tinh_r - tinh, r_rieng, rho (trung vi theo nguoi), luoi alpha.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RA, CU, KR = sys.argv[1:4]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
SO = {"benchmark": 35, "beta": 70, "eld": 100}
KTN = {("benchmark", "1.0"): "lai/benchmark_b1moi_lai_T10_seed20261107_20261114.json", ("benchmark", "0.6"): "lai/benchmark_b1moi_lai_T06_seed20261107_20261114.json",
       ("beta", "1.0"): "lai_beta/beta_b1moi_lai-xn_T10_seed20261107_20261114.json", ("beta", "0.6"): "lai_beta/beta_b1moi_lai-xn_T06_seed20261107_20261114.json",
       ("eld", "1.0"): "lai_eld/eldbeta_b1moi_lai-xn_T10_seed20261107_20261114.json", ("eld", "0.6"): "lai_eld/eldbeta_b1moi_lai-xn_T06_seed20261107_20261114.json"}
TEN = {"benchmark": "benchmark", "beta": "beta", "eld": "eldbeta"}


def ci(a):
    a = 100 * np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()), phuong_phap="hang")
    pp = "BCa"
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError("khong huu han")
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    if not (np.isfinite(r.low) and np.isfinite(r.high)):
        raise SystemExit("CI khong huu han")
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
out = dict(seed=SEED, bo={}); sha = set()
for ds in ("benchmark", "beta", "eld"):
    out["bo"][ds] = {}
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        fs_ = glob.glob(os.path.join(KR, "%s_ktnr-buoc1_%s_seed%d_*.json" % (TEN[ds], tt, SEED)))
        assert len(fs_) == 1, (ds, T, fs_)
        J = json.load(open(fs_[0], encoding="utf-8")); P = J["tung_nguoi"]
        assert J["ds"] == ds and abs(J["T"] - float(T)) < 1e-9 and J["seed"] == SEED and J["alpha"] == 0.5 and J["H"] == 5 and J["k"] == 5 and J["n0"] == 40 and J["nch"] == 22
        assert len(P) == SO[ds] and [p["fold"] for p in P] == list(range(1, SO[ds] + 1))
        sha.add((J["script_sha256"], json.dumps(J["ma_phu_sha256"], sort_keys=True)))                  # ca ma phu
        kj = json.load(open(os.path.join(CU, KTN[(ds, T)]), encoding="utf-8")); kp = kj["tung_nguoi"]
        assert abs(kj["T"] - float(T)) < 1e-9 and kj["seed"] == SEED and kj["n0"] == 40 and len(kp) == SO[ds]
        assert [p["fold"] for p in kp] == list(range(1, SO[ds] + 1)) and all(p["seed"] == q["seed"] for p, q in zip(kp, P))
        sf = np.array([p["B"]["SF"]["sau"] for p in kp], float); ktn = np.array([p["ktn"] for p in P], float)
        if not (np.isfinite(sf).all() and np.isfinite(ktn).all() and (sf >= 0).all() and (sf <= 1).all() and (ktn >= 0).all() and (ktn <= 1).all()):
            print("CONG HOI QUY: SF hoac KTN khong hop le", ds, T); sys.exit(4)
        lech = int((np.abs(sf - ktn) > 1e-9).sum())
        lech_nguon = sum(p["B"]["nguon_chon"] != q["nguon_chon"] for p, q in zip(kp, P))
        if lech or lech_nguon:
            print("CONG HOI QUY KHONG DAT", ds, T, "lech", lech, "nguoi; lech nguon", lech_nguon); sys.exit(4)
        x = lambda k: np.array([p[k] for p in P], float)
        for k_ in ("ktnr", "ktnr_a025", "ktnr_a1", "ktnr_ngau5", "ktnr_tatca", "ktnr_oz", "tinh", "tinh_r", "r_rieng"):
            v_ = x(k_); assert np.isfinite(v_).all() and (v_ >= 0).all() and (v_ <= 1).all(), (ds, T, k_)
        for p in P:                                                                                      # tinh lai rho va ktnr tu diem z da luu
            zk_, zR_, yy = np.array(p["luot"]["z_ktn"]), np.array(p["luot"]["z_R"]), np.array(p["luot"]["y"])
            assert p["luot"]["chi_so"][0] == 41 and len(yy) == len(p["luot"]["chi_so"])
            assert abs(float(((zk_ + 0.5 * zR_).argmax(1) == yy).mean()) - p["ktnr"]) < 0.02, (ds, T, p["fold"])   # sai so do lam tron 6 chu so
        e = dict(tb={k: float(100 * x(k).mean()) for k in ("ktn", "ktnr", "ktnr_a025", "ktnr_a1", "ktnr_ngau5", "ktnr_tatca", "ktnr_oz", "tinh", "tinh_r", "r_rieng")})
        e["KTNR_tru_KTN"] = ci(x("ktnr") - x("ktn"))
        e["KTNR_tru_ngau5"] = ci(x("ktnr") - x("ktnr_ngau5")); e["KTNR_tru_tatca"] = ci(x("ktnr") - x("ktnr_tatca")); e["KTNR_tru_oz"] = ci(x("ktnr") - x("ktnr_oz"))
        e["tinhR_tru_tinh"] = ci(x("tinh_r") - x("tinh"))
        rh = [p["rho"] for p in P if p["rho"] is not None]; e["rho_trung_vi"] = float(np.median(rh)); e["rho_so_nguoi"] = len(rh)
        if ds == "eld":
            ctj = json.load(open(os.path.join(CU, "ct/eldbeta_b1moi_ct_seed20261107_20261114.json"), encoding="utf-8"))
            assert ctj["ds"] == "eld" and ctj["n_subjects"] == 100 and ctj["seed"] == SEED      # tep ct khong co ma nguoi; thu tu = for u in range(nsub) (b1moi_ct_eld.py dong 84)
            ct = ctj["ket_qua"][T]["tung_nguoi"]
            assert len(ct) == 100 and all("ct_R" in p["40"] for p in ct)
            e["KTNR_tru_ctR"] = ci(x("ktnr") - np.array([p["40"]["ct_R"] for p in ct]))
        out["bo"][ds][T] = e
        print(ds, T, {k: round(v, 1) for k, v in e["tb"].items()}, "rho %.2f" % e["rho_trung_vi"])
        print("    ", " | ".join("%s %s" % (k, f(v)) for k, v in e.items() if isinstance(v, dict) and "thap" in v))
assert len(sha) == 1, "tron phien ban ma"
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
