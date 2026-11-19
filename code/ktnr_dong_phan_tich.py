#!/usr/bin/env python
"""Phan tich KTN-R tren Dong 2023.
59 nguoi, cham luot 41..160 (120 luot), chenh ghep cap theo nguoi, CI 95 % BCa 10.000 lan, seed 20261107; BCa nem loi hoac can khong huu han -> percentile.
Dung: python ktnr_dong_phan_tich.py <json ra> <ktnr dong T10> <ktnr dong T06> <ct dong json> <thu muc ban sao results/b1moi_ccanet> <json kiem du lieu> <thu muc RCS>
  Dieu kien truoc: kiem anh xa chi so -> tan so DAT (dong_kiem_du_lieu.py).
  C0 (kiem du lieu): mang tinh 1,0 s TB >= 7,5 %.
      va moi fold khong co vecto diem hang trong thanh phan cua 'final'.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
N = 59
COMMIT = "1660d42113d5721c5ec05d5f8b92aeabcd50c31f"
F_DONG = list(8.0 + 0.2 * np.arange(40))
RA, K10, K06, CT, CN, KD, RD = sys.argv[1:8]                                                         # RD: thu muc ket qua RCS
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")


def ci(a):
    a = 100 * np.asarray(a, float)
    assert np.isfinite(a).all() and len(a) == N
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()), phuong_phap="hang")
    pp = "BCa"
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    if not (np.isfinite(r.low) and np.isfinite(r.high)):
        raise SystemExit("CI khong huu han")
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


def nap_k(p, T):
    J = json.load(open(p, encoding="utf-8"))
    assert J["ds"] == "dong" and abs(J["T"] - T) < 1e-9 and J["seed"] == SEED and J["alpha"] == 0.5 and J["H"] == 5 and J["k"] == 5 and J["n0"] == 40
    assert J["NT"] == 40 and J["fs"] == 250 and J["ns"] == int(round(T * 250)) and np.allclose(J["freqs"], F_DONG)
    P = J["tung_nguoi"]; assert [q["fold"] for q in P] == list(range(1, N + 1))
    for q in P:
        for k in ("ktn", "ktnr", "tinh", "r_rieng", "ktn_ngau5"):
            assert np.isfinite(q[k]) and 0 <= q[k] <= 1, (p, q["fold"], k)
        L_ = q["luot"]; assert L_["chi_so"] == list(range(41, 161)) and len(L_["y"]) == 120
        zk, zR = np.array(L_["z_ktn"]), np.array(L_["z_R"])
        assert zk.shape == (120, 40) and zR.shape == (120, 40) and np.isfinite(zk).all() and np.isfinite(zR).all(), (p, q["fold"])
        assert len(q["ktn_ngau5_lan"]) == 20 and np.isfinite(q["ktn_ngau5_lan"]).all() and abs(np.mean(q["ktn_ngau5_lan"]) - q["ktn_ngau5"]) < 1e-12
        assert q["seed"] == SEED + 1000 * q["fold"] + int(round(T * 10)), (p, q["fold"])
        nc = q["nguon_chon"]; assert len(set(nc)) == 5 and all(1 <= v <= N and v != q["fold"] for v in nc), (p, q["fold"])
        assert L_["y"] == list(np.tile(np.arange(40), 3)), (p, q["fold"])
        assert abs(float(((zk + 0.5 * zR).argmax(1) == np.array(L_["y"])).mean()) - q["ktnr"]) < 0.02, (p, q["fold"])   # z lam tron 6 chu so
        assert np.isfinite(q["ktn_ens5"]) and 0 <= q["ktn_ens5"] <= 1
        for k in ("ktn", "ktnr", "tinh", "ktn_ens5"):                               # do chinh xac tren 120 luot: phai la boi cua 1/120
            assert abs(q[k] * 120 - round(q[k] * 120)) < 1e-6, (p, q["fold"], k)
    return J, P


def nap_cn(d, ds, T, che, nn_):
    cfg = json.load(open(os.path.join(d, "config.json")))
    fs_ = 250; ns_ = int(T * fs_) if che == "goc" else int(round(T * fs_))
    st_ = {("benchmark", "goc"): 160, ("beta", "goc"): 157, ("dong", "gt"): 165}[(ds, che)]
    assert cfg["ds"] == ds and abs(cfg["T"] - T) < 1e-9 and cfg["che"] == che and not cfg["du_nhanh"] and cfg["vong"] == 100 and cfg["commit"] == COMMIT, d
    assert cfg["ns"] == ns_ and cfg["rfs"] == fs_ and cfg["bat_dau_cua_so"] == st_ and cfg["seed"] == (41 if che == "goc" else SEED), d
    if ds == "dong":
        assert np.allclose(cfg["list_freqs"], F_DONG) and np.allclose(cfg["list_phase_pi"], 0.5 * (np.arange(40) % 4)) and cfg["NT"] == 40, d
    J = [json.load(open(f_)) for f_ in sorted(glob.glob(os.path.join(d, "fold*.json")))]
    assert [j["fold"] for j in J] == list(range(1, nn_ + 1)), (d, len(J))
    for j in J:
        assert j["script_sha256"] == cfg["script_sha256"] and j["ds"] == ds and j["che"] == che and abs(j["T"] - T) < 1e-9 and j["ns"] == ns_, (d, j["fold"])
        if ds == "dong":                                                         # nhan va so luot
            assert j["y"] == list(np.tile(np.arange(40), 4)) and len(j["du_doan_final"]) == 160, (d, j["fold"])
        assert j["du_nhanh"] is False and j["seed"] == (41 if che == "goc" else SEED + 1000 * j["fold"] + int(round(T * 10))), (d, j["fold"])
        hang = sum(j["luot_vecto_hang"][k] for k in ("CNN", "f1", "f2", "f3"))
        if hang:
            raise AssertionError("fold %d co %d luot vecto diem hang trong thanh phan cua final" % (j["fold"], hang))
        for k in ("final", "CNN", "FBTTCCA_new+CCA_new"):
            for t in ("moi_luot", "sau"):
                assert np.isfinite(j[k][t]) and 0 <= j[k][t] <= 1, (d, j["fold"], k)
    return J


f = lambda c: "%+.2f [%+.2f; %+.2f] (%s)" % (c["tb"], c["thap"], c["cao"], c["phuong_phap"])
kd = json.load(open(KD, encoding="utf-8"))
if not (kd["dat"] and kd["so_tep"] == 59):
    raise SystemExit("kiem anh xa du lieu chua dat: khong phan tich")
J10, P10 = nap_k(K10, 1.0); J06, P06 = nap_k(K06, 0.6)
assert J10["script_sha256"] == J06["script_sha256"] and J10["ma_phu_sha256"] == J06["ma_phu_sha256"]
ctj = json.load(open(CT, encoding="utf-8"))
assert ctj["ds"] == "dong" and ctj["n_subjects"] == N and ctj["seed"] == SEED and ctj["S"] == 58 and ctj["K"] == 11 and ctj["n0"] == [40]
ct = {T: ctj["ket_qua"][T]["tung_nguoi"] for T in ("1.0", "0.6")}
assert all([q["40"]["nguoi"] for q in v] == list(range(1, N + 1)) for v in ct.values())              # ghep theo ID
x = lambda P, k: np.array([q[k] for q in P], float)


def c_(T, k):
    v = np.array([q["40"][k] for q in ct[T]], float)
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), (T, k)
    return v


out = dict(seed=SEED, tb={})
for T, P in (("1.0", P10), ("0.6", P06)):
    out["tb"][T] = dict(tinh=100 * x(P, "tinh").mean(), KTN=100 * x(P, "ktn").mean(), KTN_R=100 * x(P, "ktnr").mean(), R_rieng=100 * x(P, "r_rieng").mean(),
                        KTN_ngau5=100 * x(P, "ktn_ngau5").mean(), KTN_ens5=100 * x(P, "ktn_ens5").mean(), kh_deu=100 * c_(T, "kh_deu").mean(), ct_R=100 * c_(T, "ct_R").mean(), FBCCA=100 * c_(T, "fbcca").mean())
dung0 = np.array([q["tinh"] * 120 for q in P10]); assert np.allclose(dung0, np.round(dung0), atol=1e-6)    # dem nguyen
out["C0"] = dict(tinh_T10=out["tb"]["1.0"]["tinh"], so_luot_dung=int(np.round(dung0).sum()), tong_luot=N * 120, nguong_luot=531,
                 dat=bool(int(np.round(dung0).sum()) >= 531))                                         # 531 / 7080 = 7,5 %
out["bao_rho_trung_vi_T10"] = float(np.median([q["rho"] for q in P10 if q["rho"] is not None]))
out["D1_T10"] = ci(x(P10, "ktnr") - x(P10, "tinh")); out["D1_T06"] = ci(x(P06, "ktnr") - x(P06, "tinh"))
out["D2"] = ci(x(P10, "ktnr") - x(P10, "ktn"))
out["D3"] = ci(x(P10, "ktn") - x(P10, "ktn_ngau5"))
out["D4_T10"] = ci(x(P10, "ktnr") - c_("1.0", "kh_deu")); out["D4_T06"] = ci(x(P06, "ktnr") - c_("0.6", "kh_deu"))
out["bao_KTN_tru_tinh_T10"] = ci(x(P10, "ktn") - x(P10, "tinh")); out["bao_KTNR_tru_ctR_T10"] = ci(x(P10, "ktnr") - c_("1.0", "ct_R"))
try:
    b = 100 * np.mean([j["final"]["moi_luot"] for j in nap_cn(os.path.join(CN, "goc_benchmark_T10"), "benchmark", 1.0, "goc", 35)])
    e = 100 * np.mean([j["final"]["moi_luot"] for j in nap_cn(os.path.join(CN, "goc_beta_T10"), "beta", 1.0, "goc", 70)])
    out["cong_CCA_Net"] = dict(benchmark_T10=float(b), beta_T10=float(e), dat=bool(80.02 <= b <= 88.02 and 68.01 <= e <= 76.01))
except (FileNotFoundError, AssertionError) as ex:
    out["cong_CCA_Net"] = dict(dat=False, loi=str(ex)[:200])
for T, P, tt in (("1.0", P10, "T10"), ("0.6", P06, "T06")):
    try:
        Jc = nap_cn(os.path.join(CN, "gt_dong_%s" % tt), "dong", float(T), "gt", N)
    except (FileNotFoundError, AssertionError) as ex:
        out["CN_loi_" + T] = str(ex)[:200]; continue
    cn = np.array([j["final"]["sau"] for j in Jc])
    out["tb"][T].update(CN_gt=100 * float(cn.mean()), CN_CNN=100 * float(np.mean([j["CNN"]["sau"] for j in Jc])),
                        CN_MFBtt=100 * float(np.mean([j["FBTTCCA_new+CCA_new"]["sau"] for j in Jc])))
    out["D5_" + T] = ci(x(P, "ktnr") - cn)
# bao kem RCS (khong quyet): doc nhu rcs_eld_phan_tich.py; tep nam canh tep ct (thu muc rcs/ cung cap voi ct/)
try:
    r1 = json.load(open(os.path.join(RD, "dong_rcs-tai-hien_seed20261107_20261117.json"), encoding="utf-8"))
    r2 = json.load(open(os.path.join(RD, "dong_rcs-rieng_seed20261107_20261117.json"), encoding="utf-8"))
    for r_ in (r1, r2):
        assert r_["ds"] == "dong" and r_["n_subjects"] == N and r_["nguoi"] == list(range(1, N + 1)) and r_["n0"] == 40 and r_["seed"] == SEED
    for T, P in (("1.0", P10), ("0.6", P06)):
        rc = {v: np.array([q[v]["sau"] for q in r1["ket_qua"][T]["tung_nguoi"]]) for v in ("tran_lo", "tran1R_lo", "tranG_lo", "day_du_lo") if v in r1["ket_qua"][T]["tung_nguoi"][0]}
        rc.update({v: np.array([q[v]["sau"] for q in r2["ket_qua"][T]["tung_nguoi"]]) for v in ("tranK_lo", "tranKR_lo")})
        m1, m2 = r1["ket_qua"][T]["mac"], r2["ket_qua"][T]["mac"]
        mac = dict(tran_lo=m1["tran_quyet_dinh"], tran1R_lo=m1.get("tran1R_quyet_dinh"), tranG_lo=m1.get("tranG_quyet_dinh"), tranK_lo=m2["tranK_quyet_dinh"],
                   tranKR_lo=m2["tranKR_quyet_dinh"], day_du_lo=m1["day_du_quyet_dinh"])
        trong = [v for v in ("tran_lo", "tran1R_lo", "tranG_lo", "tranK_lo", "tranKR_lo") if v in rc and mac[v] is not None and mac[v] <= 1000000]
        manh = max(trong, key=lambda v: rc[v].mean())
        out["tb"][T].update(RCS_day_du=100 * float(rc["day_du_lo"].mean()), RCS_tran_manh_nhat=100 * float(rc[manh].mean()))
        out["bao_RCS_" + T] = dict(manh_nhat=manh, mac=mac, KTNR_tru_RCS_day_du=ci(x(P, "ktnr") - rc["day_du_lo"]), KTNR_tru_RCS_tran=ci(x(P, "ktnr") - rc[manh]))
except (FileNotFoundError, AssertionError, KeyError, json.JSONDecodeError) as ex:
    out["RCS_loi"] = str(ex)[:200]
t10 = out["tb"]["1.0"]; t06 = out["tb"]["0.6"]
for T in ("1.0", "0.6"):
    print(T, {k: round(v, 1) for k, v in out["tb"][T].items()})
print("C0:", out["C0"], "| cong CCA-Net:", out["cong_CCA_Net"])
for k, v in out.items():
    if isinstance(v, dict) and "thap" in v:
        print(k, f(v))
    elif k.startswith("CN_loi") or k == "RCS_loi":
        print(k, v)
    elif k.startswith("bao_RCS_"):
        print(k, v["manh_nhat"], {a: f(b) for a, b in v.items() if isinstance(b, dict) and "thap" in b})
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
