#!/usr/bin/env python
"""Phan tich KTN-R tren Nakanishi 2015 (ban 2).
10 nguoi, cham luot 41..180 (140 luot), chenh ghep cap theo nguoi, CI 95 % BCa 10.000 lan, seed 20261107; BCa nem loi hoac can khong huu han ->
percentile (ghi phuong phap); chenh hang -> CI = TB. Bao kem p hoan vi dau chinh xac (2^10 to hop, hai phia), KHONG dung de quyet.
Dung: python ktnr_buoc2_phan_tich.py <json ra> <ktnr nak T10> <ktnr nak T06> <ct nak json> <dem mac json> <thu muc ban sao results/b1moi_ccanet>
  Kiem du lieu: FBCCA 1,0 s (5 dai, luot 41..het) TB >= 50 %.
  HB3: so MAC tu tep cua ktnr_dem_mac.py (ham quyet dinh hoan chinh) o 40 lop, 1,0 s.
       goc_beta_T10 (Bang I: 84,02 / 72,01 +- 4); kiem tai hien CCA-Net.
  HB4 (bao kem): KTN-R - ct_R, - FBCCA; KTN - CN-gt; 0,6 s; thanh phan CCA-Net (CNN, FBTTCCA_new+CCA_new). RCS chua tai hien tren Nakanishi: ghi "chua co".
"""
import glob, itertools, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
N = 10
COMMIT = "1660d42113d5721c5ec05d5f8b92aeabcd50c31f"
F_NAK = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
RA, K10, K06, CT, MAC, CN = sys.argv[1:7]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")


def p_dau(a):
    m0 = abs(a.mean())
    return float(np.mean([abs((a * np.array(s)).mean()) >= m0 - 1e-12 for s in itertools.product((1, -1), repeat=len(a))]))   # ca khi hang


def ci(a):
    a = 100 * np.asarray(a, float)
    assert np.isfinite(a).all() and len(a) == N
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()), phuong_phap="hang", p_dau=p_dau(a))
    pp = "BCa"
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    if not (np.isfinite(r.low) and np.isfinite(r.high)):
        raise SystemExit("CI khong huu han")
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp, p_dau=p_dau(a))


def nap_k(p, T):
    J = json.load(open(p, encoding="utf-8"))
    assert J["ds"] == "nak" and abs(J["T"] - T) < 1e-9 and J["seed"] == SEED and J["alpha"] == 0.5 and J["H"] == 5 and J["k"] == 5 and J["n0"] == 40
    assert J["NT"] == 12 and J["fs"] == 256 and J["ns"] == int(round(T * 256)) and np.allclose(J["freqs"], F_NAK)
    P = J["tung_nguoi"]; assert [q["fold"] for q in P] == list(range(1, N + 1))
    for q in P:
        for k in ("ktn", "ktnr", "tinh", "r_rieng"):
            assert np.isfinite(q[k]) and 0 <= q[k] <= 1, (p, q["fold"], k)
        L_ = q["luot"]; assert L_["chi_so"] == list(range(41, 181)) and len(L_["y"]) == 140
        zk, zR = np.array(L_["z_ktn"]), np.array(L_["z_R"])
        assert zk.shape == (140, 12) and zR.shape == (140, 12) and np.isfinite(zk).all() and np.isfinite(zR).all(), (p, q["fold"])
    return J, P


def nap_cn(d, ds, T, che, nn_):
    cfg = json.load(open(os.path.join(d, "config.json")))
    assert cfg["ds"] == ds and abs(cfg["T"] - T) < 1e-9 and cfg["che"] == che and not cfg["du_nhanh"] and cfg["vong"] == 100 and cfg["commit"] == COMMIT, d
    if ds == "nak":
        assert cfg["bat_dau_cua_so"] == 73 and cfg["ns"] == int(round(T * 256)) and np.allclose(cfg["list_freqs"], F_NAK) and cfg["NT"] == 12, d
    J = [json.load(open(f_)) for f_ in sorted(glob.glob(os.path.join(d, "fold*.json")))]
    assert [j["fold"] for j in J] == list(range(1, nn_ + 1)), (d, len(J))
    fs_ = {"nak": 256, "benchmark": 250, "beta": 250}[ds]
    ns_ = int(T * fs_) if che == "goc" else int(round(T * fs_))                       # goc: int nhu ma goc; gt: round (shim)
    st_ = {("nak", "goc"): 89, ("benchmark", "goc"): 160, ("beta", "goc"): 157, ("nak", "gt"): 73, ("benchmark", "gt"): 160, ("beta", "gt"): 157}[(ds, che)]
    assert cfg["ns"] == ns_ and cfg["rfs"] == fs_ and cfg["bat_dau_cua_so"] == st_ and cfg["seed"] == (41 if che == "goc" else SEED), d
    for j in J:                                                                     # tung fold khop config
        assert j["script_sha256"] == cfg["script_sha256"] and j["ds"] == ds and j["che"] == che and abs(j["T"] - T) < 1e-9 and j["ns"] == ns_, (d, j["fold"])
        assert j["du_nhanh"] is False, (d, j["fold"])
        assert j["seed"] == (41 if che == "goc" else SEED + 1000 * j["fold"] + int(round(T * 10))), (d, j["fold"])
        hang = sum(j["luot_vecto_hang"][k] for k in ("CNN", "f1", "f2", "f3"))   # vecto hang (NaN sau min-max) trong thanh phan cua 'final'
        if hang:
            raise AssertionError("fold %d co %d luot vecto diem hang trong thanh phan cua final: khong dung de ket luan" % (j["fold"], hang))
        for k in ("final", "CNN", "FBTTCCA_new+CCA_new"):
            for t in ("moi_luot", "sau"):
                assert np.isfinite(j[k][t]) and 0 <= j[k][t] <= 1, (d, j["fold"], k)
    return J


f = lambda c: "%+.2f [%+.2f; %+.2f] (%s, p dau %.4f)" % (c["tb"], c["thap"], c["cao"], c["phuong_phap"], c["p_dau"])
J10, P10 = nap_k(K10, 1.0); J06, P06 = nap_k(K06, 0.6)
assert J10["script_sha256"] == J06["script_sha256"] and J10["ma_phu_sha256"] == J06["ma_phu_sha256"]
ctj = json.load(open(CT, encoding="utf-8"))
assert ctj["ds"] == "nak" and ctj["n_subjects"] == N and ctj["seed"] == SEED and ctj["S"] == 9 and ctj["K"] == 9 and ctj["n0"] == [40]   # thu tu: for u in range(nsub)
ct = {T: ctj["ket_qua"][T]["tung_nguoi"] for T in ("1.0", "0.6")}
assert all(len(v) == N for v in ct.values())
x = lambda P, k: np.array([q[k] for q in P], float)


def c_(T, k):
    v = np.array([q["40"][k] for q in ct[T]], float)
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), (T, k)
    return v


out = dict(seed=SEED, tb={}, RCS="chua co (chua tai hien RCS tren Nakanishi)")
for T, P in (("1.0", P10), ("0.6", P06)):
    out["tb"][T] = dict(KTN=100 * x(P, "ktn").mean(), KTN_R=100 * x(P, "ktnr").mean(), tinh=100 * x(P, "tinh").mean(), R_rieng=100 * x(P, "r_rieng").mean(),
                        kh_deu=100 * c_(T, "kh_deu").mean(), ct_R=100 * c_(T, "ct_R").mean(), FBCCA=100 * c_(T, "fbcca").mean())
out["HB1"] = ci(x(P10, "ktnr") - x(P10, "ktn"))
out["HB2_T10"] = ci(x(P10, "ktnr") - c_("1.0", "kh_deu")); out["HB2_T06"] = ci(x(P06, "ktnr") - c_("0.6", "kh_deu"))
mj = json.load(open(MAC, encoding="utf-8"))
assert mj["HB3"]["gia_tri"] == mj["40lop_9kenh_250Hz_T1.0"]["tong"]
out["HB3"] = mj["HB3"]
for T, P in (("1.0", P10), ("0.6", P06)):
    out["HB4_" + T] = dict(KTNR_tru_ctR=ci(x(P, "ktnr") - c_(T, "ct_R")), KTNR_tru_FBCCA=ci(x(P, "ktnr") - c_(T, "fbcca")))
# cong tai hien CCA-Net (tinh lai)
try:
    b = 100 * np.mean([j["final"]["moi_luot"] for j in nap_cn(os.path.join(CN, "goc_benchmark_T10"), "benchmark", 1.0, "goc", 35)])
    e = 100 * np.mean([j["final"]["moi_luot"] for j in nap_cn(os.path.join(CN, "goc_beta_T10"), "beta", 1.0, "goc", 70)])
    out["cong_CCA_Net"] = dict(benchmark_T10=float(b), beta_T10=float(e), dat=bool(80.02 <= b <= 88.02 and 68.01 <= e <= 76.01))
except (FileNotFoundError, AssertionError) as ex:
    out["cong_CCA_Net"] = dict(dat=False, loi=str(ex)[:200])
# E2 (chi 1,0 s quyet), 0,6 s bao kem rieng
for T, P, tt in (("1.0", P10, "T10"), ("0.6", P06, "T06")):
    try:
        Jc = nap_cn(os.path.join(CN, "gt_nak_%s" % tt), "nak", float(T), "gt", N)
    except (FileNotFoundError, AssertionError) as ex:
        out["CN_loi_" + T] = str(ex)[:200]; continue
    cn = np.array([j["final"]["sau"] for j in Jc])
    out["tb"][T].update(CN_gt=100 * float(cn.mean()), CN_CNN=100 * float(np.mean([j["CNN"]["sau"] for j in Jc])),
                        CN_MFBtt=100 * float(np.mean([j["FBTTCCA_new+CCA_new"]["sau"] for j in Jc])))
    out["CN_vecto_hang_" + T] = {k: int(sum(j["luot_vecto_hang"][k] for j in Jc)) for k in Jc[0]["luot_vecto_hang"]}
    out["E2_" + T] = ci(x(P, "ktnr") - cn); out["KTN_tru_CN_" + T] = ci(x(P, "ktn") - cn)
for T in ("1.0", "0.6"):
    print(T, {k: round(v, 1) for k, v in out["tb"][T].items()})
print("cong CCA-Net:", out["cong_CCA_Net"])
for k, v in out.items():                                                            # chi dinh dang CI, chuoi in nguyen
    if isinstance(v, dict) and "thap" in v:
        print(k, f(v))
    elif k.startswith("HB4_"):
        print(k, {a: f(b_) for a, b_ in v.items()})
    elif k.startswith(("CN_loi", "CN_vecto")):
        print(k, v)
print("HB3", out["HB3"])
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
