#!/usr/bin/env python
"""Phan tich do nhay thu tu luot cua KTN-R.
Dung: python ktnr_hoanvi_phan_tich.py <json ra> <thu muc chua *_ktnr-hoanvi_T10/T06_*.json>
KIEM DAU VAO (sai -> thoat, khong ghi ket qua): du 8 tep (4 bo x 2 cua so), moi tep duy nhat; thu = False; so fold = so nguoi du kien (35/70/100/59);
seed, n0, K, alpha; moi nguoi: B = 6/4/4 (benchmark/beta/dong), khong B o eld; T = 10; P1 = 3 o bo 40 lop; do chinh xac trong [0, 1].
Dai luong: Delta_i = TB qua bien the cua (KTN-R - mang tinh), tuong tu KTN-R - KTN; CI 95 % BCa qua nguoi, 10 000 lan, seed 20261107.
Q = TB qua bien the - gia tri thu tu tep. Uoc luong chinh: ho B, 1,0 s (BCa); percentile chi bao kem.
Bao kem: tinh_r; do chinh xac giai doan khoi tao; bang tan so so lop co mat trong 40 luot dau (ho T); phan tich theo khoi cham chung.
Ban 3: kiem du ma bien the, hoan vi dung lai tu seed,
tap luot cham, do dai quyet dinh, nhan that y_cham, seed tung fold, thiet bi cuda, SHA dong nhat giua 8 tep.
"""
import hashlib
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RA, TM = sys.argv[1:3]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
TEN = {"benchmark": "benchmark", "beta": "beta", "eld": "eldbeta", "dong": "dong"}
SO_NGUOI = {"benchmark": 35, "beta": 70, "eld": 100, "dong": 59}
SO_B = {"benchmark": 6, "beta": 4, "dong": 4, "eld": 0}
NT_DU = {"benchmark": 40, "beta": 40, "dong": 40, "eld": 9}
SO_LUOT = {"benchmark": 240, "beta": 160, "dong": 160, "eld": 63}
PP = ("ktn", "ktnr", "tinh", "tinh_r")


def ci(a):
    a = 100 * np.asarray(a, float)
    out = dict(tb=float(a.mean()))
    if np.allclose(a, a[0]):
        out.update(thap=float(a.mean()), cao=float(a.mean()), phuong_phap="hang"); return out
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        ok = np.isfinite(r.low) and np.isfinite(r.high)
    except (ValueError, FloatingPointError, RuntimeWarning):
        ok = False
    if ok:
        out.update(thap=float(r.low), cao=float(r.high), phuong_phap="BCa"); return out
    r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval
    out.update(thap=float(r.low), cao=float(r.high), phuong_phap="percentile (chi bao kem)"); return out


def trang_thai(c):
    if c["phuong_phap"] != "BCa":
        return "khong ket luan (BCa khong xac dinh)"
    return "ben" if c["thap"] > 0 else ("nguoc chieu" if c["cao"] < 0 else "chua phan dinh")


def loi(msg):
    raise SystemExit("DAU VAO KHONG HOP LE: " + msg)


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
# ---------- kiem du dau vao truoc khi tinh bat cu gi ----------
TEPS = {}
for ds in TEN:
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        fs_ = [x for x in glob.glob(os.path.join(TM, "%s_ktnr-hoanvi_%s_seed%d_*.json" % (TEN[ds], tt, SEED))) if not x.endswith(".tmp")]
        if len(fs_) != 1:
            loi("%s %s: %d tep" % (ds, T, len(fs_)))
        J = json.load(open(fs_[0], encoding="utf-8")); P = J["tung_nguoi"]; n = SO_NGUOI[ds]
        if J.get("thu", True) is not False or J.get("thiet_bi") != "cuda" or J["so_fold_chay"] != n or J["so_nguoi_du_kien"] != n or len(P) != n \
                or [p["fold"] for p in P] != list(range(1, n + 1)):
            loi("%s %s: so fold/thu/thiet bi" % (ds, T))
        if J["ds"] != ds or abs(J["T"] - float(T)) > 1e-9 or J["seed"] != SEED or J["n0"] != 40 or J["k"] != 5 \
                or J["alpha"] != 0.5 or J["NT"] != NT_DU[ds] or J["R_T"] != 10 or J["R_P1"] != 3:
            loi("%s %s: metadata" % (ds, T))
        NT, nl, nb = NT_DU[ds], SO_LUOT[ds], SO_B[ds]
        for p in P:
            i = p["fold"]
            if p["seed"] != SEED + 1000 * i + int(round(float(T) * 10)) or p.get("thiet_bi") != "cuda":
                loi("%s %s fold %d: seed/thiet bi" % (ds, T, i))
            if [v["khoi_khoi_tao"] for v in p.get("B", [])] != list(range(1, nb + 1)) or [v["r"] for v in p["T"]] != list(range(1, 11)) \
                    or [v["r"] for v in p.get("P1", [])] != (list(range(1, 4)) if nb else []):
                loi("%s %s fold %d: ma bien the" % (ds, T, i))
            for r_, v in enumerate(p["T"], 1):                            # dung lai hoan vi tu seed, kiem hash day du
                pi = np.random.default_rng([SEED, i, r_]).permutation(nl)
                if v["pi_sha256"] != hashlib.sha256(pi.astype(np.int64).tobytes()).hexdigest() or v["chi_so_cham"] != pi[40:].tolist():
                    loi("%s %s fold %d T r%d: hoan vi khac seed" % (ds, T, i, r_))
            for r_, v in enumerate(p.get("P1", []), 1):
                rg = np.random.default_rng([SEED, i, r_, 1]); pi = np.concatenate([b * NT + rg.permutation(NT) for b in range(nb)])
                if v["pi_sha256"] != hashlib.sha256(pi.astype(np.int64).tobytes()).hexdigest():
                    loi("%s %s fold %d P1 r%d: hoan vi khac seed" % (ds, T, i, r_))
            if p["goc"]["chi_so_cham"] != list(range(40, nl)):
                loi("%s %s fold %d: tap cham goc" % (ds, T, i))
            for v in p.get("B", []):
                b = v["khoi_khoi_tao"] - 1
                if sorted(v["chi_so_cham"]) != [j for j in range(nl) if not (b * NT <= j < (b + 1) * NT)]:
                    loi("%s %s fold %d B%d: tap cham" % (ds, T, i, b + 1))
            for v in [p["goc"]] + p.get("B", []) + p["T"]:
                m = len(v["chi_so_cham"])
                if m != nl - 40 or len(set(v["chi_so_cham"])) != m or len(v["y_cham"]) != m or any(len(v["qd_" + k]) != m for k in PP):
                    loi("%s %s fold %d: do dai luot cham" % (ds, T, i))
                yy = np.asarray(v["y_cham"])
                for k in PP:
                    if not (np.isfinite(v[k]) and abs(v[k] - float((np.asarray(v["qd_" + k]) == yy).mean())) < 1e-12):
                        loi("%s %s fold %d: do chinh xac %s khong khop quyet dinh" % (ds, T, i, k))
                if not (0 <= v["khoi_tao_mang_nen"] <= 1):
                    loi("%s %s fold %d: khoi tao ngoai [0,1]" % (ds, T, i))
        TEPS[(ds, T)] = (fs_[0], J)
for khoa in ("script_sha256", "ktnr_buoc1_sha256"):
    if len({J[khoa] for _, J in TEPS.values()}) != 1:
        loi("8 tep khong cung %s" % khoa)
# ---------- tinh ----------
J0 = next(iter(TEPS.values()))[1]
out = dict(seed=SEED, hoanvi_sha256=J0["script_sha256"],
           phan_tich_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           json_goc_script_trung={"%s_%s" % k: v[1]["json_goc_script_trung_ktnr_buoc1_hien_tai"] for k, v in TEPS.items()},
           tep={"%s_%s" % k: v[0] for k, v in TEPS.items()}, bo={})
for ds in TEN:
    out["bo"][ds] = {}
    for T in ("1.0", "0.6"):
        J = TEPS[(ds, T)][1]; P = J["tung_nguoi"]
        goc = lambda k: np.array([p["goc"][k] for p in P])
        e = dict(so_nguoi=len(P), luot_sat_hoa_goc={k: int(sum(p["hoi_quy_luot_sat_hoa_goc"][k] for p in P)) for k in ("ktn", "ktnr")},
                 lech_z_max={k: float(max(p["hoi_quy_lech_diem_max"][k] for p in P)) for k in ("z_ktn", "z_R")}, co=[c for p in P for c in p["co"]],
                 goc=dict(tb={k: 100 * float(goc(k).mean()) for k in PP}, KTNR_tru_tinh=ci(goc("ktnr") - goc("tinh")),
                          KTNR_tru_KTN=ci(goc("ktnr") - goc("ktn")), khoi_tao_mang_nen=100 * float(goc("khoi_tao_mang_nen").mean())))
        for ho in ("B", "T"):
            if ho == "B" and not SO_B[ds]:
                continue
            m = lambda k: np.array([np.mean([v[k] for v in p[ho]]) for p in P])
            d_tinh, d_ktn = m("ktnr") - m("tinh"), m("ktnr") - m("ktn")
            e[ho] = dict(so_bien_the=len(P[0][ho]), tb={k: 100 * float(m(k).mean()) for k in PP}, KTNR_tru_tinh=ci(d_tinh), KTNR_tru_KTN=ci(d_ktn),
                         Q_KTNR_tru_tinh=ci(d_tinh - (goc("ktnr") - goc("tinh"))), Q_KTNR_tru_KTN=ci(d_ktn - (goc("ktnr") - goc("ktn"))),
                         khoi_tao_mang_nen=100 * float(m("khoi_tao_mang_nen").mean()),
                         ty_le_tap_nguon_trung_goc=float(np.mean([np.mean([sorted(v["nguon_chon_chi_so"]) == sorted(p["goc"]["nguon_chon_chi_so"]) for v in p[ho]]) for p in P])))
        so_lop = [v["so_lop_40_dau"] for p in P for v in p["T"]]
        e["T"]["tan_so_so_lop_40_dau"] = {str(k): int(c) for k, c in zip(*np.unique(so_lop, return_counts=True))}
        mn = [min(v["luot_moi_lop_40_dau"]) for p in P for v in p["T"]]; mx = [max(v["luot_moi_lop_40_dau"]) for p in P for v in p["T"]]
        e["T"]["tan_so_luot_it_nhat_moi_lop"] = {str(k): int(c) for k, c in zip(*np.unique(mn, return_counts=True))}
        e["T"]["tan_so_luot_nhieu_nhat_moi_lop"] = {str(k): int(c) for k, c in zip(*np.unique(mx, return_counts=True))}
        if SO_B[ds]:                                                         # theo khoi cham chung
            nb, NT = SO_B[ds], J["NT"]; cap = np.full((len(P), nb, nb), np.nan); sd = []
            for ip, p in enumerate(P):
                for v in p["B"]:
                    b = v["khoi_khoi_tao"] - 1; idx = np.asarray(v["chi_so_cham"]); kh = idx // NT
                    for c in range(nb):
                        if c == b:
                            continue
                        sel = kh == c
                        qd_r = np.asarray(v["qd_ktnr"])[sel]; qd_t = np.asarray(v["qd_tinh"])[sel]; yy = np.asarray(v["y_cham"])[sel]
                        cap[ip, b, c] = 100 * ((qd_r == yy).mean() - (qd_t == yy).mean())
                for c in range(nb):
                    sd.append(np.nanstd(cap[ip, :, c][~np.isnan(cap[ip, :, c])], ddof=1))
            e["B"]["theo_khoi_cham"] = dict(KTNR_tru_tinh_khoi_tao_x_khoi_cham=np.round(np.nanmean(cap, 0), 3).tolist(),
                                            sd_qua_khoi_khoi_tao_tb=float(np.mean(sd)), ghi_chu="bao kem; hang = khoi khoi tao, cot = khoi cham; nhan that y_cham")
        if T == "1.0":
            if SO_B[ds]:
                e["tieu_chi"] = dict(ben_khi_khoi_tao_bang_khoi_khac=trang_thai(e["B"]["KTNR_tru_tinh"]), nhanh_R_khi_khoi_tao_bang_khoi_khac=trang_thai(e["B"]["KTNR_tru_KTN"]),
                                     pham_vi="ho B doi dong thoi khoi khoi tao va tap luot cham; khong phai tac dong rieng cua khoi khoi tao")
        out["bo"][ds][T] = e
        print(ds, T, "goc", f(e["goc"]["KTNR_tru_tinh"]), "|", " | ".join("%s tinh %s ktn %s Q %s" % (h, f(e[h]["KTNR_tru_tinh"]), f(e[h]["KTNR_tru_KTN"]), f(e[h]["Q_KTNR_tru_tinh"]))
                                                                   for h in ("B", "T") if h in e), e.get("tieu_chi", ""))
json.dump(out, open(RA + ".tmp", "w", encoding="utf-8"), indent=1); os.replace(RA + ".tmp", RA)
