#!/usr/bin/env python
"""BUOC 2 KTN-R (2026-11-16): doi chung kh_deu, ct_R, FBCCA tren Nakanishi 2015 (bo xac nhan, chay MOT lan). Sinh tu b1moi_ct_eld.py bang
tao_ct_nak (scratchpad), CHI doi bo nap va hang so theo bo: 8 kenh, 12 lop, fs 256, onset 38, do tre 0,135 s, 180 luot, 10 nguoi (S = 9 nguon,
nen K = min(11, S) = 9: kh_deu dung MOI nguon, khong con buoc chon). Bo nap giong du_lieu_nak.py. Phep tinh con lai giu nguyen.

KIEM DOC LAP tren eldBETA (b1moi_eld.py): cau hinh buoc 0b giu nguyen.
eldBETA: data.EEG.Epoch (64 kenh, 1500 mau, 9 muc tieu, 7 khoi), fs 250, 0,5 s truoc kich thich, do tre 0,14 s, 9 kenh CH9,
da loc chan 50 Hz phan cung (kiem pho: eldbeta_kiem_du_lieu.py) nen KHONG loc chan them. 100 nguoi, 63 luot moi nguoi.

Them: loc chan 50 Hz nhan qua (iirnotch Q = 30, lfilter) tren du lieu tho sau khi tru TB truoc kich thich; doi chung FBCCA (5 dai, 5 hai, FBW).

--dc 0|1 chon chi so dien cuc. --im-lang: khong in so (chi thu chay thong).

BUOC 0b (b1moi_buoc0b.py): KET HOP hai thong ke chon nguon (kh_deu, kh_tc) va so dai khoi dong theo tran MAC.
Phan con lai nhu buoc 0.

B1 moi, BUOC 0: phat lai tuan tu, moi pha <= 1 M MAC moi quyet dinh.
LOSO mot muc, du lieu phat trien.

Thu tu luot = thu tu trong tep (khoi roi muc tieu): thu tu tep, khong phai thu tu ghi that.
Moi luot: giai bang cau hinh hien co -> ghi quyet dinh -> dua vao bo tich luy. Chon nguon MOT LAN sau n0 luot, roi co dinh.
Khoi dong (luot 1..n0):
  b2 : moi nguon, dai 1-2 (FBW), TB deu.         bd : Global DNN thu nho tinh (logit lo 2; chi T co logit).
Chon nguon sau n0 (chi dung dai luong da tinh trong pha khoi dong; nhan gia = quyet dinh da phat):
  pl  : do giong = tuong quan mau nhan gia cua nguoi dung (cong don phep chieu) voi mau nguon; b2: dai 1-2; bd: dai 1.
  rcs : thong ke tin cay khong nhan tich luy: (max - TB cac diem con lai) / sd cac diem con lai, tren diem khoi dong b2.
Sau n0: K nguon chon, 5 dai, TB deu.  gop: pl_b2 + z-logit DNN tinh.
Doi chung tren cung luot: b2 tinh, dnn tinh, ds_u (moi nguon 5 dai, vuot tran), oracle (nhan that cua n0 luot dau, 5 dai), rand.
Cham: luot n0+1..het ("sau") va ca phien ("phien": n0 luot dau tinh theo bo khoi dong).
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B  # noqa: E402

SEED = 20261107
WINS = [0.6, 0.8, 1.0]
N0S = [40]
K, K5 = 11, 5
R_RAND = 20
NB, NT = B.NB, B.NTGT
REG = 1e-6


def top_gev(S, Q):
    C = S.shape[0]
    ev, V = eigh((S + S.T) / 2, Q + REG * np.trace(Q) / C * np.eye(C))
    return V[:, -1]


def corr_last(a, b):
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-300)


def zrow(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-12)


NAK = "/gpfs0/users/gv_dnton/datasets/ssvep/nakanishi2015"


def main():
    global NT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nak", choices=["nak"])
    ap.add_argument("--im-lang", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logit", default="")
    ap.add_argument("--subjects", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    rng = np.random.default_rng(SEED)
    from scipy.io import loadmat
    F_TEP = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
    NT = 12; CC = 8
    B.DS["nak"] = dict(root=NAK, n=10, fs=256, onset=38, lat=0.135)
    c = B.DS[a.ds]; fs = c["fs"]; sos = B.filterbank(fs)
    nsub = a.subjects or c["n"]
    ns_max = int(round(max(WINS) * fs))
    t0 = time.time()
    P, Y = [], []
    for s in range(1, nsub + 1):
        E = loadmat(os.path.join(NAK, "s%d.mat" % s))["eeg"]
        assert E.shape == (12, 8, 1114, 15) and np.isfinite(E).all()
        X = E.transpose(3, 0, 1, 2).reshape(180, 8, 1114).astype(np.float64)                                  # khoi-chinh, muc tieu 0..11
        y = np.tile(np.arange(12), 15)
        P.append(B.preprocess(X, a.ds, sos, ns_max).astype(np.float32)); Y.append(y)
    S = nsub - 1; Kk, Kk5 = min(K, S), min(K5, S)
    w2 = B.FBW[:2] / B.FBW[:2].sum(); w5 = B.FBW / B.FBW.sum()
    print(f"nap {nsub} nguoi: {time.time()-t0:.0f} s; S = {S}", flush=True)
    res = {}
    for T in WINS:
        ns = int(round(T * fs))
        u1 = ns * (CC + NT + 1)
        nbk = int(min(2, max(1, 1000000 // (u1 * S))))                           # so dai khoi dong trong tran 1 M
        wk = B.FBW[:nbk] / B.FBW[:nbk].sum()
        X = [B.center(p[..., :ns].astype(np.float64)) for p in P]
        M = np.stack([np.stack([x[Y[s] == k].mean(0) for k in range(NT)]) for s, x in enumerate(X)])
        G = np.einsum("snbct,snbdt->sbcd", M, M)
        R = np.stack([np.einsum("nbct,nbdt->bcd", x - M[s][Y[s]], x - M[s][Y[s]]) for s, x in enumerate(X)])
        Ws = np.stack([np.stack([top_gev(G[s, b], R[s, b]) for b in range(NB)]) for s in range(nsub)])
        ts = B.center(np.einsum("sbc,snbct->snbt", Ws, M))
        ld = os.path.join(a.logit, "T%02d" % int(round(T * 10))) if a.logit else ""
        co_dnn = bool(ld) and os.path.isdir(ld)
        Qr = B.refs(F_TEP, fs, ns)
        per = []
        for u in range(nsub):
            src = [s for s in range(nsub) if s != u]
            x = X[u]; y = Y[u]; n = len(y)
            Wsr = Ws[src]; tsr = ts[src]
            zs = B.center(np.einsum("sbc,ibct->isbt", Wsr, x))                        # (n, S, NB, ns)
            rs = np.einsum("isbt,snbt->isnb", zs, tsr) / np.sqrt((zs ** 2).sum(-1)[:, :, None] * (tsr ** 2).sum(-1)[None] + 1e-300)
            s2 = rs[..., :nbk] @ wk; s5 = rs @ w5                                       # (n, S, N)
            sc_b2 = s2.mean(1); sc_u = s5.mean(1)                                     # (n, N)
            if co_dnn:
                zz = np.load(os.path.join(ld, "fold%03d_logit-tinh.npz" % (u + 1)))
                assert np.array_equal(zz["y"], y)
                sc_d = zz["logit"].astype(np.float64)
            # thong ke tin cay kieu RCS tren diem khoi dong b2 theo tung nguon
            srt = np.sort(s2, 2); oth = srt[..., :-1]
            conf = (srt[..., -1] - oth.mean(2)) / (oth.std(2) + 1e-12)                # (n, S)

            def gam_proj(idx, lab, bands, wb):
                cnt = np.bincount(lab, minlength=NT); co = np.where(cnt > 0)[0]
                g = np.zeros(S)
                for k in co:
                    m = B.center(zs[idx][lab == k][:, :, bands].mean(0))              # (S, nb, ns)
                    g += cnt[k] * (corr_last(m, tsr[:, k][:, bands]) @ wb)
                return g / cnt[co].sum()

            d = {}
            fb = np.empty((n, NT))
            for i in range(n):
                rb = np.empty((NT, NB))
                for b in range(NB):
                    qx = np.linalg.qr(x[i, b].T)[0]
                    rb[:, b] = np.linalg.svd(np.einsum("tc,kth->kch", qx, Qr), compute_uv=False)[:, 0] ** 2
                fb[i] = rb @ B.FBW
            for n0 in N0S:
                idx = np.arange(n0); sau = np.arange(n0, n); ys = y[sau]
                hit = lambda sc: float((sc[sau].argmax(1) == ys).mean())
                phien = lambda sc_kd, sc: float((np.concatenate([sc_kd[idx].argmax(1), sc[sau].argmax(1)]) == y).mean())
                e = {"b2": hit(sc_b2), "ds_u": hit(sc_u), "fbcca": hit(fb)}
                lab = sc_b2[idx].argmax(1); e["nhan_gia_dung_b2"] = float((lab == y[idx]).mean())
                g_or = gam_proj(idx, y[idx], slice(0, NB), w5)
                g_pl = gam_proj(idx, lab, slice(0, nbk), wk)
                g_rcs = conf[idx].mean(0)
                sel = lambda g, k: s5[:, np.argsort(-g, kind="stable")[:k]].mean(1)
                sc_pl = sel(g_pl, Kk)
                e["pl_b2"] = hit(sc_pl); e["pl_b2_phien"] = phien(sc_b2, sc_pl)
                e["rcs11"] = hit(sel(g_rcs, Kk)); e["rcs5"] = hit(sel(g_rcs, Kk5)); e["pl_b2_K5"] = hit(sel(g_pl, Kk5))
                e["oracle"] = hit(sel(g_or, Kk))
                e["rand"] = float(np.mean([hit(s5[:, rng.permutation(S)[:Kk]].mean(1)) for _ in range(R_RAND)]))
                e["r_pl"] = float(np.corrcoef(g_pl, g_or)[0, 1]); e["r_rcs"] = float(np.corrcoef(g_rcs, g_or)[0, 1])
                # ket hop: do tin uoc khong nhan bang hai nua (luot le / chan cua n0 luot dau), Spearman-Brown
                ia, ib = idx[0::2], idx[1::2]
                sb = lambda r: max(0.0, 2 * r / (1 + r)) if r > -1 else 0.0
                tc_pl = sb(float(np.corrcoef(gam_proj(ia, lab[0::2], slice(0, nbk), wk), gam_proj(ib, lab[1::2], slice(0, nbk), wk))[0, 1]))
                tc_rc = sb(float(np.corrcoef(conf[ia].mean(0), conf[ib].mean(0))[0, 1]))
                zc = lambda g: (g - g.mean()) / (g.std() + 1e-12)
                wt = lambda r: min(20.0, r / max(1e-6, 1 - r))
                g_deu = zc(g_pl) + zc(g_rcs)
                g_tc = wt(tc_pl) * zc(g_pl) + wt(tc_rc) * zc(g_rcs)
                e["kh_deu"] = hit(sel(g_deu, Kk)); e["kh_tc"] = hit(sel(g_tc, Kk)); e["kh_tc_phien"] = phien(sc_b2, sel(g_tc, Kk))
                # ===== LO CAI TIEN (tao_ct.py) =====
                Cx = x.shape[2]; u0 = ns * (Cx + NT + 1)
                od = np.argsort(-g_deu, kind="stable"); selK = od[:Kk]
                e["mac_kh_deu"] = 5 * Kk * u0
                K16 = int(min(16, S, 1000000 // (5 * u0))); e["K16"] = K16
                e["ct_K16"] = hit(s5[:, od[:K16]].mean(1))
                gw = g_deu[selK] - (g_deu[od[Kk]] if S > Kk else g_deu[selK].min() - 1.0); gw = gw / gw.sum()
                e["ct_W"] = hit(np.einsum("isn,s->in", s5[:, selK], gw))
                LMAX = 3
                cntk = np.bincount(lab, minlength=NT); cok = np.where(cntk > 0)[0]
                mk = np.stack([zs[idx][lab == k][:, :, :nbk].mean(0) for k in cok])        # (nk, S, nbk, ns) mau nhan gia da chieu
                tk_ = np.stack([tsr[:, k, :nbk] for k in cok])                            # (nk, S, nbk, ns)

                def seg(av, bv, l):                                                       # l > 0: nguoi dung tre hon nguon l mau
                    a_ = av[..., l:] if l >= 0 else av[..., :ns + l]
                    b_ = bv[..., :ns - l] if l >= 0 else bv[..., -l:]
                    a_ = a_ - a_.mean(-1, keepdims=True); b_ = b_ - b_.mean(-1, keepdims=True)
                    return (a_ * b_).sum(-1) / np.sqrt((a_ * a_).sum(-1) * (b_ * b_).sum(-1) + 1e-300)

                gl = np.stack([np.einsum("k,ks->s", cntk[cok] / cntk[cok].sum(), seg(mk, tk_, l) @ wk) for l in range(-LMAX, LMAX + 1)])
                lag = gl.argmax(0) - LMAX                                                  # (S,) khong dung nhan that

                def diem_tre(js):
                    return np.stack([seg(zs[:, j][:, None], tsr[j][None], int(lag[j])) @ w5 for j in js], 1)

                s_tre = diem_tre(selK)
                e["ct_TRE"] = hit(s_tre.mean(1)); e["lag_tb_abs"] = float(np.abs(lag[selK]).mean()); e["lag_khac0"] = float((lag[selK] != 0).mean())
                e["ct_TRE_K16"] = hit(diem_tre(od[:K16]).mean(1))
                HR = max([h for h in range(1, 6) if 5 * Kk * u0 + 5 * (Cx * ns + NT * 2 * h * ns) <= 1000000] or [0]); e["HR"] = HR
                if HR >= 1:
                    tt_ = np.arange(ns) / fs
                    Qh = []
                    for f_ in F_TEP:
                        Yh = np.stack([fn(2 * np.pi * m * f_ * tt_) for m in range(1, HR + 1) for fn in (np.sin, np.cos)], 1)
                        Qh.append(np.linalg.qr(Yh - Yh.mean(0))[0])
                    Qh = np.stack(Qh)                                                      # (N, ns, 2 HR)
                    Wm = Wsr[selK] * np.sign(np.einsum("jbc,bc->jb", Wsr[selK], Wsr[selK[0]]) + 1e-300)[:, :, None]
                    zr = np.einsum("bc,ibct->ibt", Wm.mean(0), x)                          # (n, NB, ns)
                    pr = np.einsum("ibt,kth->ibkh", zr, Qh)
                    sR = np.einsum("ibk,b->ik", np.sqrt((pr ** 2).sum(-1) / ((zr ** 2).sum(-1)[..., None] + 1e-300)), w5)
                    e["ct_R"] = hit(s5[:, selK].mean(1) + sR); e["ct_TRE_R"] = hit(s_tre.mean(1) + sR); e["R_rieng"] = hit(sR)
                    e["mac_ct_R"] = 5 * Kk * u0 + 5 * (Cx * ns + NT * 2 * HR * ns)
                if os.environ.get("CT_CHI_R") == "1":  # che do xac nhan: chi xuat ung vien
                    for kx in [kk_ for kk_ in list(e) if kk_.startswith("ct_") and kk_ != "ct_R"] + ["R_rieng"]:
                        e.pop(kx, None)
                e["kh_tc_K5"] = hit(sel(g_tc, Kk5))
                e["tc_pl"] = tc_pl; e["tc_rcs"] = tc_rc; e["r_kh_tc"] = float(np.corrcoef(g_tc, g_or)[0, 1])
                e["b2_phien"] = float((sc_b2.argmax(1) == y).mean())
                if co_dnn:
                    e["dnn"] = hit(sc_d); e["dnn_phien"] = float((sc_d.argmax(1) == y).mean())
                    labd = sc_d[idx].argmax(1); e["nhan_gia_dung_dnn"] = float((labd == y[idx]).mean())
                    g_bd = gam_proj(idx, labd, slice(0, 1), np.ones(1))
                    sc_bd = sel(g_bd, Kk)
                    e["pl_bd"] = hit(sc_bd); e["pl_bd_phien"] = phien(sc_d, sc_bd); e["r_pl_bd"] = float(np.corrcoef(g_bd, g_or)[0, 1])
                    zp, zd = zrow(sc_pl), zrow(sc_d)
                    e["gop"] = hit(zp + zd); e["gop_phien"] = phien(sc_d, zp + zd)
                    # tuong quan sai so: (a) trong luot, qua 39 lop sai; (b) qua luot, le dung - max sai
                    msk = np.ones((len(sau), NT), bool); msk[np.arange(len(sau)), ys] = False
                    a1 = zp[sau][msk].reshape(len(sau), NT - 1); a2 = zd[sau][msk].reshape(len(sau), NT - 1)
                    e["rho_lop_sai"] = float(np.mean(corr_last(B.center(a1), B.center(a2))))
                    l1 = zp[sau][np.arange(len(sau)), ys] - a1.max(1); l2 = zd[sau][np.arange(len(sau)), ys] - a2.max(1)
                    e["rho_le"] = float(np.corrcoef(l1, l2)[0, 1])
                d[str(n0)] = e
            per.append(d)
        tb = {n0: {k: float(np.mean([p[n0][k] for p in per])) for k in per[0][n0]} for n0 in per[0]}
        u50 = u1
        mac = {"so_dai_khoi_dong": nbk, "khoi_dong_b2": u50 * nbk * S, "khoi_dong_bd": None, "sau_K11": u50 * NB * Kk, "sau_K5": u50 * NB * Kk5, "ds_u": u50 * NB * S,
               "cap_nhat_pl_b2_mot_lan": NT * nbk * S * ns, "cap_nhat_rcs_mot_lan": 0}
        if co_dnn:
            n2 = ns // 2; dn = 3 * CC * ns + 22 * CC * ns + 22 * n2 * 22 * 2 + 22 * n2 * 22 * 10 + 22 * n2 * NT
            mac.update(dnn=dn, khoi_dong_bd=dn + ns * CC * S, gop=dn + u50 * NB * Kk)
        res[str(T)] = {"tung_nguoi": per, "tb": tb, "mac": mac}
        if not a.im_lang:
            print(T, json.dumps({n0: {k: round(v, 4) for k, v in tb[n0].items()} for n0 in tb}), json.dumps(mac), f"{time.time()-t0:.0f} s", flush=True)
        else:
            print(T, "chay thong", sorted(tb["40"].keys()), json.dumps(mac), flush=True)
    me = open(os.path.abspath(__file__), "rb").read()
    js = dict(seed=SEED, ds=a.ds, n_subjects=nsub, S=S, K=Kk, K5=Kk5, n0=N0S, windows=WINS, logit=a.logit,
              script_sha256=hashlib.sha256(me).hexdigest(), b1_diem_sha256=hashlib.sha256(open(B.__file__, "rb").read()).hexdigest(),
              ket_qua=res, ghi_chu="BUOC 2 KTN-R tren Nakanishi 2015, chay mot lan; phat lai theo thu tu tep (khoi roi muc tieu), khong phai thu tu ghi that; chon nguon mot lan sau n0 = 40; LOSO; MAC la cau hinh du kien, khong phai chi phi thuc thi cua ma mo phong")
    if not a.im_lang:
        json.dump(js, open(a.out, "w"), indent=1)
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
