#!/usr/bin/env python
"""Doi chung LUAT CHON NGUON cho CSS-R.
Lap lai dung vong mo-dun nguon cua ktnr_buoc1.chay_fold (cung seed, cung thu tu goi RNG; khong sua ktnr_buoc1.py), roi CHI doi luat chon
K = 5 nguon. Moi luat dung chung: mo-dun gop = TB tham so 5 nguon chon; w_R = vecto rieng lon nhat cua TB w_s w_s^T cua 5 nguon chon;
s = z(logit) + 0,5 z(R); cham luot 41..het. Khong doi K, n0, ALPHA, H.
Luat:
  do_tin    : luat cua bai (TB do tin cua logit tung mo-dun nguon tren 40 luot dau, khong nhan). Phai trung ktnr_buoc1 (cong hoi quy).
  giong_mau : diem giong kieu Ensemble-DNN (guney2023transfer), tren dac trung u = l1(x): voi nguon s va luot i < 40, lop du doan
              p = argmax logit cua mo-dun s; W_s = cac hang cua l2 + dW_s (22 bo loc khong gian); T_s[p] = TB u cua nguoi s o lop p;
              diem = max qua bo loc [corr^2(W T_s[p], W u_i) + R^2 hoi quy cua W u_i len tham chieu sin/cos 5 hai (da tru TB) cua lop p];
              xep nguon theo TB diem tren 40 luot dau. Khong nhan cua nguoi dung.
  co_dinh   : tap co dinh khong phu thuoc nguoi dung: do chinh xac CO NHAN cua mo-dun s tren moi nguoi nguon khac (bo s), TB qua nguoi;
              5 nguon cao nhat. Chi dung nhan cua nguoi nguon.
  tran      : TRAN TREN (dung nhan cua nguoi dung, KHONG phai phuong phap): do chinh xac co nhan cua mo-dun s tren 40 luot dau cua nguoi dung.
Ghi them do chinh xac cua tung mo-dun nguon tren luot 41..het (de biet khoang cach tran duoi cua K = 5) chi o dang tom tat.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ktnr_buoc1 as K        # noqa: E402
import b1_diem as B           # noqa: E402
import global_dnn as GD       # noqa: E402
import b1moi_lai as L         # noqa: E402
import du_lieu_eld as DE      # noqa: E402
import du_lieu_nak as DN      # noqa: E402
import du_lieu_dong as DG     # noqa: E402

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
LUAT = ("do_tin", "giong_mau", "co_dinh", "tran")


def tham_chieu_tru_tb(freqs, fs, ns, dev):
    t = np.arange(ns) / fs
    Q = []
    for f in freqs:
        Y = np.stack([fn(2 * np.pi * h * f * t) for h in range(1, K.H + 1) for fn in (np.sin, np.cos)], 1)
        Q.append(np.linalg.qr(Y - Y.mean(0))[0])
    return torch.tensor(np.stack(Q), dtype=torch.float32, device=dev)


def diem_giong(model, mod, h1s, ys, h1u, pred, Qr, NT):
    """h1s (n_s,1,C,ns) nguon; h1u (n,1,C,ns) 40 luot dau nguoi dung; pred (n,) du doan cua mo-dun nguon. Tra (n,)."""
    W = (model.l2.weight + mod["dW"])[:, 0, :, 0]                               # (22, C)
    mau = torch.stack([h1s[ys == c, 0].mean(0) for c in range(NT)])           # (NT, C, ns)
    zt = torch.einsum("kc,ict->ikt", W, mau[pred]); zx = torch.einsum("kc,ict->ikt", W, h1u[:, 0])
    zt = zt - zt.mean(-1, keepdim=True); zx = zx - zx.mean(-1, keepdim=True)
    r = (zt * zx).sum(-1) / torch.sqrt((zt * zt).sum(-1) * (zx * zx).sum(-1) + 1e-30)
    pr = torch.einsum("ikt,itq->ikq", zx, Qr[pred])
    cca2 = (pr ** 2).sum(-1) / ((zx ** 2).sum(-1) + 1e-30)
    return (r ** 2 + cca2).max(1).values


def chay_fold(i, D, a, dev, freqs, fs, NT):
    seed = K.SEED + 1000 * (i + 1) + int(round(a.T * 10))
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=L.NCH, p_cuoi=0.95).to(dev)
    ck = os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev); y = D["y"][i]
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():
        h1u = L.truoc(model, Xu)
    mods, wsrc, H1, YS = [], [], [], []
    for s in src:                                                        # = ktnr_buoc1.chay_fold dong 102-108 (cung thu tu goi RNG)
        Xs = torch.tensor(D["X"][s], device=dev); ys = torch.tensor(D["y"][s], device=dev, dtype=torch.long)
        with torch.no_grad():
            h1s = L.truoc(model, Xs)
        mods.append(L.hoc(model, h1s, ys, mc, L.BUOC_NGUON, L.LR_MOD))
        Us = h1s[:, 0].double().cpu().numpy(); Us = Us - Us.mean(-1, keepdims=True)
        wsrc.append(K.trca(Us, D["y"][s], NT))
        H1.append(h1s); YS.append(ys)
    N0 = K.N0
    yu = torch.tensor(y, device=dev, dtype=torch.long)
    with torch.no_grad():
        lgs = torch.stack([L.sau(model, h1u, m) for m in mods])                # (S, n, NT)
        diem = {"do_tin": L.do_tin(lgs[:, :N0]).mean(1)}
        Qr = tham_chieu_tru_tb(freqs, fs, ns, dev)
        diem["giong_mau"] = torch.stack([diem_giong(model, mods[j], H1[j], YS[j], h1u[:N0], lgs[j, :N0].argmax(1), Qr, NT).mean()
                                         for j in range(len(src))])
        Hc = torch.cat(H1); Yc = torch.cat(YS); ai = torch.cat([torch.full((len(h),), j, device=dev) for j, h in enumerate(H1)])
        acc_ng = torch.zeros(len(src), len(src), device=dev)
        for j in range(len(src)):
            dung = (L.sau(model, Hc, mods[j]).argmax(1) == Yc).float()
            acc_ng[j] = torch.zeros(len(src), device=dev).index_add_(0, ai, dung) / torch.bincount(ai, minlength=len(src)).float()
        acc_ng.fill_diagonal_(float("nan"))
        diem["co_dinh"] = torch.nanmean(acc_ng, 1)
        diem["tran"] = (lgs[:, :N0].argmax(2) == yu[None, :N0]).float().mean(1) + 1e-6 * diem["do_tin"]   # hoa: do tin pha
        acc_sau = (lgs[:, N0:].argmax(2) == yu[None, N0:]).float().mean(1)
    U = h1u[N0:, 0].double().cpu().numpy(); U = U - U.mean(-1, keepdims=True)
    Qb = K.co_so(freqs, fs, ns); yt = y[N0:]
    hit = lambda s_: float((s_.argmax(1) == yt).mean())
    e = {"fold": i + 1, "seed": seed, "trongso_sha256": sha(ck)}
    for lu in LUAT:
        top = torch.argsort(-diem[lu], stable=True)[:K.K]
        tch = [int(j) for j in top.cpu().numpy()]
        ms = {k: torch.stack([mods[j][k] for j in tch]).mean(0) for k in mc}
        with torch.no_grad():
            lk_t = torch.cat([L.sau(model, h1u[a_:a_ + L.LO], {k: v.clone() for k, v in ms.items()}) for a_ in range(N0, h1u.shape[0], L.LO)])
        lk = lk_t.double().cpu().numpy()
        R5 = K.diem_R(np.einsum("c,ict->it", K.w_tu([wsrc[j] for j in tch]), U), Qb)
        e[lu] = dict(nguon_chon=[src[j] + 1 for j in tch], ktn=float((lk_t.argmax(1).cpu().numpy() == yt).mean()),
                     ktnr=hit(K.z(lk) + K.ALPHA * K.z(R5)), acc_sau_tb_5_nguon=float(acc_sau[top].mean()))
    od = torch.argsort(-acc_sau, stable=True)
    e["mo_dun_nguon_sau"] = dict(tb=float(acc_sau.mean()), top5_tb=float(acc_sau[od[:5]].mean()), thap5_tb=float(acc_sau[od[-5:]].mean()))
    e["trung_tap"] = {f"{a_}_{b_}": len(set(e[a_]["nguon_chon"]) & set(e[b_]["nguon_chon"])) for a_ in LUAT for b_ in LUAT if a_ < b_}
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "eld", "dong"])
    ap.add_argument("--T", type=float, required=True); ap.add_argument("--trongso", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=0); ap.add_argument("--goc", default="", help="JSON ktnr_buoc1 de kiem hoi quy luat do_tin")
    ap.add_argument("--cho-phep-cpu", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    ds = a.ds; DE.dang_ky(B, GD, ds); DN.dang_ky(B, GD, ds); DG.dang_ky(B, GD, ds); NT = GD.NT
    nl = {**DE.SO_LUOT, **DN.SO_LUOT, **DG.SO_LUOT}[ds]; CK = {**DN.SO_KENH, **DG.SO_KENH}.get(ds, 9)
    c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    D = {"X": [], "y": []}; freqs = None
    for s in range(1, nsub + 1):                                         # = ktnr_buoc1.main dong 186-193
        X, y, fr = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and X.shape[1] == CK and np.isfinite(X).all()
        freqs = np.asarray(fr, float) if freqs is None else freqs
        Xp = GD.tien_xu_ly(X, ds, "causal", ns, fs); assert np.isfinite(Xp).all()
        D["X"].append(Xp); D["y"].append(y)
    goc = {r["fold"]: r for r in json.load(open(a.goc))["tung_nguoi"]} if a.goc else {}
    t0 = time.time(); per = []; lech = 0
    for i in range(a.folds or nsub):
        r = chay_fold(i, D, a, dev, freqs, fs, NT)
        if goc:
            g = goc[i + 1]
            r["hoi_quy"] = ok = (r["do_tin"]["nguon_chon"] == g["nguon_chon"] and abs(r["do_tin"]["ktn"] - g["ktn"]) < 1e-9
                                 and abs(r["do_tin"]["ktnr"] - g["ktnr"]) < 1e-9)
            lech += not ok
        per.append(r)
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"fold {i+1}: hoi_quy {r.get('hoi_quy')} | {time.time()-t0:.0f} s", flush=True)
    js = dict(seed=K.SEED, ds=ds, T=a.T, k=K.K, n0=K.N0, alpha=K.ALPHA, luat=LUAT, trongso=a.trongso, goc=a.goc, so_fold_lech_hoi_quy=lech,
              script_sha256=sha(os.path.abspath(__file__)), ktnr_buoc1_sha256=sha(K.__file__), thiet_bi=str(dev), tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    print("xong", time.time() - t0, "lech hoi quy", lech)


if __name__ == "__main__":
    main()
