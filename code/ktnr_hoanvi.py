#!/usr/bin/env python
"""Do nhay cua KTN-R theo thu tu luot.
 Ban 2.
Dung lai ham cua ktnr_buoc1.py (khong sua tep do). Moi luot chia: kiem cau hinh mang nen nhu ktnr_buoc1.main, nap mang nen (hash phai trung JSON goc),
hoc mo-dun nguon MOT lan dung thu tu goi RNG nhu ktnr_buoc1.chay_fold, roi danh gia cac bien the thu tu luot:
  goc : thu tu tep. CONG HOI QUY (thoat ma 4): ktn, ktnr, tinh, tinh_r trung JSON goc (|lech| <= 1e-12); nguon_chon trung DUNG THU TU;
        quyet dinh KTN va KTN-R tung luot trung argmax cua z_ktn, z_R da luu trong JSON goc (lam tron 6 chu so), tru cac luot sat hoa
        (khoang cach hai diem cao nhat cua diem lam tron < 1e-5) duoc dem va bao rieng.
  P1  : (bo 40 lop) 3 hoan vi TRONG tung khoi, rng([20261107, i, r, 1]). CONG BAT BIEN (thoat ma 5): tap nguon trung goc; quyet dinh tung luot
        cua ktn, ktnr, tinh, tinh_r (theo chi so luot goc) trung goc. Thu tu xep hang nguon khac goc chi ghi co (tong so hang lay trung binh khong phu thuoc thu tu).
  B   : (bo 40 lop) xoay khoi khoi tao; bien the b = 1 phai trung goc tung quyet dinh (thoat ma 5).
  T   : 10 hoan vi toan phien, rng([20261107, i, r]).
Moi bien the luu quyet dinh tung luot theo chi so luot goc (de phan tich theo khoi cham va de cong). Bao kem: do chinh xac giai doan khoi tao
(mang nen, 40 luot dau), so lop co mat va so luot moi lop trong 40 luot dau.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ktnr_buoc1 as K1      # noqa: E402
B, GD, L, DE, DN, DG = K1.B, K1.GD, K1.L, K1.DE, K1.DN, K1.DG
SEED, N0, K, ALPHA = K1.SEED, K1.N0, K1.K, K1.ALPHA
R_T, R_P1 = 10, 3
TOL_Z = 1e-3          # diem z ~ O(1); lech so hoc float32 GPU/CPU va lam tron 6 chu so; quyet dinh tung luot moi la cong chinh
PP = ("ktn", "ktnr", "tinh", "tinh_r")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def danh_gia(model, Xu, y, pi, mods, wsrc, mc, Qb):
    """Chay KTN, KTN-R, mang tinh tren chuoi da hoan vi pi (giong ktnr_buoc1.chay_fold). Quyet dinh tra theo chi so luot goc."""
    Xp = Xu[torch.as_tensor(pi, device=Xu.device)]; yp = y[pi]
    with torch.no_grad():
        h1 = L.truoc(model, Xp); lg0 = model(Xp)
        lgs = torch.stack([L.sau(model, h1, m) for m in mods])
    if not torch.isfinite(lgs).all():
        raise SystemExit("logit nguon khong huu han")                                           # nhu ktnr_buoc1
    g = L.do_tin(lgs[:, :N0]).mean(1)
    if not torch.isfinite(g).all():
        raise SystemExit("do tin khong huu han")
    top = torch.argsort(-g, stable=True)[:K]
    ms = {k: torch.stack([mods[int(j)][k] for j in top]).mean(0) for k in mc}
    n_ = h1.shape[0]; lk_t = []
    with torch.no_grad():
        for a_ in range(N0, n_, L.LO):
            lk_t.append(L.sau(model, h1[a_:a_ + L.LO], {k: v.clone() for k, v in ms.items()}))
    lk_t = torch.cat(lk_t)
    ktn_dec = lk_t.argmax(1).cpu().numpy(); lk = lk_t.double().cpu().numpy()
    l0 = lg0[N0:].double().cpu().numpy(); yt = yp[N0:]
    U = h1[N0:, 0].double().cpu().numpy(); U = U - U.mean(-1, keepdims=True)
    tch = [int(j) for j in top.cpu().numpy()]
    R5 = K1.diem_R(np.einsum("c,ict->it", K1.w_tu([wsrc[j] for j in tch]), U), Qb)
    zk, zR = K1.z(lk), K1.z(R5)
    for nm_, v_ in (("lk", lk), ("R5", R5), ("l0", l0), ("zk", zk), ("zR", zR)):
        if not np.isfinite(v_).all():
            raise SystemExit("bien the: %s khong huu han" % nm_)
    dec = dict(ktn=ktn_dec, ktnr=(zk + ALPHA * zR).argmax(1), tinh=l0.argmax(1), tinh_r=(K1.z(l0) + ALPHA * zR).argmax(1))
    goc_idx = np.asarray(pi)[N0:]                                                                # chi so luot goc cua cac luot cham
    h40 = np.bincount(yp[:N0], minlength=lg0.shape[1])
    v = dict(nguon_chon_chi_so=tch, chi_so_cham=goc_idx.tolist(), y_cham=[int(x) for x in yt], khoi_tao_mang_nen=float((lg0[:N0].argmax(1).cpu().numpy() == yp[:N0]).mean()),
             so_lop_40_dau=int((h40 > 0).sum()), luot_moi_lop_40_dau=h40.tolist())
    for k in PP:
        v[k] = float((dec[k] == yt).mean()); v["qd_" + k] = [int(x) for x in dec[k]]
    v["_zk"], v["_zR"] = zk, zR
    return v


def theo_goc(v):
    """Quyet dinh tung phuong phap sap theo chi so luot goc."""
    o = np.argsort(v["chi_so_cham"])
    return {k: np.asarray(v["qd_" + k])[o] for k in PP}, np.asarray(v["chi_so_cham"])[o]


def cung_quyet_dinh(a, b):
    da, ia = theo_goc(a); db, ib = theo_goc(b)
    return np.array_equal(ia, ib) and all(np.array_equal(da[k], db[k]) for k in PP)


def chay_fold(i, D, a, dev, freqs, fs, NT, goc_json):
    gj = goc_json[i]
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))
    assert gj["seed"] == seed, (gj["seed"], seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=L.NCH, p_cuoi=0.95).to(dev)
    ck = os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")
    if sha(ck) != gj["trongso_sha256"]:
        print(f"CONG HOI QUY KHONG DAT fold {i+1}: hash trong so khac JSON goc", flush=True); sys.exit(4)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev); y = np.asarray(D["y"][i])
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():                                                 # cung loi goi nhu ktnr_buoc1 (khong dung RNG)
        h1u = L.truoc(model, Xu); _ = model(Xu)
    mods, wsrc = [], []
    for s in src:                                                         # giong het vong mo-dun nguon cua ktnr_buoc1.chay_fold
        Xs = torch.tensor(D["X"][s], device=dev); ys = torch.tensor(D["y"][s], device=dev, dtype=torch.long)
        with torch.no_grad():
            h1s = L.truoc(model, Xs)
        mods.append(L.hoc(model, h1s, ys, mc, L.BUOC_NGUON, L.LR_MOD))
        Us = h1s[:, 0].double().cpu().numpy(); Us = Us - Us.mean(-1, keepdims=True)
        wsrc.append(K1.trca(Us, D["y"][s], NT))
    Qb = K1.co_so(freqs, fs, ns)
    n = len(y); nb = n // NT; khoi40 = (N0 == NT and nb * NT == n)
    if khoi40:                                                            # moi khoi dung mot luot moi lop
        for b in range(nb):
            assert np.array_equal(np.bincount(y[b * NT:(b + 1) * NT], minlength=NT), np.ones(NT, int)), (i + 1, b + 1)
    ev = lambda pi: danh_gia(model, Xu, y, np.asarray(pi), mods, wsrc, mc, Qb)
    e = {"fold": i + 1, "seed": seed, "trongso_sha256": sha(ck), "so_khoi": nb, "co": []}
    goc = ev(np.arange(n))
    # ---- cong hoi quy ----
    src_goc = [src[j] + 1 for j in goc["nguon_chon_chi_so"]]
    lech = [k for k in PP if abs(goc[k] - gj[k]) > 1e-12]
    if lech or src_goc != gj["nguon_chon"]:
        print(f"CONG HOI QUY KHONG DAT fold {i+1}: {lech} nguon {src_goc} vs {gj['nguon_chon']}", flush=True); sys.exit(4)
    zk_g, zR_g = np.asarray(gj["luot"]["z_ktn"]), np.asarray(gj["luot"]["z_R"])
    assert gj["luot"]["chi_so"][0] == N0 + 1 and len(zk_g) == n - N0 and gj["luot"]["y"] == [int(v) for v in y[N0:]]
    # Quyet dinh KTN, KTN-R tung luot phai trung argmax cua diem goc da luu, khong mien luot nao (ca 8 JSON goc: 0 luot sat hoa).
    # Mang tinh va tinh_r: JSON goc khong luu quyet dinh tung luot, chi so duoc do chinh xac (gioi han cua cong, ghi trong bao cao).
    sat = {}
    for ten, s_g in (("ktn", zk_g), ("ktnr", zk_g + ALPHA * zR_g)):
        hai = np.sort(s_g, 1)[:, -2:]; sat[ten] = int(((hai[:, 1] - hai[:, 0]) < 1e-5).sum())
        khac = s_g.argmax(1) != np.asarray(goc["qd_" + ten])
        if khac.any():
            print(f"CONG HOI QUY KHONG DAT fold {i+1}: {ten} khac {int(khac.sum())} quyet dinh", flush=True); sys.exit(4)
    for ten, s_g, s_m in (("z_ktn", zk_g, goc["_zk"]), ("z_R", zR_g, goc["_zR"])):                    # kiem rieng tung thanh phan
        dz = float(np.abs(s_m - s_g).max()); e.setdefault("hoi_quy_lech_diem_max", {})[ten] = dz
        if dz > TOL_Z and Xu.is_cuda:                                     # ban goc chay GPU; chay thu CPU chi ghi lech (mo-dun hoc 100 buoc Adam lech so hoc theo thiet bi)
            print(f"CONG HOI QUY KHONG DAT fold {i+1}: {ten} lech {dz:.2e} > {TOL_Z}", flush=True); sys.exit(4)
    e["hoi_quy_luot_sat_hoa_goc"] = sat
    e["thiet_bi"] = "cuda" if Xu.is_cuda else "cpu"
    for v in [goc]:
        v.pop("_zk"); v.pop("_zR")
    e["goc"] = goc
    if khoi40:
        e["P1"] = []
        for r in range(1, R_P1 + 1):
            rg = np.random.default_rng([SEED, i + 1, r, 1])
            pi = np.concatenate([b * NT + rg.permutation(NT) for b in range(nb)])
            v = ev(pi); v.pop("_zk"); v.pop("_zR")
            if sorted(v["nguon_chon_chi_so"]) != sorted(goc["nguon_chon_chi_so"]) or not cung_quyet_dinh(v, goc):
                print(f"CONG BAT BIEN P1 KHONG DAT fold {i+1} r {r}", flush=True); sys.exit(5)
            if v["nguon_chon_chi_so"] != goc["nguon_chon_chi_so"]:
                e["co"].append(f"P1 r{r}: thu tu xep hang nguon khac goc, cung tap")
            e["P1"].append(dict(r=r, nguon_chon_chi_so=v["nguon_chon_chi_so"], pi_sha256=hashlib.sha256(pi.astype(np.int64).tobytes()).hexdigest()))
        e["B"] = []
        for b in range(nb):
            pi = np.concatenate([np.arange(b * NT, (b + 1) * NT)] + [np.arange(c * NT, (c + 1) * NT) for c in range(nb) if c != b])
            v = ev(pi); v.pop("_zk"); v.pop("_zR"); v["khoi_khoi_tao"] = b + 1
            if b == 0 and not (v["nguon_chon_chi_so"] == goc["nguon_chon_chi_so"] and cung_quyet_dinh(v, goc)):
                print(f"CONG B1 = GOC KHONG DAT fold {i+1}", flush=True); sys.exit(5)
            e["B"].append(v)
    e["T"] = []
    for r in range(1, R_T + 1):
        pi = np.random.default_rng([SEED, i + 1, r]).permutation(n)
        v = ev(pi); v.pop("_zk"); v.pop("_zR"); v["r"] = r
        v["pi_sha256"] = hashlib.sha256(pi.astype(np.int64).tobytes()).hexdigest(); e["T"].append(v)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "eld", "dong"])
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--json-goc", required=True, help="JSON KTN-R da chay (cong hoi quy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=0, help="chi de chay thu; ket qua danh dau thu=True, phan tich chinh thuc tu choi")
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
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))                                # nhu ktnr_buoc1.main
    assert cfg["nch"] == L.NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg["subjects"] == 0 and cfg.get("ds", "benchmark") == ds
    if ds == "dong":
        assert cfg["epochs"] == 1000 and cfg["lo"] == 100 and abs(cfg["dropout_cuoi"] - 0.95) < 1e-9 and cfg["seed"] == SEED and cfg.get("du_lieu_dong_sha256") == sha(DG.__file__)
    G = json.load(open(a.json_goc, encoding="utf-8"))
    assert G["ds"] == ds and abs(G["T"] - a.T) < 1e-9 and G["seed"] == SEED and G["n0"] == N0 and G["k"] == K and G["alpha"] == ALPHA and G["NT"] == NT
    assert len(G["tung_nguoi"]) == nsub
    # JSON goc cua Benchmark/BETA/eld do ban ktnr_buoc1 truoc khi them bo nak/dong sinh (ba bo cu khong doi phep tinh); ghi lai hash, viec chung minh
    # cung phep tinh la cong hoi quy tung quyet dinh o tren, khong phai trung hash.
    sha_goc_khop = G["script_sha256"] == sha(K1.__file__)
    goc = G["tung_nguoi"]; assert [p["fold"] for p in goc] == list(range(1, nsub + 1))
    D = {"X": [], "y": []}; freqs = None
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and X.shape[1] == CK and np.isfinite(X).all()
        assert np.array_equal(np.bincount(y, minlength=NT), np.full(NT, nl // NT))
        freqs = np.asarray(fr, float) if freqs is None else freqs
        assert np.allclose(freqs, fr)
        Xp = GD.tien_xu_ly(X, ds, "causal", ns, fs); assert np.isfinite(Xp).all()
        D["X"].append(Xp); D["y"].append(y)
    assert len(freqs) == NT
    nf = a.folds or nsub
    for i in range(nf):
        assert os.path.exists(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")), i + 1
    t0 = time.time(); per = []
    tam = a.out + ".dang-chay.tmp"
    if os.path.exists(tam):                                               # chay tiep: chi nhan tep tam cung cau hinh, fold lien tiep tu 1
        cu = json.load(open(tam))
        if isinstance(cu, dict) and cu.get("script_sha256") == sha(os.path.abspath(__file__)) and cu.get("json_goc_sha256") == sha(a.json_goc) \
                and [p["fold"] for p in cu["tung_nguoi"]] == list(range(1, len(cu["tung_nguoi"]) + 1)):
            per = cu["tung_nguoi"]; print(f"chay tiep tu fold {len(per) + 1}", flush=True)
    for i in range(len(per), nf):
        per.append(chay_fold(i, D, a, dev, freqs, fs, NT, goc))
        json.dump(dict(script_sha256=sha(os.path.abspath(__file__)), json_goc_sha256=sha(a.json_goc), tung_nguoi=per), open(tam + ".x", "w")); os.replace(tam + ".x", tam)
        r = per[-1]
        print(f"fold {i+1}: goc ktnr {r['goc']['ktnr']:.3f} | B {len(r.get('B', []))} T {len(r['T'])} | lech z {r['hoi_quy_lech_diem_max']} | {time.time()-t0:.0f} s", flush=True)
    me = os.path.abspath(__file__)
    js = dict(seed=SEED, ds=ds, T=a.T, n0=N0, k=K, alpha=ALPHA, NT=NT, R_T=R_T, R_P1=R_P1, so_nguoi_du_kien=nsub, so_fold_chay=nf, thu=bool(a.folds) or dev.type != "cuda", thiet_bi=dev.type,
              json_goc=a.json_goc, json_goc_sha256=sha(a.json_goc), json_goc_script_sha256=G["script_sha256"],
              json_goc_script_trung_ktnr_buoc1_hien_tai=sha_goc_khop, trongso=a.trongso,
              script_sha256=sha(me), ktnr_buoc1_sha256=sha(K1.__file__),
              ma_phu_sha256={os.path.basename(m_.__file__): sha(m_.__file__) for m_ in (B, GD, DE, L, DN, DG)}, tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    if os.path.exists(a.out + ".dang-chay.tmp"):
        os.remove(a.out + ".dang-chay.tmp")
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
