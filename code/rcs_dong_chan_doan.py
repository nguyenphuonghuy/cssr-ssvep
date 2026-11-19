#!/usr/bin/env python
"""Chan doan RCS tren Dong 2023.
Dung lai ham cua rcs_tai_hien.py (TRCA gop lop 5 dai + FBCCA 5 dai, trong so 1:1 = "TRCA w/R"), khong thich nghi.
Moi nguoi dich u, nguon = 58 nguoi con lai (co nhan, nhu RCS). Cham luot 41..het (chi so 0-based >= 40) va ca phien.
  D0 rcs_chon : 5 dai dien do luat gioi thieu cua RCS chon (khong nhan), giong rep_wR cua rcs_tai_hien (doi chung).
  D1 nhan_that: 5 nguon co do chinh xac cao nhat tren luot 1..40 cua nguoi dich khi dung RIENG mo hinh w/R cua tung nguon
                (DUNG NHAN THAT cua nguoi dich, la chan tren cua khau chon nguon).
  D2 tat_ca   : gop ca 58 nguon.
  D3 ngau_nhien: 5 nguon ngau nhien, 20 lan, rng(SEED + 1000 u + r).
Dung: python rcs_dong_chan_doan.py --out <json> [--subjects n]
"""
import argparse, json, os, sys, time, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B            # noqa: E402
import du_lieu_dong as DG      # noqa: E402
import rcs_tai_hien as R       # noqa: E402

SEED = 20261107
WINS = [0.6, 0.8, 1.0]
N0, K, N_NN = 40, 5, 20
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def mo_hinh(X, Y, nguon, NT, NB, C, ns):
    W = np.empty((NB, C, NT)); M = np.empty((NT, NB, C, ns))
    for k in range(NT):
        D = np.concatenate([X[s][Y[s] == k] for s in nguon])
        W[:, :, k] = R.trca_w(D); M[k] = D.mean(0)
    Tp, Tn = R.chieu_mau(W, M)
    return W, Tp, Tn


def diem_tat(W, Tp, Tn, x, A):
    return np.stack([R.diem(W, Tp, Tn, x[i]).T @ A for i in range(len(x))])   # (n, NT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    DG.dang_ky(B, None, "dong")
    c = B.DS["dong"]; fs = c["fs"]; sos = B.filterbank(fs); A = R.A; NB = R.NB; NT = 40
    nsub = a.subjects or c["n"]; ns_max = int(round(max(WINS) * fs)); t0 = time.time()
    P, Y = [], []
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject("dong", s, B.CH9)
        freqs = np.asarray(fr, float)
        P.append(B.preprocess(X, "dong", sos, ns_max).astype(np.float32)); Y.append(y)
    C = P[0].shape[2]
    print(f"nap {nsub} nguoi: {time.time()-t0:.0f} s", flush=True)
    me = os.path.abspath(__file__); d = os.path.dirname(me)
    out = dict(seed=SEED, ds="dong", n_subjects=nsub, n0=N0, k=K, so_lan_ngau_nhien=N_NN, windows=WINS,
               script_sha256=sha(me), rcs_tai_hien_sha256=sha(os.path.join(d, "rcs_tai_hien.py")),
               b1_diem_sha256=sha(os.path.join(d, "b1_diem.py")), ket_qua={})
    for T in WINS:
        ns = int(round(T * fs))
        X = [B.center(p[..., :ns].astype(np.float64)) for p in P]
        Qr = B.refs(freqs, fs, ns)
        FB5 = []
        for x in X:
            f = np.empty((x.shape[0], NB, NT))
            for i in range(x.shape[0]):
                for b in range(NB):
                    qx = np.linalg.qr(x[i, b].T)[0]
                    f[i, b] = np.linalg.svd(np.einsum("tc,kth->kch", qx, Qr), compute_uv=False)[:, 0]
            FB5.append(np.einsum("ibk,b->ik", f, A))
        Ms = [mo_hinh(X, Y, [s], NT, NB, C, ns) for s in range(nsub)]         # mo hinh rieng tung nguon
        print(T, f"mo hinh nguon xong {time.time()-t0:.0f} s", flush=True)
        per = []
        for u in range(nsub):
            src = [s for s in range(nsub) if s != u]; x = X[u]; y = Y[u]; fb5 = FB5[u]
            cham = lambda sc: dict(sau=float((sc[N0:].argmax(1) == y[N0:]).mean()), phien=float((sc.argmax(1) == y).mean()))
            # do chinh xac tren luot 1..40 cua tung mo hinh nguon rieng (w/R), dung nhan that
            acc40 = np.array([((diem_tat(*Ms[s], x[:N0], A) + fb5[:N0]).argmax(1) == y[:N0]).mean() for s in src])
            # D0: luat gioi thieu cua RCS (khong nhan), giong rcs_tai_hien
            al = np.zeros(len(src)); prev = None; cnt = 0; top = None
            for i in range(R.N_REC):
                for j, s in enumerate(src):
                    r = R.diem(Ms[s][0], Ms[s][1], Ms[s][2], x[i])
                    al[j] += R.alpha_t(fb5[i] + r.T @ A)
                top = frozenset(np.argsort(al / (i + 1), kind="stable")[:K].tolist())
                cnt = cnt + 1 if top == prev else 1
                prev = top
                if cnt >= R.M_ON:
                    break
            nhom = {"rcs_chon": [src[j] for j in sorted(top)],
                    "nhan_that": [src[j] for j in np.argsort(-acc40, kind="stable")[:K]],
                    "tat_ca": src}
            e = dict(nguoi=u + 1, acc40_tot_nhat=float(acc40.max()), fbcca5=cham(fb5))
            for ten, ng in nhom.items():
                W, Tp, Tn = mo_hinh(X, Y, ng, NT, NB, C, ns)
                e[ten] = cham(diem_tat(W, Tp, Tn, x, A) + fb5); e[ten]["nguon"] = [int(s) + 1 for s in ng] if ten != "tat_ca" else "58 nguon"
            nn = []
            for r_ in range(N_NN):
                ng = np.random.default_rng(SEED + 1000 * u + r_).choice(src, K, replace=False).tolist()
                W, Tp, Tn = mo_hinh(X, Y, ng, NT, NB, C, ns)
                nn.append(cham(diem_tat(W, Tp, Tn, x, A) + fb5))
            e["ngau_nhien"] = dict(sau=float(np.mean([v["sau"] for v in nn])), phien=float(np.mean([v["phien"] for v in nn])))
            per.append(e)
            if (u + 1) % 10 == 0:
                print(T, "nguoi", u + 1, f"{time.time()-t0:.0f} s", flush=True)
        tb = {k: 100 * float(np.mean([p[k]["sau"] for p in per])) for k in ("fbcca5", "rcs_chon", "nhan_that", "tat_ca", "ngau_nhien")}
        out["ket_qua"][str(T)] = dict(tb_sau=tb, tung_nguoi=per)
        print(T, "TB sau (%):", " ".join("%s %.2f" % kv for kv in tb.items()), flush=True)
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
