#!/usr/bin/env python
"""Buoc 1 cua kiem doc lap tren eldBETA: CHI kiem doc du lieu va pho, KHONG chay ung vien nao,
khong dung mau nguoi nguon. eldBETA: data.EEG.Epoch (64 kenh, 1500 mau, 9 muc tieu, 7 khoi), fs 250, 0,5 s truoc kich thich.
(1) ti le nang luong 48-52 Hz trong 8-88 Hz (9 kenh cham-dinh), TB va lon nhat qua nguoi;
(2) FBCCA chuan (5 dai cua b1_diem, 5 hai, FBW) voi tan so trong Suppl_info, T = 1,0 va 2,0 s, do tre 0,14 s (gia dinh nhu Benchmark),
    va voi anh xa sai (tan so tang dan) de kiem thu tu muc tieu.
"""
import json, os, sys, hashlib
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B  # noqa: E402

ROOT = "/gpfs0/users/gv_dnton/datasets/ssvep/eldbeta"
NB = B.NB


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        sys.exit("tep ket qua da co")
    B.DS["eld"] = dict(root=ROOT, n=100, fs=250, onset=125, lat=0.14)
    fs = 250; sos = B.filterbank(fs)
    ratio, acc = [], {}
    info = None
    for s in range(1, 101):
        d = loadmat(os.path.join(ROOT, "S%d.mat" % s), squeeze_me=True, struct_as_record=False)["data"]
        E = d.EEG.Epoch; si = d.Suppl_info
        assert E.shape == (64, 1500, 9, 7) and np.isfinite(E).all(), (s, E.shape)
        names = [str(r[3]).strip().upper() for r in si.Channel]
        idx = [names.index(c) for c in B.CH9]
        fr = np.asarray(si.Frequency, float).ravel()
        if info is None:
            info = dict(tan_so=fr.tolist(), pha=np.asarray(si.Phase, float).ravel().tolist(), kenh=[names[i] for i in idx], srate=int(si.Srate))
        assert np.allclose(fr, info["tan_so"]) and int(si.Srate) == 250
        X = E[idx].transpose(3, 2, 0, 1).reshape(63, 9, 1500).astype(np.float64)      # khoi-chinh, muc tieu 0..8
        y = np.tile(np.arange(9), 7)
        xc = X - X.mean(2, keepdims=True)
        f, p = welch(xc.reshape(-1, 1500), fs, nperseg=500); p = p.mean(0)
        ratio.append(float(p[(f >= 48) & (f <= 52)].sum() / p[(f >= 8) & (f <= 88)].sum()))
        P = B.preprocess(X, "eld", sos, 500)                                          # (63, NB, 9, 500)
        for T in (1.0, 2.0):
            ns = int(T * fs); x = B.center(P[..., :ns])
            for nm, ff in (("dung", fr), ("sai", np.sort(fr))):
                Qr = B.refs(ff, fs, ns)
                sc = np.empty((63, 9))
                for i in range(63):
                    rb = np.empty((9, NB))
                    for b in range(NB):
                        qx = np.linalg.qr(x[i, b].T)[0]
                        rb[:, b] = np.linalg.svd(np.einsum("tc,kth->kch", qx, Qr), compute_uv=False)[:, 0] ** 2
                    sc[i] = rb @ B.FBW
                acc.setdefault(f"fbcca_{nm}_T{T}", []).append(float((sc.argmax(1) == y).mean()))
    res = dict(thong_tin=info, ti_le_50Hz_tb=float(np.mean(ratio)), ti_le_50Hz_trung_vi=float(np.median(ratio)), ti_le_50Hz_max=float(np.max(ratio)),
               so_nguoi_ti_le_tren_0_1=int(np.sum(np.array(ratio) > 0.1)),
               fbcca={k: dict(tb=float(np.mean(v)), sd=float(np.std(v, ddof=1)), min=float(np.min(v))) for k, v in acc.items()},
               script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
