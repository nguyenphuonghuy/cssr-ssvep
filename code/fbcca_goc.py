#!/usr/bin/env python
"""Kiem tai hien FBCCA theo giao thuc cua cac bai du lieu goc.
Thuat toan giu nguyen fbcca_dung.py (Chen 2015 pt. 6: rho_c = sum_m w_m rho_{m,c}^2, w_m = m^-1.25 + 0.25; 5 dai [8m, 88] Hz
Chebyshev I cua b1_diem.filterbank; 5 hai), chi doi tien xu ly sang loc hai chieu (sosfiltfilt). Cham MOI luot (FBCCA khong hoc).
Hai giao thuc:
  bai_du_lieu : giao thuc cua bai du lieu goc. Tru TB theo kenh tren ca epoch, loc hai chieu tren ca epoch, cat [onset + tre, + ns).
                Benchmark (Wang 2017): tre 0,14 s; BETA (Liu 2020): them loc thong dai 3-100 Hz hai chieu (Butterworth bac 4, thay cho
                eegfilt FIR), tre 0,13 s; Dong (Dong va Tian 2023): tre 0,16 s (da chan 50 Hz trong tep).
                Bao ca ITR tung nguoi (bit/phut, thoi gian chuyen 0,55 s va 2 s) de so truc tiep voi ITR cong bo.
  oacca       : giao thuc cua bai OACCA (Wong 2022), de so voi moc CCA cua ban chuyen OACCA (0,613 / 0,571):
                doan [onset, onset + tre + 2 s) loc hai chieu, cat tu tre, cua so 0,6..1,5 s; tre 0,12 s (Benchmark), 0,13 s (BETA).
                Bao ca bien the khong binh phuong (sum_m w_m rho_m) nhu moc CCA cua ma OACCA.
Dung: python fbcca_goc.py --ds benchmark|beta|dong --giao-thuc bai_du_lieu|oacca --out <json> [--subjects n] [--cpu 8]
"""
import argparse, json, os, sys, time, hashlib
from multiprocessing import Pool
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B          # noqa: E402
import du_lieu_dong as DG    # noqa: E402

SEED = 20261107
CFG = {
    ("benchmark", "bai_du_lieu"): dict(tre=0.14, wins=[0.6, 0.8, 1.0, 1.25, 1.5, 1.75]),
    ("beta", "bai_du_lieu"): dict(tre=0.13, wins=[0.6, 0.8, 1.0, 1.2, 1.4]),
    ("dong", "bai_du_lieu"): dict(tre=0.16, wins=[0.6, 0.8, 1.0, 2.4, 4.0]),
    ("benchmark", "oacca"): dict(tre=0.12, wins=[w / 10 for w in range(6, 16)]),
    ("beta", "oacca"): dict(tre=0.13, wins=[w / 10 for w in range(6, 16)]),
}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
G = {}


def itr(P, M, T):
    if P <= 1.0 / M:
        return 0.0
    v = np.log2(M) + P * np.log2(P) + (0.0 if P >= 1 else (1 - P) * np.log2((1 - P) / (M - 1)))
    return float(v * 60.0 / T)


def khoi_tao(ds):
    DG.dang_ky(B, None, ds)


def mot_nguoi(args):
    ds, gt, s = args
    c = B.DS[ds]; fs = c["fs"]; on = c["onset"]; cf = CFG[(ds, gt)]
    lat = int(round(cf["tre"] * fs)); sos = B.filterbank(fs)
    X, y, fr = B.load_subject(ds, s, B.CH9)
    if gt == "bai_du_lieu":
        X = X - X.mean(axis=2, keepdims=True)
        if ds == "beta":
            X = sosfiltfilt(butter(4, [3 / (fs / 2), 100 / (fs / 2)], btype="bandpass", output="sos"), X, axis=2)
        st = on + lat
        F = np.stack([sosfiltfilt(so, X, axis=2) for so in sos], 1)[..., st:]             # (luot, NB, C, t)
    else:
        seg = X[:, :, on: on + lat + 2 * fs]
        assert seg.shape[2] == lat + 2 * fs, (ds, s, seg.shape)
        F = np.stack([sosfiltfilt(so, seg, axis=2) for so in sos], 1)[..., lat:]
    M = len(fr); kq = {}
    for T in cf["wins"]:
        ns = int(round(T * fs)); assert F.shape[3] >= ns, (ds, s, T, F.shape)
        Qy = B.refs(fr, fs, ns)
        r = np.empty((len(y), B.NB, M))
        for i in range(len(y)):
            for m in range(B.NB):
                qx = np.linalg.qr(B.center(F[i, m, :, :ns]).T)[0]
                r[i, m] = np.linalg.svd(np.einsum("tc,ktj->kcj", qx, Qy), compute_uv=False)[:, 0]
        acc = float(((np.einsum("ibk,b->ik", r ** 2, B.FBW)).argmax(1) == y).mean())
        acc1 = float(((np.einsum("ibk,b->ik", r, B.FBW)).argmax(1) == y).mean())
        kq[str(T)] = dict(acc=acc, acc_khong_binh_phuong=acc1, itr_055=itr(acc, M, T + 0.55), itr_2=itr(acc, M, T + 2.0))
    return dict(nguoi=s, n_luot=int(len(y)), kq=kq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "dong"])
    ap.add_argument("--giao-thuc", required=True, choices=["bai_du_lieu", "oacca"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--cpu", type=int, default=8)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    assert (a.ds, a.giao_thuc) in CFG
    khoi_tao(a.ds)
    n = a.subjects or B.DS[a.ds]["n"]; t0 = time.time()
    with Pool(a.cpu, initializer=khoi_tao, initargs=(a.ds,)) as p:
        per = p.map(mot_nguoi, [(a.ds, a.giao_thuc, s) for s in range(1, n + 1)])
    cf = CFG[(a.ds, a.giao_thuc)]; tb = {}
    for T in map(str, cf["wins"]):
        acc = np.array([e["kq"][T]["acc"] for e in per])
        tb[T] = dict(acc=100 * float(acc.mean()), acc_khong_binh_phuong=100 * float(np.mean([e["kq"][T]["acc_khong_binh_phuong"] for e in per])),
                     itr_055_tb_nguoi=float(np.mean([e["kq"][T]["itr_055"] for e in per])),
                     itr_2_tb_nguoi=float(np.mean([e["kq"][T]["itr_2"] for e in per])))
        print(a.ds, a.giao_thuc, T, "acc %.2f (khong binh phuong %.2f) ITR0.55 %.2f ITR2 %.2f" % (
            tb[T]["acc"], tb[T]["acc_khong_binh_phuong"], tb[T]["itr_055_tb_nguoi"], tb[T]["itr_2_tb_nguoi"]), flush=True)
    if a.giao_thuc == "oacca":
        tb["tb_0.6_1.5"] = dict(acc=float(np.mean([tb[str(w)]["acc"] for w in cf["wins"]])),
                                acc_khong_binh_phuong=float(np.mean([tb[str(w)]["acc_khong_binh_phuong"] for w in cf["wins"]])))
        print("TB 0,6..1,5 s:", tb["tb_0.6_1.5"], flush=True)
    me = os.path.abspath(__file__); d = os.path.dirname(me)
    out = dict(seed=SEED, ds=a.ds, giao_thuc=a.giao_thuc, tre_s=cf["tre"], cua_so=cf["wins"], so_nguoi=n,
               script_sha256=sha(me), b1_diem_sha256=sha(os.path.join(d, "b1_diem.py")), tb=tb, tung_nguoi=per,
               thoi_gian_s=time.time() - t0)
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
    print("xong %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
