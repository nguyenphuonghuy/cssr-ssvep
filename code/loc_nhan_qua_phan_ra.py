#!/usr/bin/env python
"""Tach anh huong cua cach loc len FBCCA, CCA mot dai va OACCA (nam dai, mot dai). Thuat toan va day loc giu nguyen,
chi doi tien xu ly. Tai dung b1_diem (filterbank, refs, FBW), doi_thu_oacca (bo_loc, tham_chieu, chay) va oacca_bon_bo (nap,
dat_so_dai), khong sua cac tep do.
Cach loc F:
  nq    : nhu bai. Tru TB doan truoc kich thich, loc nhan qua tu mau a (a = 0 hoac a = dau cua so - p), cat [st, st + n).
  pk    : pha khong (sosfiltfilt) tren doan [0, st + n) da tru TB, dung dung cuoi cua so (khong dung mau sau quyet dinh).
  pk_nt : pha khong nhu bai OACCA: doan tho [onset, onset + tre + 2 s), cat tu tre + d (co dung mau sau cua so).
st = onset + tre + d. Bien the "giu_T": cua so dai T; "giu_qd": cua so dai T - d (giu thoi diem quyet dinh).
FBCCA va CCA mot dai dung day loc b1_diem.filterbank (khong chan 50 Hz) nhu bai; OACCA dung bo_loc cua doi_thu_oacca (chan 50 Hz).
Cham theo thu tu tep: luot 41..het ("sau") va ca phien ("phien"); luu quyet dinh tung luot de kiem hoi quy va ghep cap.
Dung: python loc_nhan_qua_phan_ra.py --ds benchmark --pp fbcca|oacca5|oacca1 --F nq --p 0 --d 0 --bien-the giu_T --T 1.0 --out <json>
  (--pp fbcca tinh luon CCA mot dai tu cung tuong quan tung dai). --p 0 nghia la loc tu mau 0 nhu bai; --p 0.39 nghia la 0,39 s truoc dau cua so.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
from scipy.signal import filtfilt, lfilter, sosfilt, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B            # noqa: E402
import doi_thu_oacca as O      # noqa: E402
import oacca_bon_bo as OB      # noqa: E402
import du_lieu_eld as DE       # noqa: E402
import du_lieu_dong as DG      # noqa: E402
import du_lieu_nak as DN       # noqa: E402

SEED = 20261107
N0 = 40
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def cat(X, F, on, lat, d, n, fs, sos, notch, p_mau):
    """X (luot, C, t) tho. Tra (luot, NB, C, n)."""
    st = on + lat + d
    out = np.empty((X.shape[0], len(sos), X.shape[1], n))
    if F == "pk_nt":
        seg = X[:, :, on: on + lat + 2 * fs]
        assert seg.shape[2] == lat + 2 * fs and lat + d + n <= seg.shape[2]
        if notch is not None:
            seg = filtfilt(notch[0], notch[1], seg, axis=2)
        for k, so in enumerate(sos):
            out[:, k] = sosfiltfilt(so, seg, axis=2)[:, :, lat + d: lat + d + n]
        return out
    assert st + n <= X.shape[2]
    Xb = X[:, :, : st + n] - X[:, :, :on].mean(axis=2, keepdims=True)
    if F == "nq":
        a = 0 if p_mau <= 0 else max(0, st - p_mau)
        seg = Xb[:, :, a:]
        if notch is not None:
            seg = lfilter(notch[0], notch[1], seg, axis=2)
        for k, so in enumerate(sos):
            out[:, k] = sosfilt(so, seg, axis=2)[:, :, st - a: st - a + n]
        return out
    assert F == "pk"
    seg = Xb
    if notch is not None:
        seg = filtfilt(notch[0], notch[1], seg, axis=2)
    for k, so in enumerate(sos):
        out[:, k] = sosfiltfilt(so, seg, axis=2)[:, :, st: st + n]
    return out


def diem(dec, y):
    ok = np.asarray(dec) == np.asarray(y)
    return dict(sau=float(ok[N0:].mean()), phien=float(ok.mean()), dec=[int(v) for v in dec])


def fbcca_nguoi(ds, s, cf):
    c = B.DS[ds]; fs = c["fs"]
    X, y, fr = B.load_subject(ds, s, B.CH9)
    sig = cat(X, cf["F"], c["onset"], cf["lat"], cf["d"], cf["n"], fs, B.filterbank(fs), None, cf["p_mau"])
    Qy = B.refs(fr, fs, cf["n"])
    rho = np.zeros((len(y), sig.shape[1], len(fr)))
    for i in range(len(y)):
        for m in range(sig.shape[1]):
            qx = np.linalg.qr(B.center(sig[i, m]).T)[0]
            rho[i, m] = np.linalg.svd(np.einsum("tc,ktj->kcj", qx, Qy), compute_uv=False)[:, 0]
    return dict(nguoi=s, fbcca=diem((B.FBW[None, :, None] * rho ** 2).sum(1).argmax(1), y), cca1=diem(rho[:, 0].argmax(1), y))


def oacca_nguoi(ds, s, nb, cf):
    OB.dat_so_dai(nb)
    c = B.DS[ds]; fs = c["fs"]
    X, y, fr, ph, nguon = OB.nap(ds, s)
    notch, sos = O.bo_loc(fs)
    sig = cat(X, cf["F"], c["onset"], cf["lat"], cf["d"], cf["n"], fs, sos, notch, cf["p_mau"])
    Y, Qy = O.tham_chieu(fr, ph, cf["n"], fs)
    d1, d6, yt = O.chay(sig, y, Y, Qy, np.arange(len(y)))
    return dict(nguoi=s, nguon_pha=nguon, oacca=diem(d6, yt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "eld", "dong", "nak"])
    ap.add_argument("--pp", required=True, choices=["fbcca", "oacca5", "oacca1"])
    ap.add_argument("--F", required=True, choices=["nq", "pk", "pk_nt"])
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--d", type=float, default=0.0, help="giay")
    ap.add_argument("--bien-the", default="giu_T", choices=["giu_T", "giu_qd"])
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--nguoi", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    assert not (a.pp != "fbcca" and a.ds == "nak"), "OACCA chua co pha kich thich cho Nakanishi"
    for d in B.DS:
        B.DS[d]["root"] = os.environ.get("B1_ROOT_" + d.upper(), B.DS[d]["root"])
    DE.dang_ky_B(B); DG.dang_ky(B, None, "dong"); DN.dang_ky(B, None, "nak")
    c = B.DS[a.ds]; fs = c["fs"]
    d = int(round(a.d * fs)); n = int(round((a.T - (a.d if a.bien_the == "giu_qd" else 0.0)) * fs))
    cf = dict(F=a.F, lat=int(round(c["lat"] * fs)), d=d, n=n, p_mau=int(round(a.p * fs)))
    ng = list(range(1, c["n"] + 1)) if a.nguoi == "all" else list(range(int(a.nguoi.split("-")[0]), int(a.nguoi.split("-")[1]) + 1))
    t0 = time.time(); per = []
    for s in ng:
        per.append(fbcca_nguoi(a.ds, s, cf) if a.pp == "fbcca" else oacca_nguoi(a.ds, s, {"oacca5": 5, "oacca1": 1}[a.pp], cf))
    keys = ["fbcca", "cca1"] if a.pp == "fbcca" else ["oacca"]
    tb = {k: dict(sau=100 * float(np.mean([e[k]["sau"] for e in per])), phien=100 * float(np.mean([e[k]["phien"] for e in per]))) for k in keys}
    me = os.path.abspath(__file__); hd = os.path.dirname(me)
    js = dict(seed=SEED, ds=a.ds, pp=a.pp, F=a.F, p_s=a.p, d_s=a.d, bien_the=a.bien_the, T_s=a.T, cua_so_mau=n, dich_mau=d,
              do_tre_mau=cf["lat"], onset=c["onset"], fs=fs, n0=N0, nguoi=ng, tb=tb, thoi_gian_s=time.time() - t0,
              script_sha256=sha(me), b1_diem_sha256=sha(os.path.join(hd, "b1_diem.py")), doi_thu_oacca_sha256=sha(os.path.join(hd, "doi_thu_oacca.py")),
              tung_nguoi=per)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(js, tung_nguoi=None), open(a.out[:-5] + "_config.json", "w"), indent=1)
    tmp = a.out + ".tmp"
    json.dump(js, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
    print(a.ds, a.pp, a.F, "p", a.p, "d", a.d, a.bien_the, " ".join("%s %.2f" % (k, v["sau"]) for k, v in tb.items()), "%.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
