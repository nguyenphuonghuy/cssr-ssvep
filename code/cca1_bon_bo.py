#!/usr/bin/env python
"""Single-band CCA (first subband of FBCCA only), an under-budget training-free comparison.
Same preprocessing as fbcca_dung.py and KTN-R/b1_diem: pre-stimulus mean removal, causal Chebyshev I filtering from sample 0,
window [onset + latency, + ns); band 1 = [8, 88] Hz of b1_diem.filterbank; sin/cos references with 5 harmonics (b1_diem.refs).
Decision = argmax_c of the largest canonical correlation rho_{1,c}. Scored on trials 41..end ("sau") and on the whole session ("phien").
Regression gate (--kiem): the 5-band FBCCA computed here from the same per-band correlations must reproduce
results/accuracy/fbcca-dung_seed20261107_20261118.json per subject.
Usage: python cca1_bon_bo.py --out <json> [--bo benchmark beta eld dong] [--kiem <fbcca json>]
"""
import argparse, json, os, sys, time, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402
import du_lieu_eld as DE     # noqa: E402
import du_lieu_dong as DG    # noqa: E402

N0 = 40
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bo", nargs="*", default=["benchmark", "beta", "eld", "dong"])
    ap.add_argument("--kiem", default=None)
    ap.add_argument("--nguoi-toi-da", type=int, default=0, help="0 = all; >0 for a smoke test")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("output exists, not overwriting")
    me = os.path.abspath(__file__)
    ref = json.load(open(a.kiem)) if a.kiem else None
    out = dict(seed=None, phuong_phap="CCA 1 dai [8, 88] Hz, 5 hai, loc nhan qua, cham luot 41..het va ca phien",
               script_sha256=sha(me), b1_diem_sha256=sha(os.path.join(os.path.dirname(me), "b1_diem.py")), bo={}, kiem_fbcca={})
    for ds in a.bo:
        DE.dang_ky(B, GD, ds); DG.dang_ky(B, GD, ds)
        c = B.DS[ds]; fs = c["fs"]; sos = B.filterbank(fs); out["bo"][ds] = {}
        nmax = c["n"] if a.nguoi_toi_da <= 0 else min(c["n"], a.nguoi_toi_da)
        for T in (1.0, 0.6):
            ns = int(round(T * fs)); t0 = time.time(); per = []; lech = 0
            for s in range(1, nmax + 1):
                X, y, fr = B.load_subject(ds, s, B.CH9)
                Xp = B.preprocess(X, ds, sos, ns)
                Qy = B.refs(fr, fs, ns)
                n, nb, C, _ = Xp.shape; N = len(fr)
                rho = np.zeros((n, nb, N))
                for i in range(n):
                    for m in range(nb):
                        qx = np.linalg.qr(B.center(Xp[i, m]).T)[0]
                        rho[i, m] = np.linalg.svd(np.einsum("tc,ktj->kcj", qx, Qy), compute_uv=False)[:, 0]
                d1 = rho[:, 0].argmax(1)
                d5 = (B.FBW[None, :, None] * rho ** 2).sum(1).argmax(1)
                ok1, ok5 = d1 == y, d5 == y
                e = dict(nguoi=s, sau=float(ok1[N0:].mean()), phien=float(ok1.mean()), n_cham=int(n - N0),
                         fbcca_sau=float(ok5[N0:].mean()), fbcca_phien=float(ok5.mean()))
                if ref is not None and ds in ref["bo"]:
                    r = ref["bo"][ds][str(T)]["tung_nguoi"][s - 1]
                    assert r["nguoi"] == s
                    lech += int(abs(r["sau"] - e["fbcca_sau"]) > 1e-12 or abs(r["phien"] - e["fbcca_phien"]) > 1e-12)
                per.append(e)
            mac = int(B.macs("fbcca", C, ns, n=N))
            out["bo"][ds][str(T)] = dict(tb_sau=100 * float(np.mean([p["sau"] for p in per])), tb_phien=100 * float(np.mean([p["phien"] for p in per])),
                                        fbcca_tb_sau=100 * float(np.mean([p["fbcca_sau"] for p in per])), so_nguoi=len(per), C=C, N=N, mac_1dai=mac,
                                        tung_nguoi=per)
            if ref is not None:
                out["kiem_fbcca"]["%s_%s" % (ds, T)] = dict(so_nguoi_lech=lech)
                assert lech == 0, ("FBCCA khong tai hien", ds, T, lech)
            print(ds, T, "%d nguoi, C=%d, N=%d, MAC %d, kiem lech %d, %.0f s" % (len(per), C, N, mac, lech, time.time() - t0), flush=True)
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
