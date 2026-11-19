#!/usr/bin/env python
"""OACCA (Wong et al., IEEE TBME 2022, DOI 10.1109/TBME.2021.3133594) on the four B1 datasets, project protocol ("nq"), with 5 or 1 subbands.
Reuses the OACCA class, filtering and references of doi_thu_oacca.py (port of edwin465/SSVEP-OACCA v1.2, GPL-3.0)
without modifying that file: the number of subbands is set by rebinding doi_thu_oacca.NB and doi_thu_oacca.FB before use.
Protocol "nq": pre-stimulus mean removal, causal filtering from sample 0, project latency, file trial order, windows 0.6 and 1.0 s,
scored on trials 41..end ("sau") and on the whole session ("phien"). OACCA adapts from trial 1 using its own predictions only (no labels).
Phases: Benchmark (Freq_Phase.mat) and BETA (suppl_info) as in doi_thu_oacca.nap; Dong 2023 (k mod 4) x 0.5 pi (du_lieu_dong);
eldBETA: Suppl_info.Phase if present, otherwise 0 (recorded in the output).
Regression gate (--kiem-goc): with 5 subbands, decisions on Benchmark subjects 1-2 must equal those of doi_thu_oacca.py --che-do nq.
Usage: python oacca_bon_bo.py --ds benchmark|beta|eld|dong --so-dai 5|1 --out <json> [--nguoi 1-5] [--kiem-goc]
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doi_thu_oacca as O    # noqa: E402
import b1_diem as B          # noqa: E402
import du_lieu_eld as DE     # noqa: E402
import du_lieu_dong as DG    # noqa: E402

N0 = 40
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def dat_so_dai(nb):
    O.NB = nb
    O.FB = np.array([k ** -1.25 + 0.25 for k in range(1, nb + 1)])


def nap(ds, s):
    """Tra X (luot, C, t) tho, y, freqs, phases, nguon_pha."""
    if ds in ("benchmark", "beta"):
        X, y, fr, ph = O.nap(ds, s, B)
        return X, y, fr, ph, "tep"
    if ds == "dong":
        X, y, fr = DG.nap(s)
        return X, y, fr, DG.PHA_DONG_PI * np.pi, "du_lieu_dong (k mod 4) x 0.5 pi"
    from scipy.io import loadmat
    X, y, fr = DE.nap(s, B.CH9)
    si = loadmat(os.path.join(DE.ELD, "S%d.mat" % s), squeeze_me=True, struct_as_record=False)["data"].Suppl_info
    if hasattr(si, "Phase"):
        ph = np.asarray(si.Phase, float).ravel()
        if ph.size == len(fr) and np.isfinite(ph).all():
            return X, y, fr, ph, "Suppl_info.Phase"
    return X, y, fr, np.zeros(len(fr)), "0 (khong co Phase)"


def mot_nguoi(ds, s, nb, wins, fs, on, lat):
    dat_so_dai(nb)
    X, y, fr, ph, nguon = nap(ds, s)
    assert np.isfinite(X).all() and np.isfinite(fr).all() and np.isfinite(ph).all()
    sig = O.tien_xu_ly(X, "nq", on, lat, max(wins), fs)
    e = {"nguoi": s, "nguon_pha": nguon, "T": {}}
    for n in wins:
        Y, Qy = O.tham_chieu(fr, ph, n, fs)
        d1, d6, yt = O.chay(sig[:, :, :, :n], y, Y, Qy, np.arange(len(y)))
        f = lambda d: dict(sau=float((d[N0:] == yt[N0:]).mean()), phien=float((d == yt).mean()), dec=[int(v) for v in d])
        e["T"]["%.1f" % (n / fs)] = dict(cca=f(d1), oacca=f(d6))
    return e, X.shape[1], len(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "eld", "dong"])
    ap.add_argument("--so-dai", type=int, required=True, choices=[1, 5])
    ap.add_argument("--out", required=True)
    ap.add_argument("--nguoi", default="all")
    ap.add_argument("--kiem-goc", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("output exists, not overwriting")
    for d in B.DS:
        B.DS[d]["root"] = os.environ.get("B1_ROOT_" + d.upper(), B.DS[d]["root"])
    DE.dang_ky_B(B); DG.dang_ky(B, None, "dong")
    c = B.DS[a.ds]; fs = c["fs"]; on = c["onset"]; lat = int(round(c["lat"] * fs))
    wins = [int(round(0.6 * fs)), int(round(1.0 * fs))]
    ng = list(range(1, c["n"] + 1)) if a.nguoi == "all" else list(range(int(a.nguoi.split("-")[0]), int(a.nguoi.split("-")[1]) + 1))
    kiem = {}
    if a.kiem_goc:
        assert a.ds == "benchmark" and a.so_dai == 5
        import subprocess, tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "goc.json")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "doi_thu_oacca.py"),
                        "--ds", "benchmark", "--che-do", "nq", "--nguoi", "1-2", "--out", tmp], check=True)
        goc = json.load(open(tmp))["tung_nguoi"]
        for g in goc:
            e, _, _ = mot_nguoi("benchmark", g["nguoi"], 5, wins, fs, on, lat)
            for T in g["T"]:
                for k in ("cca", "oacca"):
                    assert g["T"][T][k]["dec"] == e["T"][T][k]["dec"], ("lech hoi quy", g["nguoi"], T, k)
        kiem = dict(nguoi=[1, 2], ket_qua="trung khop quyet dinh voi doi_thu_oacca.py --che-do nq", doi_thu_oacca_sha256=sha(O.__file__))
        print("kiem hoi quy: dat", flush=True)
    t0 = time.time(); per = []
    for s in ng:
        e, C, N = mot_nguoi(a.ds, s, a.so_dai, wins, fs, on, lat)
        per.append(e)
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"nguoi {s} xong {time.time()-t0:.0f} s", flush=True)
    dat_so_dai(a.so_dai)
    js = dict(seed=O.SEED, ds=a.ds, che_do="nq", so_dai=a.so_dai, n0=N0, nguoi=ng, fs=fs, onset=on, do_tre_mau=lat, cua_so_mau=wins,
              mac={("%.1f" % (n / fs)): O.dem_mac(C, n, N) for n in wins}, kiem_hoi_quy=kiem,
              script_sha256=sha(os.path.abspath(__file__)), doi_thu_oacca_sha256=sha(O.__file__), b1_diem_sha256=sha(B.__file__),
              thoi_gian_s=time.time() - t0, tung_nguoi=per)
    tmp = a.out + ".tmp"
    json.dump(js, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
    if os.path.exists(a.out + ".dang-chay.tmp"):
        os.remove(a.out + ".dang-chay.tmp")
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
