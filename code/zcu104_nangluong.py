#!/usr/bin/env python
"""Energy per decision on the ZCU104.
Run in ~/ssvep_zcu104_pkg/doi_thu (same files as zcu104_doithu_ktnr.py). Three roles:
  --vai-tro lay-mau : sampler process. Every 1/--hz s, append "t_monotonic P_ina226_uW P_irps43_pin_uW P_irps44_pin_uW" to --out until killed.
                      INA226 = 12 V input of the whole board; IRPS5401 0x43 / 0x44 = input power of the two PMBus regulators.
  --vai-tro moc     : print time.monotonic() (phase markers for idle windows).
  --vai-tro chay    : build the decision function of --pp with zcu104_doithu_ktnr.lam_* (same code as the joint timing session),
                      warm up --khoi-dong trials, then make back-to-back decisions on Benchmark subject 1 for --giay seconds.
                      Writes one JSON line with t_dau, t_cuoi (time.monotonic, the same CLOCK_MONOTONIC as the sampler), the number of decisions
                      and the per-decision times. Decisions are not compared with labels here (decision equality was checked in the timing session).
  --vai-tro khoitao : CSS-R initialization trials (S source modules scored per trial), same code path as zcu104_dolai.khoitao, looped for --giay s.
Energy is computed offline by zcu104_nangluong_tong_hop.py: E = (mean P in [t_dau, t_cuoi] - mean P in the idle window before) x mean decision time.
"""
import argparse, contextlib, functools, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def tim_hwmon():
    r = {}
    for h in sorted(os.listdir("/sys/class/hwmon")):
        p = os.path.join("/sys/class/hwmon", h)
        ten = open(os.path.join(p, "name")).read().strip()
        dev = os.path.realpath(os.path.join(p, "device"))
        if ten == "ina226":
            r["ina226"] = os.path.join(p, "power1_input")
        elif ten == "irps5401":
            r["irps_" + dev.rsplit("-", 1)[-1]] = os.path.join(p, "power1_input")     # 0043 / 0044, power1 = pin
    assert "ina226" in r, r
    return r


def lay_mau(a):
    tep = tim_hwmon(); keys = ["ina226", "irps_0043", "irps_0044"]
    fs = [open(tep[k]) if k in tep else None for k in keys]
    dt = 1.0 / a.hz; f = open(a.out, "a", buffering=1)
    f.write("# cot: t_monotonic " + " ".join(keys) + " (uW)\n")
    t_next = time.monotonic()
    while True:
        vals = []
        for h in fs:
            if h is None:
                vals.append("nan"); continue
            h.seek(0); vals.append(h.read().strip())
        f.write("%.6f %s\n" % (time.monotonic(), " ".join(vals)))
        t_next += dt
        time.sleep(max(0.0, t_next - time.monotonic()))


def chay(a):
    import torch
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); np.random.seed(a.seed)
    import zcu104_doithu_ktnr as Z
    if a.pp.startswith("oacca"):                       # OACCA (zcu104_oacca.py): decision + adaptation update, trials in file order, cycled
        import zcu104_oacca as ZO
        quyet, Xr, _ = ZO.lam_oacca(a.T, a.seed, int(a.pp[5:]))
    else:
        f = {"ktnr": Z.lam_ktnr, "fbcca": Z.lam_fbcca, "rcs": Z.lam_rcs, "ccanet": Z.lam_ccanet,
             "ed22": functools.partial(Z.lam_ed, nch=22), "edgoc": functools.partial(Z.lam_ed, nch=120)}[a.pp]
        quyet, _, _ = f(a.T, a.seed)
        Xr = Z.nap(1)[0]
    ngu = torch.no_grad() if a.pp != "ccanet" else contextlib.nullcontext()
    tt = []
    with ngu:
        for i in range(a.khoi_dong):
            quyet(Xr[i % len(Xr)])
        t_dau = time.monotonic(); i = 0
        while True:
            t0 = time.perf_counter(); quyet(Xr[i % len(Xr)]); tt.append(time.perf_counter() - t0); i += 1
            if time.monotonic() - t_dau >= a.giay and i >= a.n_toi_thieu:
                break
        t_cuoi = time.monotonic()
    return dict(vai_tro="chay", pp=a.pp, T=a.T, luong=a.luong, seed=a.seed, t_dau=t_dau, t_cuoi=t_cuoi, so_quyet_dinh=len(tt),
                tb_ms=1e3 * float(np.mean(tt)), tung_luot_ms=[round(1e3 * v, 4) for v in tt])


def khoitao(a):
    import torch
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); np.random.seed(a.seed)
    import global_dnn as GD
    import zcu104_dolai as DL
    rng = np.random.default_rng(a.seed)
    C, ns, N, S = 9, int(round(a.T * DL.FS)), a.N, a.S
    m = GD.GlobalDNN(C, ns, N, nch=22).eval()
    dW = [0.01 * torch.randn_like(m.l2.weight) for _ in range(S)]; db = [0.01 * torch.randn_like(m.l2.bias) for _ in range(S)]
    Ws = [m.l2.weight + d for d in dW]; bs = [m.l2.bias + d for d in db]
    wsrc = [v / np.linalg.norm(v) for v in rng.standard_normal((S, C))]
    X = torch.randn(40, 1, 3, C, ns)
    tk = []
    with torch.no_grad():
        for _ in range(5):
            m(X[0]); DL.sau(m, m.l1(X[0]), Ws[0], bs[0])
        t_dau = time.monotonic()
        while True:
            acc = torch.zeros(S)
            for i in range(40):
                t0 = time.perf_counter()
                x = X[i]
                int(m(x).argmax(1))
                h1 = m.l1(x)
                for s in range(S):
                    acc[s] += DL.do_tin(DL.sau(m, h1, Ws[s], bs[s]))[0]
                if i == 39:
                    top = torch.argsort(-acc, stable=True)[:5]
                    torch.stack([dW[int(j)] for j in top]).mean(0); torch.stack([db[int(j)] for j in top]).mean(0)
                    DL.w_tu([wsrc[int(j)] for j in top])
                tk.append(time.perf_counter() - t0)
            if time.monotonic() - t_dau >= a.giay:
                break
        t_cuoi = time.monotonic()
    return dict(vai_tro="khoitao", S=S, N=N, T=a.T, luong=a.luong, seed=a.seed, t_dau=t_dau, t_cuoi=t_cuoi, so_quyet_dinh=len(tk),
                tb_ms=1e3 * float(np.mean(tk)), tung_luot_ms=[round(1e3 * v, 3) for v in tk])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vai-tro", required=True, choices=["lay-mau", "moc", "chay", "khoitao"])
    ap.add_argument("--pp", choices=["ktnr", "fbcca", "rcs", "ed22", "edgoc", "ccanet", "oacca5", "oacca1"])
    ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--luong", type=int, default=4)
    ap.add_argument("--giay", type=float, default=30.0); ap.add_argument("--n-toi-thieu", type=int, default=3)
    ap.add_argument("--khoi-dong", type=int, default=20); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--S", type=int, default=34); ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--hz", type=float, default=50.0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.vai_tro == "moc":
        print("%.6f" % time.monotonic()); return
    if a.vai_tro == "lay-mau":
        lay_mau(a); return
    if os.path.exists(a.out):
        sys.exit("output exists, not overwriting")
    r = chay(a) if a.vai_tro == "chay" else khoitao(a)
    import hashlib
    r["script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    r["loadavg_sau"] = open("/proc/loadavg").read().strip()
    open(a.out, "w").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
