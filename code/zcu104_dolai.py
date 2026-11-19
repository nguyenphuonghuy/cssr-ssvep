#!/usr/bin/env python
"""Do lai CSS-R tren PS cua ZCU104 kem do bat dinh (phan B, C, D).
Chay trong ~/ssvep_zcu104_pkg/doi_thu (can global_dnn.py, b1_diem.py, du_lieu_ben/, trongso_T10/). In MOT dong JSON ra --out.
  --phan daucuoi : nhu zcu104_ktnr_daucuoi.py (dong ho bat dau khi luot tho 9 x 1500 da o bo nho: tru TB truoc kich thich -> bank 3 dai sosfilt
                   -> mang da gop (no_grad) -> nhanh R -> z gop -> argmax), them: luu thoi gian TUNG luot, thong tin nen tang,
                   --kieu-R f64|f32 (phan D), --mo-dun <npz> (phan B: mo-dun gop va w_R THAT cua fold 17, du lieu S17, trong so fold017;
                   chi 1,0 s; them buoc so quyet dinh luot 41..240 theo thu tu tep voi quyet dinh HPC).
  --phan khoitao : luot khoi tao nhu zcu104_ktnr_dotre.py (mang nen quyet dinh + cham S mo-dun + cong don do tin; luot 40 them TB 5 mo-dun
                   va w_R), lap --lap lan 40 luot trong mot tien trinh, luu TUNG luot (phan C). Khong gom bank loc (nhu ban cu).
"""
import argparse, glob, json, os, platform, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.signal import sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FS, ALPHA, H = 250, 0.5, 5


def co_so(freqs, fs, ns):
    t = np.arange(ns) / fs
    return np.stack([np.linalg.qr(np.stack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)], 1))[0] for f in freqs])


def diem_R1(yv, Qb):
    p = np.einsum("t,ntk->nk", yv, Qb)
    ny = np.linalg.norm(yv)
    return np.sqrt((p ** 2).sum(-1)) / ny if ny > 0 else np.zeros(Qb.shape[0], dtype=yv.dtype)


def z_np(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def w_tu(ws):
    M = sum(np.outer(w, w) for w in ws) / len(ws)
    lam, V = np.linalg.eigh(M)
    m = lam.max()
    return V[:, int(np.where(lam >= m - 1e-12 * max(1.0, abs(m)))[0][0])]


def do_tin(lg):
    v, _ = torch.sort(lg, 1, descending=True)
    r = v[:, 1:]
    return (v[:, 0] - r.mean(1)) / (r.std(1) + 1e-12)


def doc(p):
    try:
        return open(p).read().strip()
    except Exception as e:
        return "loi: %s" % type(e).__name__


def nen_tang():
    cf = "/sys/devices/system/cpu/cpu%d/cpufreq/"
    v = dict(uname=" ".join(platform.uname()), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
             loadavg=doc("/proc/loadavg"), os_release=doc("/etc/os-release").splitlines()[:2],
             governor=[doc(cf % i + "scaling_governor") for i in range(4)], tan_so_khz=[doc(cf % i + "scaling_cur_freq") for i in range(4)],
             tan_so_co_the=doc(cf % 0 + "scaling_available_frequencies"),
             dong_ho_perf_counter=vars(time.get_clock_info("perf_counter")),
             bien_moi_truong={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
             torch_luong=torch.get_num_threads(), torch_luong_interop=torch.get_num_interop_threads(),
             torch_cau_hinh=torch.__config__.parallel_info().splitlines()[:6])
    nd = {}
    for p in glob.glob("/sys/bus/iio/devices/iio:device*/in_temp*"):
        nd[os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)] = doc(p)
    v["nhiet_do_iio"] = nd
    try:
        from threadpoolctl import threadpool_info
        v["blas"] = [{k: d.get(k) for k in ("internal_api", "version", "num_threads", "threading_layer", "filepath")} for d in threadpool_info()]
    except Exception as e:
        v["blas"] = "loi: %s" % e
    return v


def q(v):
    v = np.asarray(v, float)
    return dict(tb_ms=1e3 * float(v.mean()), sd_ms=1e3 * float(v.std(ddof=1)), trung_vi_ms=1e3 * float(np.median(v)),
                p95_ms=1e3 * float(np.percentile(v, 95)), p99_ms=1e3 * float(np.percentile(v, 99)), max_ms=1e3 * float(v.max()), n=int(len(v)))


def sau(m, h1, W, b):
    return m.fc(m.l4(torch.relu(m.l3(Fn.conv2d(h1, W, b)))).flatten(1))


def daucuoi(a, GD, B):
    ten = "benchmark_S17" if a.mo_dun else "benchmark_S1"
    z = np.load("du_lieu_ben/%s.npz" % ten); Xr = z["X"]; freqs = z["freqs"]
    c = B.DS["benchmark"]; on = c["onset"]; st = on + int(round(c["lat"] * FS)); ns = int(round(a.T * FS)); C, N = 9, 40
    rng = np.random.default_rng(a.seed)
    _, sos = GD.bo_loc(FS)
    m = GD.GlobalDNN(C, ns, N, nch=22)
    nap_that = a.T == 1.0
    if nap_that:
        m.load_state_dict(torch.load("trongso_T10/fold017_trongso.pt", map_location="cpu"))
    m.eval()
    dt = np.float32 if a.kieu_R == "f32" else np.float64
    with torch.no_grad():
        if a.mo_dun:
            md = np.load(a.mo_dun)
            m.l2.weight += torch.from_numpy(md["dW"]).to(m.l2.weight.dtype); m.l2.bias += torch.from_numpy(md["db"]).to(m.l2.bias.dtype)
            wR = md["wR"].astype(dt)
        else:
            m.l2.weight += 0.01 * torch.randn_like(m.l2.weight); m.l2.bias += 0.01 * torch.randn_like(m.l2.bias)
            w = rng.standard_normal(C); wR = (w / np.linalg.norm(w)).astype(dt)
    Qb = co_so(freqs, FS, ns).astype(dt)

    def quyet(xr):
        xb = xr[:, : st + ns] - xr[:, :on].mean(axis=1, keepdims=True)
        u = np.empty((len(sos), C, ns), np.float32)
        for r, so in enumerate(sos):
            u[r] = sosfilt(so, xb, axis=1)[:, st: st + ns]
        t1 = time.perf_counter()
        h1 = m.l1(torch.from_numpy(u)[None])
        lg = m.fc(m.l4(torch.relu(m.l3(m.l2(h1)))).flatten(1))
        U = h1[0, 0].numpy().astype(dt)
        yv = wR @ U; yv = yv - yv.mean()
        s = z_np(lg.numpy()[0].astype(dt)) + ALPHA * z_np(diem_R1(yv, Qb))
        return int(s.argmax()), t1

    out = dict(phan="daucuoi", T=a.T, luong=a.luong, seed=a.seed, kieu_R=a.kieu_R, mo_dun=os.path.basename(a.mo_dun) if a.mo_dun else None,
               du_lieu=ten, trong_so_nen_that=nap_that, n=a.n, khoi_dong=a.khoi_dong, nen_tang_truoc=nen_tang())
    tb, tl, tt = [], [], []
    with torch.no_grad():
        if a.mo_dun:
            ref = np.load(a.mo_dun)
            idx = ref["chi_so"] - 1
            d = np.array([quyet(Xr[i])[0] for i in idx])
            out["so_quyet_dinh_hpc"] = dict(n=int(len(idx)), so_khac=int((d != ref["quyet_dinh"]).sum()),
                                            do_chinh_xac_board=float((d == ref["y"]).mean()), do_chinh_xac_hpc=float((ref["quyet_dinh"] == ref["y"]).mean()))
        for i in range(a.khoi_dong + a.n):
            xr = Xr[i % len(Xr)]
            t0 = time.perf_counter(); _, t1 = quyet(xr); t2 = time.perf_counter()
            if i >= a.khoi_dong:
                tb.append(t1 - t0); tl.append(t2 - t1); tt.append(t2 - t0)
    out.update(bank_loc=q(tb), loi=q(tl), tong=q(tt), tung_luot_tong_ms=[round(1e3 * v, 4) for v in tt], nen_tang_sau=dict(loadavg=doc("/proc/loadavg")))
    return out


def khoitao(a, GD):
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    C, ns, N, S = 9, int(round(a.T * FS)), a.N, a.S
    m = GD.GlobalDNN(C, ns, N, nch=22).eval()
    dW = [0.01 * torch.randn_like(m.l2.weight) for _ in range(S)]; db = [0.01 * torch.randn_like(m.l2.bias) for _ in range(S)]
    Ws = [m.l2.weight + d for d in dW]; bs = [m.l2.bias + d for d in db]
    wsrc = [v / np.linalg.norm(v) for v in rng.standard_normal((S, C))]
    X = torch.randn(40, 1, 3, C, ns)
    out = dict(phan="khoitao", S=S, N=N, T=a.T, luong=a.luong, seed=a.seed, lap=a.lap, nen_tang_truoc=nen_tang())
    tk = []
    with torch.no_grad():
        for _ in range(5):
            m(X[0]); sau(m, m.l1(X[0]), Ws[0], bs[0])
        for _ in range(a.lap):
            acc = torch.zeros(S)
            for i in range(40):
                t0 = time.perf_counter()
                x = X[i]
                int(m(x).argmax(1))
                h1 = m.l1(x)
                for s in range(S):
                    acc[s] += do_tin(sau(m, h1, Ws[s], bs[s]))[0]
                if i == 39:
                    top = torch.argsort(-acc, stable=True)[:5]
                    torch.stack([dW[int(j)] for j in top]).mean(0); torch.stack([db[int(j)] for j in top]).mean(0)
                    w_tu([wsrc[int(j)] for j in top])
                tk.append(time.perf_counter() - t0)
    v = np.asarray(tk)
    out.update(khoi_tao=q(v), so_luot_vuot_500ms=int((v > 0.5).sum()), so_luot_vuot_400ms=int((v > 0.4).sum()),
               tung_luot_ms=[round(1e3 * x, 3) for x in tk], nen_tang_sau=dict(loadavg=doc("/proc/loadavg")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phan", required=True, choices=["daucuoi", "khoitao"])
    ap.add_argument("--T", type=float, required=True); ap.add_argument("--luong", type=int, required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--khoi-dong", type=int, default=20)
    ap.add_argument("--kieu-R", default="f64", choices=["f64", "f32"]); ap.add_argument("--mo-dun", default="")
    ap.add_argument("--S", type=int, default=99); ap.add_argument("--N", type=int, default=9); ap.add_argument("--lap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261107); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("da co: khong ghi de")
    if a.mo_dun and a.T != 1.0:
        sys.exit("mo-dun that chi co o 1,0 s")
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); np.random.seed(a.seed)
    import global_dnn as GD
    import b1_diem as B
    r = daucuoi(a, GD, B) if a.phan == "daucuoi" else khoitao(a, GD)
    import hashlib
    r["script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    open(a.out, "w").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
