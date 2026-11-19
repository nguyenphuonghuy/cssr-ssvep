#!/usr/bin/env python
"""Chay mot script cua chuoi CSS-R (global_dnn_huanluyen.py, ktnr_buoc1.py, ...) voi tien xu ly da doi, khong sua tep nao.
Thay global_dnn.tien_xu_ly khi chay:
  nq : giu nguyen (loc nhan qua sosfilt tu mau 0 sau khi tru TB doan truoc kich thich), dung de kiem hoi quy tep boc.
  pk : cung tru TB, cung day loc ba dai cua global_dnn.bo_loc, nhung loc pha khong (sosfiltfilt) tren bo dem [0, dau cua so + ns),
       cat [dau cua so, + ns). Chi dung mau den thoi diem quyet dinh.
Ghi tep danh dau <dich>_tien-xu-ly.json (che do, sha256 tep boc, script dich) canh dau ra (--out cua script dich).
Dung: python chay_loc.py --che-do nq|pk -- <script.py> <tham so cua script>
"""
import hashlib, json, os, runpy, sys
import numpy as np
from scipy.signal import sosfilt, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))


def tao_pk(GD):
    def tien_xu_ly_pk(X, ds, loc, ns, fs):
        assert loc == "causal", loc
        c = GD.B.DS[ds]; on = c["onset"]; st = on + int(round(c["lat"] * fs))
        ba, sos = GD.bo_loc(fs)
        out = np.empty((X.shape[0], GD.NSB, X.shape[1], ns), np.float32)
        Xb = X[:, :, : st + ns] - X[:, :, :on].mean(axis=2, keepdims=True)
        for r, so in enumerate(sos):
            out[:, r] = sosfiltfilt(so, Xb, axis=2)[:, :, st: st + ns]
        return out
    return tien_xu_ly_pk


def main():
    i = sys.argv.index("--")
    che_do = sys.argv[sys.argv.index("--che-do") + 1]
    assert che_do in ("nq", "pk")
    dich = os.path.abspath(sys.argv[i + 1]); targs = sys.argv[i + 2:]
    sys.path.insert(0, os.path.dirname(dich))
    import global_dnn as GD   # noqa: E402  (cung doi tuong module ma script dich se import)
    if che_do == "pk":
        GD.tien_xu_ly = tao_pk(GD)
    out = targs[targs.index("--out") + 1]
    dau = (out.rstrip("/\\") if os.path.splitext(out)[1] == "" else os.path.splitext(out)[0]) + "_tien-xu-ly.json"
    os.makedirs(os.path.dirname(os.path.abspath(dau)), exist_ok=True)
    json.dump(dict(che_do=che_do, tep_boc_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                   script=os.path.basename(dich), script_sha256=hashlib.sha256(open(dich, "rb").read()).hexdigest(), tham_so=targs),
              open(dau, "w"), indent=1)
    sys.argv = [dich] + targs
    runpy.run_path(dich, run_name="__main__")


if __name__ == "__main__":
    main()
