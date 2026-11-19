#!/usr/bin/env python
"""Phan E: do CCA-Net dau-cuoi trong torch.no_grad, khong sua zcu104_doithu_ktnr.py.
Thay contextlib.nullcontext trong module do bang torch.no_grad (chi dong 419 dung nullcontext cho CCA-Net), roi goi main() voi cung tham so.
Ban ghi them truong che_do_grad = "no_grad".
"""
import json, os, sys, types
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zcu104_doithu_ktnr as D  # noqa: E402

D.contextlib = types.SimpleNamespace(nullcontext=torch.no_grad)
if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1]
    D.main()
    r = json.loads(open(out).read())
    r["che_do_grad"] = "no_grad"; r["wrapper"] = os.path.basename(__file__)
    open(out, "w").write(json.dumps(r) + "\n")
