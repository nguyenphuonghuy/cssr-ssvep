#!/usr/bin/env python
"""Tong hop do lai CSS-R tren ZCU104. Doc <thu muc phien>/{B1,B2,B3,C,D,E}/*.json,
ghi zcu104_dolai_tong-hop_seed20261107_20261120.json canh thu muc phien. Khong ghi de.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

P = sys.argv[1]
OUT = os.path.join(os.path.dirname(P.rstrip("/\\")), "zcu104_dolai_tong-hop_seed20261107_20261120.json")
CU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "zcu104", "doi-thu-b1", "phien_doithu_20261118", "do")


def ci(v):
    v = np.asarray(v, float); k = len(v); m = float(v.mean()); sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.975, k - 1) * sd / np.sqrt(k))
    return dict(k=k, tb_ms=m, sd_ms=sd, ci95_ms=[m - h, m + h], nua_ci_phan_tram=100 * h / m)


def doc(pat):
    return [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(P, pat)))]


def main():
    if os.path.exists(OUT):
        sys.exit("da co: khong ghi de")
    kq = {"daucuoi": {}, "khop_quyet_dinh_hpc": {}}
    for T in (1.0, 0.6):
        for L in (4, 1):
            k = "T%.1f_L%d" % (T, L)
            ph, tung, loi = {}, [], {}
            for s in ("B1", "B2", "B3"):
                rs = doc("%s/daucuoi_T%.1f_L%d_r*.json" % (s, T, L))
                if not rs:
                    continue
                ph[s] = ci([r["tong"]["tb_ms"] for r in rs])
                loi[s] = ci([r["loi"]["tb_ms"] for r in rs])
                ph[s]["max_luot_ms"] = max(r["tong"]["max_ms"] for r in rs)
                tung += [x for r in rs for x in r["tung_luot_tong_ms"]]
                if T == 1.0:
                    kq["khop_quyet_dinh_hpc"].setdefault(k, []).extend(r["so_quyet_dinh_hpc"]["so_khac"] for r in rs)
            tb_phien = [ph[s]["tb_ms"] for s in ph]
            tat_ca = [r["tong"]["tb_ms"] for s in ph for r in doc("%s/daucuoi_T%.1f_L%d_r*.json" % (s, T, L))]
            tu = np.asarray(tung)
            kq["daucuoi"][k] = dict(theo_phien=ph, loi_theo_phien=loi, gop_tien_trinh=ci(tat_ca),
                                       giua_phien=dict(tb_ms=float(np.mean(tb_phien)), sd_ms=float(np.std(tb_phien, ddof=1)) if len(tb_phien) > 1 else None,
                                                       min_ms=min(tb_phien), max_ms=max(tb_phien)),
                                       tung_luot=dict(n=int(tu.size), p50=float(np.percentile(tu, 50)), p95=float(np.percentile(tu, 95)),
                                                      p99=float(np.percentile(tu, 99)), max=float(tu.max()), vuot_10ms=int((tu > 10).sum())))
    kq["khop_quyet_dinh_hpc"] = {k: dict(so_lan=len(v), tong_so_khac=int(sum(v))) for k, v in kq["khop_quyet_dinh_hpc"].items()}
    kq["khoitao"] = {}
    for T in (0.6, 1.0):
        rs = doc("C/khoitao_S99_T%.1f_L4_r*.json" % T)
        tu = np.asarray([x for r in rs for x in r["tung_luot_ms"]])
        kq["khoitao"]["S99_T%.1f_L4" % T] = dict(tien_trinh=len(rs), n=int(tu.size), tb=float(tu.mean()), p50=float(np.percentile(tu, 50)),
                                                    p95=float(np.percentile(tu, 95)), p99=float(np.percentile(tu, 99)), max=float(tu.max()),
                                                    vuot_400ms=int((tu > 400).sum()), vuot_500ms=int((tu > 500).sum()),
                                                    tb_tien_trinh=ci([r["khoi_tao"]["tb_ms"] for r in rs]))
    kq["f32"] = {}
    for L in (4, 1):
        f32 = doc("D/daucuoi_f32_T1.0_L%d_r*.json" % L); f64 = doc("B1/daucuoi_T1.0_L%d_r*.json" % L)
        a = [r["loi"]["tb_ms"] for r in f64]; b = [r["loi"]["tb_ms"] for r in f32]
        kq["f32"]["T1.0_L%d" % L] = dict(loi_f64=ci(a), loi_f32=ci(b), giam_ms=float(np.mean(a) - np.mean(b)),
                                            tong_f64=ci([r["tong"]["tb_ms"] for r in f64]), tong_f32=ci([r["tong"]["tb_ms"] for r in f32]),
                                            so_khac_hpc_f32=int(sum(r["so_quyet_dinh_hpc"]["so_khac"] for r in f32)))
    kq["ccanet_no_grad"] = {}
    for T in (1.0, 0.6):
        for L in (4, 1):
            ng = doc("E/ccanet_nograd_T%.1f_L%d_r*.json" % (T, L))
            cu = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(CU, "ccanet_T%.1f_L%d_r*.json" % (T, L))))]
            a = [r["tong"]["tb_ms"] for r in cu]; b = [r["tong"]["tb_ms"] for r in ng]
            cs = kq["daucuoi"]["T%.1f_L%d" % (T, L)]["gop_tien_trinh"]["tb_ms"]
            kq["ccanet_no_grad"]["T%.1f_L%d" % (T, L)] = dict(autograd_phien_cu=ci(a), no_grad=ci(b), giam_phan_tram=100 * (1 - np.mean(b) / np.mean(a)),
                                                        ti_le_no_grad_so_cssr_gop=float(np.mean(b) / cs), duoi_500ms=bool(max(r["tong"]["max_ms"] for r in ng) <= 500))
    kq["nen_tang"] = {k: doc("B1/daucuoi_T1.0_L4_r1.json")[0]["nen_tang_truoc"][k] for k in ("governor", "tan_so_khz", "dong_ho_perf_counter", "blas", "uname", "nhiet_do_iio")}
    json.dump(kq, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in kq.items() if k != "nen_tang"}, indent=1, ensure_ascii=False)[:9000])


if __name__ == "__main__":
    main()
