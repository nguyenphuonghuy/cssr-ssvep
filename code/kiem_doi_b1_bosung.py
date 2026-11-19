#!/usr/bin/env python
"""Kiem tren Crossref cac DOI bo sung cho danh muc B1, gop voi tep kiem cu.
Dung: python kiem_doi_b1_bosung.py <kiem-doi cu json> <json ra>
Ung vien SSVEP lay tu nguon/doi_da_kiem.csv (Q1/Q2, tu 2020, co tom tat); Wolpaw (ITR) va Efron (BCa) la nguon goc phuong phap.
"""
import json, sys, time, urllib.request, urllib.parse

MOI = {
    "fbmsi": "10.1109/TNSRE.2021.3073165",
    "cf160": "10.1088/1741-2552/ac0bfa",
    "sf_transfer": "10.1088/1741-2552/ac6b57",
    "sf_transfer2": "10.1088/1741-2552/ac81ee",
    "wong_iist": "10.1109/TNSRE.2020.3019276",
    "cst_tl": "10.1109/TNSRE.2023.3250953",
    "ravi_cnn": "10.1088/1741-2552/ab6a67",
    "tfanet": "10.1109/JBHI.2024.3510740",
    "evomoe": "10.1109/JBHI.2025.3565882",
    "unsup_selfsim": "10.3389/fnins.2023.1161511",
    "fpga_bci": "10.3390/s22010318",
    "emb_tim": "10.1109/TIM.2026.3712961",
    "femba": "10.1109/TBME.2026.3683482",
    "wear_review": "10.1109/JSEN.2023.3287983",
    "gzsl_lite": "10.1109/TBME.2025.3553204",
    "wolpaw2002": "10.1016/S1388-2457(02)00057-3",
    "efron1987": "10.1080/01621459.1987.10478410",
}
out = json.load(open(sys.argv[1], encoding="utf-8"))
loi = 0
for k, d in MOI.items():
    try:
        m = json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(d),
                                                                     headers={"User-Agent": "doi-check-script/1.0"}), timeout=30))["message"]
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        out[k] = dict(doi=d, co=True, tieu_de=m["title"][0], tap_chi=(m.get("container-title") or [""])[0], nam=yr,
                      tac_gia_dau=(m.get("author") or [{}])[0].get("family", ""), tap=m.get("volume"), so=m.get("issue"), trang=m.get("page") or m.get("article-number"))
        print("OK ", k, yr, out[k]["tap_chi"][:40], "|", out[k]["tieu_de"][:80])
    except Exception as e:
        loi += 1; out[k] = dict(doi=d, co=False, loi=str(e)); print("LOI", k, d, e)
    time.sleep(0.5)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("loi", loi)
