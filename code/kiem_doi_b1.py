#!/usr/bin/env python
"""Kiem DOI tai lieu trich trong dan y B1 tren Crossref (api.crossref.org/works/<doi>). In tieu de, tap chi, nam, tac gia dau.
Dung: python kiem_doi_b1.py <json ra>
"""
import json, sys, time, urllib.request, urllib.parse

DOI = {
    "global_dnn": "10.1109/TBME.2021.3110440",
    "ensemble_dnn": "10.1088/1741-2552/acacca",
    "cca_net": "10.1109/TIM.2025.3571139",
    "larsnet": "10.1109/TIM.2026.3660428",
    "rcs": "10.1109/TNSRE.2023.3245654",
    "ss_itrca": "10.1109/JBHI.2025.3577813",
    "oaeca": "10.1109/JBHI.2025.3644250",
    "oacca": "10.1109/TBME.2021.3133594",
    "dtw_tta": "10.1109/TBME.2026.3692973",
    "duan2026low": "10.3390/app16083799",
    "unified_online_adapt": "10.1109/JBHI.2026.3689883",
    "tt_cca": "10.1088/1741-2560/12/4/046006",
    "fbcca": "10.1088/1741-2560/12/4/046008",
    "trca": "10.1109/TBME.2017.2694818",
    "benchmark": "10.1109/TNSRE.2016.2627556",
    "beta": "10.3389/fnins.2020.00627",
    "eldbeta": "10.1038/s41597-022-01372-9",
    "nakanishi2015": "10.1371/journal.pone.0140703",
    "dong2023": "10.26599/BSA.2023.9050020",
}
out = {}
for k, d in DOI.items():
    url = "https://api.crossref.org/works/" + urllib.parse.quote(d)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "doi-check-script/1.0"})
        m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        au = m.get("author", [{}])[0]
        out[k] = dict(doi=d, co=True, tieu_de=(m.get("title") or [""])[0], tap_chi=(m.get("container-title") or [""])[0], nam=yr,
                      tac_gia_dau=au.get("family", ""), tap=m.get("volume"), so=m.get("issue"), trang=m.get("page"))
    except Exception as ex:
        out[k] = dict(doi=d, co=False, loi=str(ex)[:120])
    print(k, json.dumps(out[k], ensure_ascii=False))
    time.sleep(0.5)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
