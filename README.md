# CSS-R: calibration-free SSVEP decoding on embedded processors by unlabeled source-subject selection

Code and per-subject results for the manuscript "A Calibration-Free SSVEP Decoder for Embedded Processors Using Unlabeled
Source-Subject Selection" (D. N. Ton, N. P. Huy; working title; target: IEEE Transactions on Instrumentation and Measurement).

**Status: private working archive.** Comments and file names are in Vietnamese.

The code was tidied after the runs. Comments and the `ghi_chu_seed` note in `results/accuracy/ccanet/*/config.json` were
edited. The analysis and summary scripts no longer compute the internal decision labels and bookkeeping fields used
during development, and these fields were also removed from the result files. Estimates, confidence intervals and per-subject
values are unchanged: rerunning the analysis scripts on the included result files reproduces the stored analysis outputs.
Script hashes stored in the result files refer to the versions before these edits.

## Names used in the code
| Paper | Code and result files |
|---|---|
| CSS (confidence-based source selection) | `KTN`, `ktn` |
| CSS-R (CSS with a reference branch) | `KTNR`, `ktnr` |
| Base network | `tinh` |
| Base network with the reference branch | `tinh_r` |

Internal names (`ktn`, `ktnr`, ...) are kept so that result files, logs and stored script hashes stay traceable.

## Layout
- `code/`: decoders, reproductions of the comparison methods, analysis scripts, MAC counting and ZCU104 timing.
  - `ktnr_hoanvi*.py`: trial-order sensitivity analysis.
  - `doi_thu_*.py`, `rcs_*.py`, `ensemble_dnn*.py`, `cca_net_*.py`, `fbcca_*.py`: reproductions of the comparison methods.
    The CCA-Net driver runs the official CCA-Net code (commit 1660d42), which is not included; obtain it from the original repository.
  - `cca1_bon_bo.py`, `oacca_bon_bo.py`, `gan_ngan_sach_phan_tich.py`: single-band CCA and OACCA (5 and 1 subbands) under the
    protocol of the paper, paired against CSS-R (`results/accuracy/gan_ngan_sach/`).
  - `ktnr_chon_nguon_dc*.py`: other source-selection rules (`results/accuracy/chon_nguon_dc/`).
  - Filtering protocol (`results/accuracy/loc_nhan_qua/`): `loc_nhan_qua_phan_ra.py` runs FBCCA, single-band CCA and OACCA
    with causal filtering, zero-phase filtering of the buffer up to the decision time (`pk`), and zero-phase filtering with
    look-ahead (`pk_nt`). It also varies pre-roll and window shift (`phan_ra/`, summary by `loc_nhan_qua_cap.py`,
    regression check `loc_nhan_qua_kiem.py`). `chay_loc.py` runs the CSS-R scripts with `pk` filtering without modifying them
    (`loc_pk_cssr.pbs`, check `loc_pk_kiem_hoi_quy.py`), and `loc_pk_cssr_cap.py` pairs the results per subject (`cssr_pk/`).
    The base network is retrained for `pk`; GPU training is not bit-exact.
  - Reproduction checks against published values: `doi_thu_oacca.py --che-do goc` with `oacca_tai_hien.py` for OACCA
    (`results/accuracy/oacca_goc/`), `fbcca_goc.py` for FBCCA under the dataset papers' protocols (`results/accuracy/fbcca_goc/`),
    and `rcs_dong_chan_doan.py` for the source sets of RCS on Dong 2023 (`results/accuracy/rcs_dong/`).
  - `zcu104_*.py`, `zcu104_*.sh`: timing measurement on the ZCU104 processing system. `zcu104_dolai*` repeats the end-to-end
    timing of CSS-R (`results/zcu104/dolai_20261120/`), `zcu104_nangluong*` measures energy per decision
    (`results/zcu104/nangluong_20261120/`), and `zcu104_oacca*` times OACCA (`results/zcu104/oacca_20261121/`).
- `results/accuracy/`: per-subject JSON results and logs (seed 20261107) and analysis outputs.
- `results/zcu104/`: raw timing records (JSON lines) and summaries. `doi-thu-b1/phien_doithu_20261118/` is the joint session with the
  comparison methods (session log, 168 timing files, 12 check files).
- `SHA256SUMS.txt`: checksums of every file in this archive.

No EEG data, network weights or logits are included.

## Data
Benchmark, BETA, eldBETA, Nakanishi 2015 and Dong 2023 are public; see the manuscript for references and licenses.

## License
MIT for the authors' code and the result files, except the files listed in `NOTICE`, which are ports of GPL-3.0 code and remain under GPL-3.0.
