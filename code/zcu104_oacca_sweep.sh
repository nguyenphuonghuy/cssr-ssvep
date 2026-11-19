#!/bin/bash
# OACCA on the ZCU104. Run in ~/ssvep_zcu104_pkg/doi_thu:
#   nohup bash zcu104_oacca_sweep.sh oacca_<date> > sweep_<ma>.log 2>&1 &
# 1) decision check against the HPC run (Benchmark subject 1, 240 trials, both windows, 5 and 1 subbands): stop on any mismatch.
# 2) timing: OACCA 5 subbands at 1.0 / 0.6 s with 4 / 1 threads, OACCA 1 subband and CSS-R (same-session reference) at 1.0 s, 4 threads;
#    7 processes each, 10 s cooldown.
# 3) energy (sampler 50 Hz, 30 s idle + 30 s loop, 3 repetitions): oacca5 L4, oacca5 L1, oacca1 L4, ktnr L4.
PH=${1:?can ma phien}
PY=~/ssvep_zcu104_pkg/venv/bin/python
cd ~/ssvep_zcu104_pkg/doi_thu
[ -e $PH ] && { echo "phien $PH da co: khong ghi de"; exit 2; }
mkdir -p $PH/kiem $PH/gio $PH/chay
sha256sum zcu104_oacca.py zcu104_oacca_sweep.sh zcu104_nangluong.py doi_thu_oacca.py zcu104_doithu_ktnr.py du_lieu_ben/benchmark_S1.npz ref_oacca/*.json > $PH/sha.txt
export CUDA_VISIBLE_DEVICES=""
dat_luong() { export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1; }
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq > $PH/cpufreq.txt
echo "BAT DAU $(date +%H:%M:%S) $(cat /proc/loadavg)"
dat_luong 4
for nb in 5 1; do for T in 1.0 0.6; do
  $PY zcu104_oacca.py --T $T --luong 4 --so-dai $nb --ref ref_oacca/benchmark_oacca${nb}_nq.json --out $PH/kiem/oacca${nb}_T$T.json || { echo "KIEM LOI"; exit 4; }
  $PY -c "import json,sys; r=json.load(open('$PH/kiem/oacca${nb}_T$T.json')); k=r['so_quyet_dinh_hpc']; print('KIEM oacca$nb T$T', k, r['tong']['tb_ms']); sys.exit(0 if k['so_khac']==0 else 5)" || { echo "KIEM KHONG DAT: dung"; exit 5; }
done; done
echo "KIEM_DAT $(date +%H:%M:%S)"
for L in 4 1; do
  dat_luong $L
  for T in 1.0 0.6; do for r in 1 2 3 4 5 6 7; do
    $PY zcu104_oacca.py --T $T --luong $L --so-dai 5 --seed $((20261107 + r)) --out $PH/gio/oacca5_T${T}_L${L}_r$r.json
    echo "gio oacca5 T=$T L=$L r=$r ma=$? $(date +%H:%M:%S)"; sleep 10
  done; done
done
dat_luong 4
for r in 1 2 3 4 5 6 7; do
  $PY zcu104_oacca.py --T 1.0 --luong 4 --so-dai 1 --seed $((20261107 + r)) --out $PH/gio/oacca1_T1.0_L4_r$r.json
  echo "gio oacca1 r=$r ma=$? $(date +%H:%M:%S)"; sleep 10
  $PY zcu104_doithu_ktnr.py --pp ktnr --T 1.0 --luong 4 --seed $((20261107 + r)) --out $PH/gio/ktnr_T1.0_L4_r$r.json
  echo "gio ktnr r=$r ma=$? $(date +%H:%M:%S)"; sleep 10
done
echo "GIO_XONG $(date +%H:%M:%S)"
$PY zcu104_nangluong.py --vai-tro lay-mau --hz 50 --out $PH/mau_cong_suat.txt &
LM=$!
trap 'kill $LM 2>/dev/null' EXIT
mot() {
  local ten=$1; shift
  echo "nghi_dau $ten $($PY zcu104_nangluong.py --vai-tro moc)" >> $PH/moc.txt
  sleep 30
  echo "nghi_cuoi $ten $($PY zcu104_nangluong.py --vai-tro moc)" >> $PH/moc.txt
  $PY zcu104_nangluong.py "$@" --out $PH/chay/$ten.json
  echo "$ten ma=$? $(date +%H:%M:%S) $(cat /proc/loadavg)"
  sleep 10
}
for r in 1 2 3; do
  dat_luong 4; mot oacca5_L4_r$r --vai-tro chay --pp oacca5 --T 1.0 --luong 4 --giay 30 --seed $((20261107 + r))
  dat_luong 1; mot oacca5_L1_r$r --vai-tro chay --pp oacca5 --T 1.0 --luong 1 --giay 30 --seed $((20261107 + r))
  dat_luong 4; mot oacca1_L4_r$r --vai-tro chay --pp oacca1 --T 1.0 --luong 4 --giay 30 --seed $((20261107 + r))
  dat_luong 4; mot ktnr_L4_r$r --vai-tro chay --pp ktnr --T 1.0 --luong 4 --giay 30 --seed $((20261107 + r))
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq >> $PH/cpufreq.txt
echo "SWEEP_DONE $(date +%H:%M:%S)"
