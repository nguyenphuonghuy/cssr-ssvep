#!/bin/bash
# Energy session on the ZCU104. Run in ~/ssvep_zcu104_pkg/doi_thu:
#   nohup bash zcu104_nangluong_sweep.sh nangluong_<date> > sweep_<ma>.log 2>&1 &
# One sampler (50 Hz) runs for the whole session. For each configuration: 30 s idle marked by two "moc" calls, then a 30 s
# back-to-back decision loop (CCA-Net 60 s). Configurations: 1.0 s window, 4 and 1 threads, methods ktnr fbcca rcs ed22 edgoc ccanet
# and CSS-R initialization (S = 34, 40 classes), 3 repetitions in rotated order, 10 s cooldown.
PH=${1:?can ma phien}
PY=~/ssvep_zcu104_pkg/venv/bin/python
cd ~/ssvep_zcu104_pkg/doi_thu
[ -e $PH ] && { echo "phien $PH da co: khong ghi de"; exit 2; }
mkdir -p $PH/chay $PH/thu
sha256sum zcu104_nangluong.py zcu104_nangluong_sweep.sh zcu104_doithu_ktnr.py zcu104_dolai.py global_dnn.py b1_diem.py > $PH/sha.txt
(cd cca_net/goc && sha256sum -c SHA256SUMS.txt) || { echo "MA GOC CCA-NET BI SUA"; exit 3; }
export CUDA_VISIBLE_DEVICES=""
dat_luong() { export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1; }
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq > $PH/cpufreq.txt
$PY zcu104_nangluong.py --vai-tro lay-mau --hz 50 --out $PH/mau_cong_suat.txt &
LM=$!
trap 'kill $LM 2>/dev/null' EXIT
echo "BAT DAU $(date +%H:%M:%S) $(cat /proc/loadavg)"
# thu thong (ngan)
dat_luong 4
for pp in ktnr fbcca; do
  $PY zcu104_nangluong.py --vai-tro chay --pp $pp --giay 2 --out $PH/thu/$pp.json || { echo "THU $pp LOI"; exit 4; }
done
$PY zcu104_nangluong.py --vai-tro khoitao --giay 2 --out $PH/thu/khoitao.json || { echo "THU khoitao LOI"; exit 4; }
echo "THU_DAT $(date +%H:%M:%S)"
mot() {  # $1 = ten, $2.. = tham so
  local ten=$1; shift
  echo "nghi_dau $ten $($PY zcu104_nangluong.py --vai-tro moc)" >> $PH/moc.txt
  sleep 30
  echo "nghi_cuoi $ten $($PY zcu104_nangluong.py --vai-tro moc)" >> $PH/moc.txt
  $PY zcu104_nangluong.py "$@" --out $PH/chay/$ten.json
  echo "$ten ma=$? $(date +%H:%M:%S) $(cat /proc/loadavg)"
  sleep 10
}
DS=(ktnr fbcca rcs ed22 edgoc ccanet khoitao)
for r in 1 2 3; do
  for L in 4 1; do
    dat_luong $L
    for k in 0 1 2 3 4 5 6; do
      pp=${DS[$(( (k + r - 1) % 7 ))]}
      g=30; [ $pp = ccanet ] && g=60
      if [ $pp = khoitao ]; then
        mot khoitao_L${L}_r$r --vai-tro khoitao --T 1.0 --luong $L --S 34 --N 40 --giay $g --seed $((20261107 + r))
      else
        mot ${pp}_L${L}_r$r --vai-tro chay --pp $pp --T 1.0 --luong $L --giay $g --seed $((20261107 + r))
      fi
    done
  done
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq >> $PH/cpufreq.txt
echo "SWEEP_DONE $(date +%H:%M:%S)"
