#!/bin/bash
# Sweep do lai CSS-R tren ZCU104. Chay trong ~/ssvep_zcu104_pkg/doi_thu:
#   nohup bash zcu104_dolai_sweep.sh dolai_20261120 > sweep_dolai_20261120.log 2>&1 &
# Thu tu: thu thong -> B phien 1 -> C khoi tao S = 99 -> D nhanh R float32 -> E CCA-Net no_grad -> B phien 2 -> nghi 30 phut -> B phien 3.
# B: 1,0 s dung mo-dun va w_R THAT (fold 17, S17; kem so quyet dinh voi HPC), 0,6 s trong so ngau nhien; 1 va 4 luong; 7 tien trinh x 200 luot.
PH=${1:?can ma phien}
PY=~/ssvep_zcu104_pkg/venv/bin/python
MD=du_lieu_ben/benchmark_cssr-mo-dun-that_fold017_T10_seed20261107_20261120.npz
cd ~/ssvep_zcu104_pkg/doi_thu
[ -e $PH ] && { echo "phien $PH da co: khong ghi de"; exit 2; }
mkdir -p $PH/B1 $PH/B2 $PH/B3 $PH/C $PH/D $PH/E $PH/thu
sha256sum zcu104_dolai.py zcu104_ccanet_nograd.py zcu104_doithu_ktnr.py zcu104_dolai_sweep.sh global_dnn.py b1_diem.py $MD trongso_T10/fold017_trongso.pt > $PH/sha.txt
(cd cca_net/goc && sha256sum -c SHA256SUMS.txt) || { echo "MA GOC CCA-NET BI SUA"; exit 3; }
export CUDA_VISIBLE_DEVICES=""
dat_luong() { export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1; }
echo "BAT DAU $(date +%H:%M:%S) $(cat /proc/loadavg)"
# thu thong
dat_luong 4
$PY zcu104_dolai.py --phan daucuoi --T 1.0 --luong 4 --mo-dun $MD --n 5 --khoi-dong 2 --out $PH/thu/daucuoi.json || { echo "THU daucuoi LOI"; exit 4; }
$PY -c "import json,sys; r=json.load(open('$PH/thu/daucuoi.json')); k=r['so_quyet_dinh_hpc']; print('khop_quyet_dinh_hpc', k); sys.exit(0 if k['so_khac']==0 else 5)" || { echo "KHONG KHOP HPC: dung"; exit 5; }
$PY zcu104_dolai.py --phan khoitao --T 0.6 --luong 4 --lap 1 --out $PH/thu/khoitao.json || { echo "THU khoitao LOI"; exit 4; }
$PY zcu104_ccanet_nograd.py --pp ccanet --T 1.0 --luong 4 --n 2 --khoi-dong 1 --out $PH/thu/ccanet.json || { echo "THU ccanet LOI"; exit 4; }
echo "THU_DAT $(date +%H:%M:%S)"

phien_B() {
  for L in 4 1; do
    dat_luong $L
    for T in 1.0 0.6; do
      for r in 1 2 3 4 5 6 7; do
        sd=$((20261107 + r)); md=""; [ $T = 1.0 ] && md="--mo-dun $MD"
        $PY zcu104_dolai.py --phan daucuoi --T $T --luong $L --seed $sd $md --out $PH/$1/daucuoi_T${T}_L${L}_r$r.json
        echo "$1 T=$T L=$L r=$r ma=$? $(date +%H:%M:%S)"; sleep 10
      done
    done
  done
}
phien_B B1
dat_luong 4
for T in 0.6 1.0; do
  for r in 1 2 3 4 5 6 7; do
    $PY zcu104_dolai.py --phan khoitao --S 99 --N 9 --T $T --luong 4 --lap 4 --seed $((20261107 + r)) --out $PH/C/khoitao_S99_T${T}_L4_r$r.json
    echo "C T=$T r=$r ma=$? $(date +%H:%M:%S)"; sleep 10
  done
done
for L in 4 1; do
  dat_luong $L
  for r in 1 2 3 4 5 6 7; do
    $PY zcu104_dolai.py --phan daucuoi --T 1.0 --luong $L --seed $((20261107 + r)) --mo-dun $MD --kieu-R f32 --out $PH/D/daucuoi_f32_T1.0_L${L}_r$r.json
    echo "D L=$L r=$r ma=$? $(date +%H:%M:%S)"; sleep 10
  done
done
for L in 4 1; do
  dat_luong $L
  for T in 1.0 0.6; do
    for r in 1 2 3 4 5 6 7; do
      $PY zcu104_ccanet_nograd.py --pp ccanet --T $T --luong $L --n 20 --khoi-dong 2 --seed $((20261107 + r)) --phien $PH --out $PH/E/ccanet_nograd_T${T}_L${L}_r$r.json
      echo "E T=$T L=$L r=$r ma=$? $(date +%H:%M:%S)"; sleep 10
    done
  done
done
phien_B B2
echo "nghi 30 phut $(date +%H:%M:%S)"; sleep 1800
phien_B B3
echo "SWEEP_DONE $(date +%H:%M:%S) $(cat /proc/loadavg)"
