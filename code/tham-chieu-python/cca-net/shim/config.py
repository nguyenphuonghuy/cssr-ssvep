# Thay cho goc/config.py (ban goc gan cung Dataset = 2 va duong dan /home/wc/...): doc bo va cua so tu bien moi truong CN_DS, CN_T.
# Moi hang so khac giu NGUYEN nhu goc/config.py (commit 1660d42): Nh = 4, Nm = 3, n_epochs = 100, batch_size = 128, dropout 0,95, list_phase viet cung.
import os
import numpy as np
from scipy.io import loadmat

GOC = "/gpfs0/users/gv_dnton/datasets/ssvep"
BETA = "/gpfs0/users/gv_nphuy/datasets/BETA/raw"          # cung nguon voi KTN (b1_diem.DS["beta"])


class Config(object):
    def __init__(self):
        self.Dataset = int(os.environ["CN_DS"])          # 0 UCSD (Nakanishi 2015), 1 Benchmark, 2 BETA
        self.LOOC = 0
        self.sample_length = float(os.environ["CN_T"])
        self.Nh = 4
        self.Nm = 3
        self.n_epochs = 100
        self.batch_size = 128
        if self.Dataset == 0:
            self.C = 8
            self.rfs = 256
            self.dropout = 0.95
            self.num_class = 12
            self.nBlock = 15
            self.list_freqs = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]).T
            self.list_phase = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5]) * np.pi
        if self.Dataset == 1:
            self.C = 9
            self.rfs = 250
            self.dropout = 0.95
            self.num_class = 40
            self.nBlock = 6
            self.list_freqs = loadmat(os.path.join(GOC, "benchmark", "Freq_Phase.mat"))['freqs'][0]
            self.list_phase = np.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5]) * np.pi
        if self.Dataset == 2:
            self.C = 9
            self.rfs = 250
            self.dropout = 0.95
            self.num_class = 40
            self.nBlock = 4
            # ban goc doc Freqs_Beta.mat (khong co trong ban phat hanh BETA); thay bang data.suppl_info.freqs cua S1 (cung mot bang cho moi nguoi)
            si = loadmat(os.path.join(BETA, "S1.mat"))["data"][0][0][1]
            self.list_freqs = np.asarray(si["freqs"][0][0]).ravel()
            self.list_phase = np.array([1.5, 0, 0.5, 1] * 10) * np.pi  # viet cung nhu goc (lech -0,5 pi so voi tep)
        if self.Dataset == 3:                                             # Dong 2023 (them 2026-11-17; ma goc khong co bo nay)
            self.C = 8
            self.rfs = 250
            self.dropout = 0.95
            self.num_class = 40
            self.nBlock = 4
            self.list_freqs = 8.0 + 0.2 * np.arange(40)                   # hinh 1b cua bai Dong 2023
            self.list_phase = 0.5 * (np.arange(40) % 4) * np.pi
        self.T = int(self.sample_length * self.rfs)                       # goc: cat bo phan le
        if os.environ.get("CN_CHE") == "gt":
            self.T = int(round(self.sample_length * self.rfs))           # giao thuc B1: ns = round(T fs) (chi khac o 256 Hz, 0,6 s: 154 thay 153)
        self.save_path = os.path.abspath(os.curdir) + '/result/'


config = Config()
