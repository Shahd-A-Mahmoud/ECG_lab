import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.uic import loadUi
from denoising.non_local_means import Non_local_means
from PyQt5 import QtWidgets, uic
import numpy as np
from scipy import signal
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # loadUi('ecg.2ui.ui', self)
        uic.loadUi('ecg.2ui.ui', self)

        self.upload_button = self.findChild(QPushButton, "Upload")
        self.upload_button.clicked.connect(self.upload_data)

        self.data = None
        self.non_local_means = Non_local_means(self.data)



    def upload_data(self):
        pass

    def denoise_ecg_signal(self, signal, fs=500):
        # butterworth Low pass filter
        nyquist_rate = fs / 2
        passband = 50 / nyquist_rate
        # stopband = 60 / nyquist_rate
        b, a = signal.butter(4, passband, btype='low', fs=fs)
        filtered_signal = signal.filtfilt(b, a, signal)

        # LOESS --> baseline wander removal
        x = np.arange(len(filtered_signal))
        loess_fit = lowess(filtered_signal, x, frac=0.1, it=0, is_sorted=True)[:, 1]
        baseline_removed = filtered_signal - loess_fit

        # applying non_local_means
        noise_variance = 0.1
        window_size = 5
        patch_size = 3
        denoised_signal = self.non_local_means.apply_non_local_means(baseline_removed, noise_variance, window_size, patch_size)

        return denoised_signal


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())






