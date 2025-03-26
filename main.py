import sys

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.uic import loadUi
from denoising.non_local_means import Non_local_means
from PyQt5 import QtWidgets, uic
import numpy as np
from scipy import signal
import pandas as pd
# from statsmodels.nonparametric.smoothers_lowess import lowess
from PyQt5 import QtWidgets, QtGui, QtCore


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # loadUi('ecg.2ui.ui', self)
        uic.loadUi('ecg2.ui', self)

        self.upload_button = self.findChild(QPushButton, "Upload")
        self.upload_button.clicked.connect(self.upload_data)
        # fix the icon sizes
        icon_upload = QIcon("Deliveriables/upload (1).ico")
        pixmap_upload = icon_upload.pixmap(128, 128)  # Force a larger resolution from ICO file
        self.upload_button.setIcon(QIcon(pixmap_upload))
        self.upload_button.setIconSize(QSize(80, 80))


        self.HeartRateIcon = self.findChild(QPushButton,"HeartRateIcon")
        icon_heart = QIcon("Deliveriables/heartrate.ico")
        pixmap_heart = icon_heart.pixmap(128, 128)  # Force a larger resolution from ICO file
        self.HeartRateIcon.setIcon(QIcon(pixmap_heart))
        self.HeartRateIcon.setIconSize(QSize(90, 90))

        self.pressureIcon = self.findChild(QPushButton, "pressureIcon")
        icon_pressure = QIcon("Deliveriables/pressure.ico")
        pixmap_pressure = icon_pressure.pixmap(256, 256)  # Force a larger resolution from ICO file
        self.pressureIcon.setIcon(QIcon(pixmap_pressure))
        self.pressureIcon.setIconSize(QSize(160, 160))

        self.spo2Icon = self.findChild(QPushButton, "spo2Icon")
        icon_spo2 = QIcon("Deliveriables/spo2 (2).ico")
        pixmap_spo2 = icon_spo2.pixmap(256, 256)  # Force a larger resolution from ICO file
        self.spo2Icon.setIcon(QIcon(pixmap_spo2))
        self.spo2Icon.setIconSize(QSize(150, 150))

        self.temperatureIcon = self.findChild(QPushButton, "TempIcon")
        icon_temp = QIcon("Deliveriables/temperature (1).ico")
        pixmap_temp = icon_temp.pixmap(256, 256)  # Force a larger resolution from ICO file
        self.temperatureIcon.setIcon(QIcon(pixmap_temp))
        self.temperatureIcon.setIconSize(QSize(1000, 1000))

        ## Alarm Button

        self.alarmButton = self.findChild(QPushButton, "AlarmButton")
        icon_alarm = QIcon("Deliveriables/9-removebg-preview.ico")
        pixmap_alarm = icon_alarm.pixmap(256, 256)  # Force a larger resolution from ICO file
        self.alarmButton.setIcon(QIcon(pixmap_alarm))
        self.alarmButton.setIconSize(QSize(1000, 1000))



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






