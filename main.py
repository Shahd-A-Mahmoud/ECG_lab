
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PyQt5.uic import loadUi
from denoising.non_local_means import Non_local_means
import numpy as np
import scipy.signal as sig  # Renamed import to avoid conflict
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
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
        self.non_local_means = None
        self.denoised_data = None

        self.ecg_widget = self.findChild(QtWidgets.QWidget, "EcgWidget")
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout(self.ecg_widget)
        layout.addWidget(self.canvas)
        self.ecg_widget.setLayout(layout)


    def upload_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ECG Data File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                df = pd.read_csv(file_path)
                print(f"got data from {file_path}")
                self.data = df
                self.non_local_means = Non_local_means(self.data)

                denoised_data = {}
                for column in df.columns:
                    denoised_signal = self.denoise_ecg_signal(df[column].values)
                    denoised_data[column] = denoised_signal

                # Denoised data back to dataframe
                self.denoised_data = pd.DataFrame(denoised_data)

                print(f"original shape{df.shape}")
                print(f"denoised shape{self.denoised_data.shape}")

                self.plot_denoised_data()

            except Exception as e:
                print(f"error loading file: {str(e)}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")


    def denoise_ecg_signal(self, ecg_signal, fs=500):
        #low pass filtering
        fp = 50
        fs_stop = 60
        Passband_ripple = 1
        Stopband_attenuations = 2.5
        nyquist_rate = fs / 2
        wp = fp / nyquist_rate
        ws = fs_stop / nyquist_rate
        n, wn = sig.buttord(wp, ws, Passband_ripple, Stopband_attenuations)
        b, a = sig.butter(n, wn, btype='low')
        filtered_signal = sig.filtfilt(b, a, ecg_signal)
        print("butterworth filter done")

        #RLOESS --> smoothing for baseline wander removal
        t = np.arange(len(filtered_signal))
        yy2 = lowess(filtered_signal, t, frac=0.1, it=0, is_sorted=True)[:, 1]
        baseline_removed = filtered_signal - yy2
        print("Baseline wander removed")

        #noise var estimation
        Dl1 = baseline_removed.copy()
        for k in range(1, len(Dl1) - 1):
            Dl1[k] = (2 * Dl1[k] - Dl1[k - 1] - Dl1[k + 1]) / np.sqrt(6)
        noise_variance = 1.4826 * np.median(np.abs(Dl1 - np.median(Dl1)))
        print(f"Estimated noise var: {noise_variance}")
        window_size = 5
        patch_size = 3
        denoised_signal = self.non_local_means.apply_non_local_means(
            baseline_removed, noise_variance, window_size, patch_size
        )

        return denoised_signal
    def plot_denoised_data(self):
        if self.denoised_data is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Assuming we're plotting all leads (I, II, III, aVR, aVL, aVF, V1-V6)
        # time = np.arange(len(self.denoised_data)) / 500  # Assuming fs=500 Hz
        # for column in self.denoised_data.columns:
        #     ax.plot(time, self.denoised_data[column], label=column)

        time = np.arange(len(self.denoised_data)) / 500  # Assuming fs=500 Hz
        first_lead = self.denoised_data.columns[0]  # Get the name of the first column (e.g., "I")
        ax.plot(time, self.denoised_data[first_lead], label="Lead I", color='blue')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title('ECG Signal')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())






