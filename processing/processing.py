import numpy as np
import neurokit2 as nk
import pandas as pd

def classify_ecg(ecg_signal, sampling_rate=500):
    ecg_signal = np.array(ecg_signal, dtype=float)

    # Process ECG Signal
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

    # Extract R-peaks
    r_peaks = info["ECG_R_Peaks"]

    # Compute HRV metrics
    hrv_features = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)

    # ECG delineation (extract P-waves, QRS, etc.)
    delineate_signals, delineate_info = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=sampling_rate, method="peak")

    # Extract Features
    features = nk.ecg_analyze(signals, sampling_rate=sampling_rate)

    # Extract HR and RR variability
    hr = features.get("ECG_Rate_Mean", [np.nan])[0]
    std_rr = hrv_features.get("HRV_SDNN", [np.nan])[0]

    # Extract PR Interval & QRS Duration from delineation
    pr_interval = delineate_info.get("ECG_PQ_Mean", np.nan)
    qrs_duration = delineate_info.get("ECG_QRS_Mean", np.nan)

    # Infer P-wave presence
    p_wave_presence = not np.isnan(pr_interval)

    # Debugging Output
    print(f"[DEBUG] HR (BPM): {hr:.2f}")
    print(f"[DEBUG] HRV SDNN (ms): {std_rr:.2f}")
    print(f"[DEBUG] PR Interval (ms): {pr_interval if not np.isnan(pr_interval) else 'N/A'}")
    print(f"[DEBUG] QRS Duration (ms): {qrs_duration if not np.isnan(qrs_duration) else 'N/A'}")
    print(f"[DEBUG] P-Wave Presence: {p_wave_presence}")

    # Classification logic
    if std_rr > 50 and hr > 100 and not p_wave_presence:
        return "Atrial Fibrillation"
    elif 150 <= hr <= 250 and 20 < std_rr < 30 or p_wave_presence:
        return "Atrial Tachycardia"
    elif 60 <= hr <= 100 or p_wave_presence:
        return "Sinus Rhythm"
    elif 100 < hr < 150 and std_rr < 30 or p_wave_presence:
        return "Sinus Tachycardia"
    elif hr > 180 and std_rr < 20 and not p_wave_presence:
        return "Supraventricular Tachycardia"
    else:
        return "Unclassified"

# Load ECG Data
df = pd.read_csv("../data/filtered/SVT/MUSE_20180111_155633_99000.csv", header=None)

# Ensure numeric conversion
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Extract ECG signal
ecg_signal = df.iloc[:, 1].values

# Detect arrhythmia
classification = classify_ecg(ecg_signal)
print("Detected Arrhythmia:", classification)
