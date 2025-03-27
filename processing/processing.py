import numpy as np
import neurokit2 as nk
import pandas as pd


def classify_ecg(ecg_signal, sampling_rate=500, method="pantompkins1985"):
    # Convert to numeric if needed
    ecg_signal = np.array(ecg_signal, dtype=float)

    # Process ECG Signal (without automatic peak detection)
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

    # Manually detect R-peaks using a different method
    peaks, peak_info = nk.ecg_findpeaks(signals["ECG_Clean"], method=method, sampling_rate=sampling_rate)
    rpeaks = np.where(peaks["ECG_R_Peaks"] == 1)[0]

    print(f"[DEBUG] R-peaks detected ({method}): {len(rpeaks)} peaks found")

    if len(rpeaks) < 2:
        print("[ERROR] Not enough R-peaks detected. Cannot compute HRV.")
        return "Unclassified"

    # Compute HRV features
    hrv_features = nk.hrv(rpeaks, sampling_rate=sampling_rate)

    # Extract HR and HRV features
    hr = np.mean(nk.ecg_rate(rpeaks, sampling_rate=sampling_rate))
    std_rr = hrv_features.get("HRV_SDNN", [np.nan])[0]
    rmssd = hrv_features.get("HRV_RMSSD", [np.nan])[0]

    print(f"[DEBUG] HR (BPM): {hr:.2f}")
    print(f"[DEBUG] HRV SDNN (ms): {std_rr:.2f}")
    print(f"[DEBUG] HRV RMSSD (ms): {rmssd:.2f}")

    # Classification Rules
    if std_rr > 50 and hr > 100:
        return "Atrial Fibrillation (AFIB)"
    elif 250 > hr > 150 and std_rr < 30:
        return "Atrial Tachycardia (AT)"
    elif 250 > hr > 150 and std_rr > 30:
        return "Atrial Flutter (AF)"
    elif 60 <= hr <= 100 and rmssd > 50:
        return "Sinus Arrhythmia (SA)"
    elif hr < 60 and std_rr > 40:
        return "Sinus Bradycardia (SB)"
    elif hr > 100 and std_rr < 30:
        return "Sinus Tachycardia (ST)"
    elif hr > 180 and std_rr < 20:
        return "Supraventricular Tachycardia (SVT)"
    else:
        return "Unclassified"


# Load ECG Data
df = pd.read_csv("../data/filtered/ST/MUSE_20180111_155115_19000.csv", header=None)

# Ensure numeric conversion
df = df.apply(pd.to_numeric, errors='coerce')  # Convert to float, set errors to NaN

# Drop NaN values
df = df.dropna()

# Extract ECG signal
ecg_signal = df.iloc[:, 1].values  # Assuming ECG data is in the second column
print(f"ECG Signal Length: {len(ecg_signal)}")
print(f"ECG Signal First 10 Values: {ecg_signal[:10]}")
print(f"ECG Signal Mean: {np.mean(ecg_signal)}, Std: {np.std(ecg_signal)}")

# Try different R-peak detection methods
for method in ["pantompkins1985", "hamilton2002", "elgendi2010"]:
    print(f"\n🔹 Using R-peak detection method: {method}")
    classification = classify_ecg(ecg_signal, method=method)
    print("Detected Arrhythmia:", classification)
