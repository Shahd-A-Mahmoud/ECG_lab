import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from biosppy.signals import ecg
import matplotlib.pyplot as plt
from denoising.non_local_means import Non_local_means

def filter_lead(signal, fs=500, lowcut=0.5, highcut=45):
    """Bandpass filter for ECG signals with improved stability"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal, padlen=150)  # Increased padlen for stability

# ========================
# 1. Enhanced Data Loading and Preprocessing
# ========================

def load_ecg_csv(file_path):
    """Load 12-lead ECG from CSV with validation"""
    try:
        df = pd.read_csv(file_path)
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        available_leads = [col for col in leads if col in df.columns]
        
        if not available_leads:
            raise ValueError("No valid ECG leads found in CSV")
            
        return {lead: df[lead].values for lead in available_leads}
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")

def preprocess_ecg(ecg_leads, nlm_params=None, fs=500):
    """Enhanced preprocessing with quality checks"""
    processed = {}
    
    for lead, signal in ecg_leads.items():
        try:
            # Remove DC offset
            signal = signal - np.mean(signal)
            
            # Apply NLM denoising if parameters provided
            if nlm_params:
                nlm = Non_local_means(None)
                signal = nlm.apply_non_local_means(
                    data=signal,
                    noise_var=nlm_params['noise_var'],
                    window_size=nlm_params['window_size'],
                    patch_size=nlm_params['patch_size']
                )
            
            # Apply bandpass filter
            filtered = filter_lead(signal, fs)
            processed[lead] = filtered
            
        except Exception as e:
            print(f"Error processing lead {lead}: {str(e)}")
            continue
            
    return processed

# ========================
# 2. Advanced Feature Extraction
# ========================

def detect_r_peaks(signal, fs=500):
    """Robust R-peak detection with multiple fallbacks"""
    try:
        # Try BioSPPy first
        result = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
        return result['rpeaks']
    except:
        try:
            # Fallback to Pan-Tompkins-like detection
            diff = np.diff(np.abs(signal))
            peaks, _ = find_peaks(diff, height=np.percentile(diff, 95), 
                                distance=int(0.6*fs))
            return peaks
        except:
            # Final fallback to simple threshold
            peaks, _ = find_peaks(signal, height=np.mean(signal)*1.5, 
                                distance=int(0.5*fs))
            return peaks

def analyze_p_waves(signal, r_peaks, fs=500):
    """Comprehensive P-wave analysis"""
    if len(r_peaks) < 3:
        return (False, False, 0)
    
    # Extract P-wave segments (100ms before R peak)
    p_segments = []
    for r in r_peaks:
        start = max(0, r - int(0.1*fs))
        end = max(0, r - int(0.02*fs))
        if end > start:
            p_segments.append(signal[start:end])
    
    if len(p_segments) < 2:
        return (False, False, 0)
    
    # P-wave presence detection
    autocorrs = [np.correlate(seg-np.mean(seg), (seg-np.mean(seg)), mode='same') 
                for seg in p_segments]
    presence_scores = [np.max(ac)/np.mean(ac[ac > 0]) for ac in autocorrs]
    presence = np.mean(presence_scores) > 2.5
    
    # P-wave morphology analysis
    template = p_segments[np.argmax(presence_scores)]
    correlations = [np.corrcoef(template-np.mean(template), 
                   seg-np.mean(seg))[0,1] for seg in p_segments]
    morph_score = np.mean(correlations)
    morphology = morph_score > 0.6
    
    return (presence, morphology, morph_score)

def extract_advanced_features(signal, r_peaks, fs=500):
    """Comprehensive feature extraction with quality metrics"""
    if len(r_peaks) < 3:
        return None
    
    # RR intervals and heart rate
    rr = np.diff(r_peaks)/fs
    hr = 60/np.mean(rr)
    hr_var = np.std(60/rr)
    rr_var = np.std(rr)/np.mean(rr)
    
    # P-wave analysis
    p_presence, p_morph, p_score = analyze_p_waves(signal, r_peaks, fs)
    
    # QRS analysis
    qrs_durations = []
    for r in r_peaks:
        q = max(0, r - int(0.05*fs))
        s = min(len(signal)-1, r + int(0.05*fs))
        qrs_durations.append((s-q)/fs)
    avg_qrs = np.mean(qrs_durations)
    
    # Signal quality metrics
    noise_level = np.percentile(np.abs(signal[:int(0.5*fs)]), 95)
    baseline_drift = np.mean(np.abs(np.diff(signal[:int(2*fs)])))
    
    return {
        'hr': hr,
        'hr_var': hr_var,
        'rr_var': rr_var,
        'p_present': p_presence,
        'p_morph': p_morph,
        'p_score': p_score,
        'qrs_duration': avg_qrs,
        'noise_level': noise_level,
        'baseline_drift': baseline_drift,
        'r_peaks_found': len(r_peaks)
    }

# ========================
# 3. Enhanced Classification
# ========================

def assess_lead_quality(features):
    """Determine if lead is suitable for analysis"""
    if features is None:
        return False
        
    quality_flags = [
        features['noise_level'] < 0.3,
        features['baseline_drift'] < 0.5,
        features['r_peaks_found'] >= 3,
        features['qrs_duration'] < 0.15
    ]
    
    return sum(quality_flags) >= 3

def classify_rhythm(features):
    """Enhanced rhythm classification with more conditions"""
    if not assess_lead_quality(features):
        return "Unreliable - Poor Signal Quality"
    
    # Atrial Fibrillation criteria
    if (features['hr'] > 100 and 
        features['rr_var'] > 0.15 and 
        not features['p_present'] and
        features['p_score'] < 0.4):
        return "Atrial Fibrillation"
    
    # Sinus Tachycardia criteria
    if (features['hr'] > 100 and 
        features['hr'] <= 150 and
        features['p_present'] and 
        features['p_score'] > 0.7 and
        features['rr_var'] < 0.1):
        return "Sinus Tachycardia"
    
    # SVT criteria
    if (features['hr'] > 150 and 
        features['qrs_duration'] < 0.12 and
        (not features['p_present'] or features['p_score'] < 0.5)):
        return "Supraventricular Tachycardia"
    
    # Bradycardia criteria
    if (features['hr'] < 50 and 
        features['p_present'] and
        features['p_score'] > 0.6):
        return "Sinus Bradycardia"
    
    # Normal sinus rhythm
    if (50 <= features['hr'] <= 100 and 
        features['p_present'] and 
        features['p_score'] > 0.7 and
        features['rr_var'] < 0.1):
        return "Normal Sinus Rhythm"
    
    return "Unclassified Arrhythmia"

# ========================
# 4. Multi-Lead Consensus
# ========================

def multi_lead_classification(ecg_leads, fs=500):
    """Get diagnosis from multiple leads with weighted voting"""
    diagnoses = []
    confidences = []
    features_list = []
    
    for lead, signal in ecg_leads.items():
        r_peaks = detect_r_peaks(signal, fs)
        features = extract_advanced_features(signal, r_peaks, fs)
        
        if features is None:
            continue
            
        diagnosis = classify_rhythm(features)
        
        # Calculate confidence score (0-1)
        confidence = (
            0.3 * (1 - min(features['noise_level'], 0.5)/0.5) +
            0.3 * (features['p_score'] if features['p_present'] else 0.5) +
            0.2 * (1 - min(features['rr_var'], 0.5)/0.5) +
            0.2 * (min(len(r_peaks), 10)/10)
        )
        
        diagnoses.append(diagnosis)
        confidences.append(confidence)
        features_list.append(features)
    
    if not diagnoses:
        return "Insufficient Quality Leads", None
    
    # Weighted voting
    unique_dx = list(set(diagnoses))
    votes = {dx: 0 for dx in unique_dx}
    
    for dx, conf in zip(diagnoses, confidences):
        votes[dx] += conf
    
    final_dx = max(votes.items(), key=lambda x: x[1])[0]
    
    # Aggregate features from all leads
    avg_features = {
        'hr': np.mean([f['hr'] for f in features_list]),
        'hr_var': np.mean([f['hr_var'] for f in features_list]),
        'rr_var': np.mean([f['rr_var'] for f in features_list]),
        'p_present': np.mean([f['p_present'] for f in features_list]) > 0.5,
        'p_score': np.mean([f['p_score'] for f in features_list]),
        'qrs_duration': np.mean([f['qrs_duration'] for f in features_list]),
        'noise_level': np.mean([f['noise_level'] for f in features_list]),
        'confidence': np.mean(confidences)
    }
    
    return final_dx, avg_features

# ========================
# 5. Visualization and Reporting
# ========================

def plot_ecg_analysis(original, processed, diagnosis, features):
    """Enhanced visualization with features"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Original signal
    ax1.plot(original, 'b-', alpha=0.7, linewidth=0.5)
    ax1.set_title(f"Original Signal | Diagnosis: {diagnosis}")
    
    # Processed signal with R-peaks
    ax2.plot(processed, 'r-', alpha=0.7, linewidth=0.5)
    r_peaks = detect_r_peaks(processed, 500)
    ax2.plot(r_peaks, processed[r_peaks], 'go', markersize=4)
    ax2.set_title("Processed Signal with R-peaks")
    
    # Add feature information
    textstr = '\n'.join([
        f'Heart Rate: {features["hr"]:.1f} bpm',
        f'RR Variability: {features["rr_var"]:.3f}',
        f'P-waves: {"Present" if features["p_present"] else "Absent"}',
        f'P-wave Morphology Score: {features["p_score"]:.2f}',
        f'QRS Duration: {features["qrs_duration"]*1000:.1f} ms',
        f'Confidence: {features["confidence"]*100:.1f}%'
    ])
    
    plt.gcf().text(0.95, 0.5, textstr, bbox=dict(facecolor='white', alpha=0.5),
                  verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

# ========================
# 6. Main Pipeline
# ========================

if __name__ == "__main__":
    # Configuration
    config = {
        'nlm_params': {
            'noise_var': 0.1,
            'window_size': 5,
            'patch_size': 3
        },
        'fs': 500
    }
    
    try:
        # Load and preprocess
        print("Loading ECG data...")
        raw_ecg = load_ecg_csv('Equipment II/ECG_lab/data/Sinus Tachycardia (ST)/MUSE_20180111_155115_19000.csv')
        print(f"Found leads: {list(raw_ecg.keys())}")
        
        print("Preprocessing ECG...")
        processed_ecg = preprocess_ecg(raw_ecg, config['nlm_params'], config['fs'])
        
        # Analyze and classify
        print("Analyzing rhythms...")
        diagnosis, features = multi_lead_classification(processed_ecg, config['fs'])
        
        # Visualize results
        print("\n===== Results =====")
        print(f"Final Diagnosis: {diagnosis}")
        print(f"Features: {features}")
        
        # Plot lead II for visualization
        if 'II' in processed_ecg:
            plot_ecg_analysis(raw_ecg['II'], processed_ecg['II'], diagnosis, features)
        else:
            first_lead = next(iter(processed_ecg.keys()))
            plot_ecg_analysis(raw_ecg[first_lead], processed_ecg[first_lead], diagnosis, features)
            
    except Exception as e:
        print(f"Processing failed: {str(e)}")