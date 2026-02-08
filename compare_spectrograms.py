import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import argparse
import os
import soundfile as sf
from scipy.signal import spectrogram

def analyze_audio(file1, file2, output_img="spectrogram_comparison.png"):
    # Load Audio
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)
    
    # Resample if needed
    if sr1 != sr2:
        print(f"Resampling {file2} from {sr2} to {sr1}")
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1
        
    # Trim to same length
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # Parameters for analysis
    n_fft = 1024
    hop_length = 256
    
    # Compute Spectrograms
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    # Plotting
    plt.figure(figsize=(15, 12))
    
    # 1. Waveforms
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y1, sr=sr1, alpha=0.5)
    plt.title(f"Original Waveform: {os.path.basename(file1)}")
    
    plt.subplot(3, 2, 2)
    librosa.display.waveshow(y2, sr=sr2, color='r', alpha=0.5)
    plt.title(f"Reconstructed Waveform: {os.path.basename(file2)}")
    
    # 2. Spectrograms
    plt.subplot(3, 2, 3)
    librosa.display.specshow(D1, sr=sr1, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Spectrogram")
    
    plt.subplot(3, 2, 4)
    librosa.display.specshow(D2, sr=sr2, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Reconstructed Spectrogram")
    
    # 3. Difference & Zoom
    # Difference
    plt.subplot(3, 2, 5)
    diff = D2 - D1
    librosa.display.specshow(diff, sr=sr1, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Difference (Recon - Orig) [Red=Added Noise]")
    
    # Metrics
    mse = np.mean((y1 - y2)**2)
    
    # Spectral Convergence
    spec_conv = np.linalg.norm(np.abs(librosa.stft(y1)) - np.abs(librosa.stft(y2))) / np.linalg.norm(np.abs(librosa.stft(y1)))
    
    # Detect Discontinuities (High freq energy bursts at frame boundaries)
    # This is a heuristic
    
    plt.subplot(3, 2, 6)
    plt.axis('off')
    plt.text(0.1, 0.9, f"Metrics Analysis:", fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f"MSE: {mse:.6f}", fontsize=12)
    plt.text(0.1, 0.6, f"Spectral Convergence Error: {spec_conv:.4f}", fontsize=12)
    
    # Phase Continuity Check (Derivative of Phase)
    # Large variance in phase derivative often indicates phase scrambling
    S1 = librosa.stft(y1)
    S2 = librosa.stft(y2)
    phase1 = np.angle(S1)
    phase2 = np.angle(S2)
    
    # Unwrapped phase diff across time
    d_phase1 = np.diff(np.unwrap(phase1, axis=1), axis=1)
    d_phase2 = np.diff(np.unwrap(phase2, axis=1), axis=1)
    
    phase_smoothness1 = np.std(d_phase1)
    phase_smoothness2 = np.std(d_phase2)
    
    plt.text(0.1, 0.4, f"Phase Smoothness (Orig): {phase_smoothness1:.4f}", fontsize=12)
    plt.text(0.1, 0.3, f"Phase Smoothness (Recon): {phase_smoothness2:.4f}", fontsize=12)
    plt.text(0.1, 0.2, f"(Higher value = More random phase/Noise)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Comparison saved to {output_img}")
    
    # Print textual analysis
    print("-" * 50)
    print(f"Analysis Report:")
    print(f"MSE: {mse:.6f}")
    print(f"Spectral Convergence Error: {spec_conv:.4f}")
    print(f"Phase Smoothness (Orig): {phase_smoothness1:.4f}")
    print(f"Phase Smoothness (Recon): {phase_smoothness2:.4f}")
    
    if phase_smoothness2 > phase_smoothness1 * 1.5:
        print(">> High phase irregularity detected. This strongly correlates with 'robotic' or 'phasiness' artifacts.")
        
    # Check for Frame Boundary Artifacts (Periodicity in Error)
    # Hop size in samples (assuming 16k, 8ms hop -> 128 samples)
    # Let's check autocorrelation of the residual
    residual = y1 - y2
    acf = np.correlate(residual, residual, mode='full')
    acf = acf[len(acf)//2:]
    
    # Check peaks at common hop sizes (128, 256, etc.)
    hops_to_check = [128, 200, 256, 400, 512] # 8ms, 12.5ms, 16ms, 25ms, 32ms
    print(">> Periodicity Check (Autocorrelation of Residual):")
    for h in hops_to_check:
        if h < len(acf):
            val = acf[h] / acf[0]
            print(f"   Lag {h}: {val:.4f}")
            if val > 0.05:
                print(f"   !! Significant periodic error at Lag {h} samples. This suggests Frame Boundary Artifacts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="Path to original wav")
    parser.add_argument("file2", help="Path to reconstructed wav")
    args = parser.parse_args()
    
    analyze_audio(args.file1, args.file2)
