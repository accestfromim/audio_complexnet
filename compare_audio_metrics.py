import argparse
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi
from tabulate import tabulate

def load_wav(path, target_sr=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    sr, audio = wavfile.read(path)
    
    # Normalize to [-1, 1] if integer
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    if target_sr is not None and sr != target_sr:
        # Simple resampling
        # Calculate number of samples
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample_poly(audio, target_sr, sr)
        sr = target_sr
        
    return sr, audio

def calculate_sisdr(ref, est):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    """
    eps = 1e-8
    # Zero-mean
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    
    # Calculate alpha
    ref_energy = np.sum(ref ** 2) + eps
    alpha = np.sum(est * ref) / ref_energy
    
    # Target and noise components
    e_target = alpha * ref
    e_res = est - e_target
    
    # SI-SDR
    numerator = np.sum(e_target ** 2) + eps
    denominator = np.sum(e_res ** 2) + eps
    
    si_sdr = 10 * np.log10(numerator / denominator)
    return si_sdr

def main():
    parser = argparse.ArgumentParser(description="Compare audio metrics (PESQ, STOI, SI-SDR)")
    parser.add_argument("ref_wav", help="Reference ground truth audio")
    parser.add_argument("deg_wavs", nargs="+", help="List of degraded/reconstructed audio files to compare")
    args = parser.parse_args()

    print(f"Loading Reference: {args.ref_wav}")
    try:
        sr_ref_orig, audio_ref_orig = load_wav(args.ref_wav)
    except Exception as e:
        print(f"Error loading reference file: {e}")
        return

    results = []
    
    for deg_path in args.deg_wavs:
        try:
            print(f"Processing: {deg_path}")
            sr_deg, audio_deg = load_wav(deg_path)
            
            # 1. SI-SDR Calculation (Match SR first if needed, usually native SR)
            # We assume native SR comparison for SI-SDR to be most accurate for waveform generation tasks
            # But we need lengths to match exactly.
            
            # Resample deg to ref SR for SI-SDR if different
            if sr_deg != sr_ref_orig:
                audio_deg_sisdr = resample_poly(audio_deg, sr_ref_orig, sr_deg)
            else:
                audio_deg_sisdr = audio_deg
                
            # Align lengths
            min_len = min(len(audio_ref_orig), len(audio_deg_sisdr))
            ref_trim = audio_ref_orig[:min_len]
            deg_trim = audio_deg_sisdr[:min_len]
            
            si_sdr_val = calculate_sisdr(ref_trim, deg_trim)
            
            # 2. PESQ and STOI (Require 16k usually)
            target_metric_sr = 16000
            
            # Resample both for metrics
            if sr_ref_orig != target_metric_sr:
                audio_ref_16k = resample_poly(audio_ref_orig, target_metric_sr, sr_ref_orig)
            else:
                audio_ref_16k = audio_ref_orig
                
            if sr_deg != target_metric_sr:
                audio_deg_16k = resample_poly(audio_deg, target_metric_sr, sr_deg)
            else:
                audio_deg_16k = audio_deg
                
            # Align lengths again for 16k
            min_len_16k = min(len(audio_ref_16k), len(audio_deg_16k))
            ref_16k_trim = audio_ref_16k[:min_len_16k]
            deg_16k_trim = audio_deg_16k[:min_len_16k]
            
            # PESQ (Wideband)
            try:
                pesq_val = pesq(target_metric_sr, ref_16k_trim, deg_16k_trim, 'wb')
            except Exception as e:
                print(f"  PESQ Error: {e}")
                pesq_val = float('nan')
                
            # STOI
            try:
                stoi_val = stoi(ref_16k_trim, deg_16k_trim, target_metric_sr, extended=False)
            except Exception as e:
                print(f"  STOI Error: {e}")
                stoi_val = float('nan')
                
            results.append([
                os.path.basename(deg_path),
                f"{pesq_val:.4f}",
                f"{stoi_val:.4f}",
                f"{si_sdr_val:.4f}"
            ])
            
        except Exception as e:
            print(f"Error processing {deg_path}: {e}")
            results.append([os.path.basename(deg_path), "Error", "Error", "Error"])

    headers = ["File", "PESQ (WB)", "STOI", "SI-SDR (dB)"]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
