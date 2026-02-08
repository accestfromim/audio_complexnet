import os
import glob
import soundfile as sf
import numpy as np
import torch

def compute_snr(ref, est):
    """Compute Signal-to-Noise Ratio."""
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    
    noise = ref - est
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return 100.0
        
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def main():
    output_dir = "overfit_results_v4"
    gt_path = os.path.join(output_dir, "gt.wav")
    
    if not os.path.exists(gt_path):
        print(f"Ground truth file not found: {gt_path}")
        return

    gt_audio, sr = sf.read(gt_path)
    print(f"Ground Truth Loaded: {gt_path}, SR={sr}, Len={len(gt_audio)}")

    # Find step files
    step_files = glob.glob(os.path.join(output_dir, "step_*.wav"))
    step_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
    print("\nAnalysis Results:")
    print("-" * 60)
    print(f"{'Step':<10} | {'SNR (dB)':<10} | {'MSE':<15} | {'Max Abs Diff':<15}")
    print("-" * 60)
    
    for step_path in step_files:
        step_num = int(os.path.basename(step_path).split("_")[1].split(".")[0])
        est_audio, _ = sf.read(step_path)
        
        # Compute metrics
        snr = compute_snr(gt_audio, est_audio)
        
        min_len = min(len(gt_audio), len(est_audio))
        mse = np.mean((gt_audio[:min_len] - est_audio[:min_len]) ** 2)
        max_diff = np.max(np.abs(gt_audio[:min_len] - est_audio[:min_len]))
        
        print(f"{step_num:<10} | {snr:<10.2f} | {mse:<15.6f} | {max_diff:<15.6f}")

if __name__ == "__main__":
    main()
