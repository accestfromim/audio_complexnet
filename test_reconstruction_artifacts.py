
import torch
import torch.nn.functional as F
import numpy as np

def overlap_add_mode(frames, hop_length, window, mode="analysis_only"):
    """
    mode:
      "analysis_only": sum(x*w) / sum(w)
      "synthesis": sum(x*w*w) / sum(w^2)
    """
    num_frames, frame_len = frames.shape
    total_len = (num_frames - 1) * hop_length + frame_len
    
    output = torch.zeros(total_len)
    overlap_count = torch.zeros(total_len)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_len
        
        chunk = frames[i] # Assume x*w
        
        if mode == "synthesis":
            chunk = chunk * window
            weight = window ** 2
        else:
            weight = window
            
        output[start:end] += chunk
        overlap_count[start:end] += weight
        
    mask = overlap_count > 1e-4
    output[mask] /= overlap_count[mask]
    return output

def test():
    # Setup
    sr = 16000
    frame_ms = 25.0
    hop_ms = 8.0 # 68% overlap
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    
    # Generate Sine Wave
    t = torch.arange(sr * 1.0) / sr
    x = torch.sin(2 * torch.pi * 440 * t)
    
    # Frame and Window (Analysis)
    window = torch.hann_window(frame_len)
    frames = x.unfold(0, frame_len, hop_len)
    frames_w = frames * window # x * w
    
    # Add Noise (Simulate Quantization Error)
    # Noise at edges is particularly harmful for OLA
    noise = torch.randn_like(frames_w) * 0.1
    frames_noisy = frames_w + noise
    
    # Reconstruct
    rec_analysis = overlap_add_mode(frames_noisy, hop_len, window, mode="analysis_only")
    rec_synthesis = overlap_add_mode(frames_noisy, hop_len, window, mode="synthesis")
    
    # Crop to center to ignore boundary effects
    center_start = sr // 4
    center_end = sr // 2
    
    rec_analysis = rec_analysis[center_start:center_end]
    rec_synthesis = rec_synthesis[center_start:center_end]
    gt = x[center_start:center_end]
    
    # Error
    err_analysis = (rec_analysis - gt).abs().mean()
    err_synthesis = (rec_synthesis - gt).abs().mean()
    
    print(f"Analysis Only Error: {err_analysis.item():.6f}")
    print(f"Synthesis Window Error: {err_synthesis.item():.6f}")
    
    # Check smoothness (diff)
    diff_analysis = (rec_analysis[1:] - rec_analysis[:-1]).abs().mean()
    diff_synthesis = (rec_synthesis[1:] - rec_synthesis[:-1]).abs().mean()
    
    print(f"Smoothness (lower is smoother): Analysis={diff_analysis:.6f}, Synthesis={diff_synthesis:.6f}")

if __name__ == "__main__":
    test()
