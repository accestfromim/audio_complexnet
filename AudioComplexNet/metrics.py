
import torch
import numpy as np
from pesq import pesq
from pystoi import stoi
import torch.nn.functional as F

def calculate_pesq(ref, est, sr):
    """
    Calculate PESQ score (Wideband).
    Args:
        ref (np.ndarray): Reference audio (1D).
        est (np.ndarray): Estimated audio (1D).
        sr (int): Sample rate.
    Returns:
        float: PESQ score (usually -0.5 to 4.5).
    """
    if sr not in [8000, 16000]:
        # PESQ only supports 8k and 16k.
        # If needed, resample here, but for now assuming 16k based on project.
        return 0.0
    
    try:
        # 'wb' for wideband (16k), 'nb' for narrowband (8k)
        mode = 'wb' if sr == 16000 else 'nb'
        score = pesq(sr, ref, est, mode)
        return score
    except Exception as e:
        print(f"PESQ Error: {e}")
        return 0.0

def calculate_stoi(ref, est, sr):
    """
    Calculate STOI score.
    Args:
        ref (np.ndarray): Reference audio (1D).
        est (np.ndarray): Estimated audio (1D).
        sr (int): Sample rate.
    Returns:
        float: STOI score (0.0 to 1.0).
    """
    try:
        score = stoi(ref, est, sr, extended=False)
        return score
    except Exception as e:
        print(f"STOI Error: {e}")
        return 0.0

def calculate_sisdr(ref, est):
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    Args:
        ref (torch.Tensor): Reference audio [..., T].
        est (torch.Tensor): Estimated audio [..., T].
    Returns:
        torch.Tensor: SI-SDR score in dB.
    """
    # Ensure inputs are tensors
    if not isinstance(ref, torch.Tensor):
        ref = torch.tensor(ref)
    if not isinstance(est, torch.Tensor):
        est = torch.tensor(est)
        
    # Zero-mean
    ref = ref - torch.mean(ref, dim=-1, keepdim=True)
    est = est - torch.mean(est, dim=-1, keepdim=True)
    
    # Calculate alpha
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + 1e-8
    alpha = torch.sum(est * ref, dim=-1, keepdim=True) / ref_energy
    
    # Target and noise components
    e_target = alpha * ref
    e_res = est - e_target
    
    # SI-SDR
    # Add epsilon to numerator to avoid log(0) if e_target is 0 (silence)
    numerator = torch.sum(e_target ** 2, dim=-1) + 1e-8
    denominator = torch.sum(e_res ** 2, dim=-1) + 1e-8
    
    si_sdr = 10 * torch.log10(numerator / denominator)
    return si_sdr

def compute_metrics_batch(refs, ests, sr):
    """
    Compute metrics for a batch of audio.
    Args:
        refs (torch.Tensor): Reference audio [B, T].
        ests (torch.Tensor): Estimated audio [B, T].
        sr (int): Sample rate.
    Returns:
        dict: Average metrics (PESQ, STOI, SI-SDR).
    """
    pesq_scores = []
    stoi_scores = []
    sisdr_scores = []
    
    # Move to CPU/Numpy for PESQ/STOI
    refs_np = refs.detach().cpu().numpy()
    ests_np = ests.detach().cpu().numpy()
    
    for i in range(len(refs_np)):
        r = refs_np[i]
        e = ests_np[i]
        
        # PESQ
        p = calculate_pesq(r, e, sr)
        pesq_scores.append(p)
        
        # STOI
        s = calculate_stoi(r, e, sr)
        stoi_scores.append(s)
    
    # SI-SDR (Can be done on GPU batch)
    sisdr = calculate_sisdr(refs, ests)
    sisdr_scores = sisdr.detach().cpu().numpy().tolist()
    
    return {
        "PESQ": np.mean(pesq_scores),
        "STOI": np.mean(stoi_scores),
        "SI-SDR": np.mean(sisdr_scores)
    }
