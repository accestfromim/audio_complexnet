import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
import os
import soundfile as sf
import numpy as np

from AudioComplexNet.modeling_audio_codec import AudioCodec, Discriminator
from AudioComplexNet.utils import frame2vector, overlap_add
from AudioComplexNet.losses import MultiScaleSTFTLoss
from prepare import dataset, collator, custom_freqs

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    # Training
    OUTPUT_DIR = "overfit_results_v4"
    NUM_STEPS = 3000
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 500
    BATCH_SIZE = 4
    LR = 3e-4
    
    # Audio
    SR = 16000
    FRAME_MS = 32.0 # Standard 75% Overlap
    HOP_MS = 8.0 # 125Hz STFT (128 samples) -> 62.5Hz Tokens (Perfect Alignment)
    
    # Loss Weights
    W_ADV = 1.0
    W_MAG = 45.0
    W_LOG_MAG = 1.0
    W_COMPLEX = 45.0 # Weight for Complex L1 Loss (Real/Imag)
    W_COMMIT = 10.0
    W_TIME = 10.0

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Helper Functions
# ==============================================================================

def hinge_loss(fake, real):
    """GAN Hinge Loss."""
    loss_fake = torch.mean(F.relu(1.0 + fake))
    loss_real = torch.mean(F.relu(1.0 - real))
    return loss_fake + loss_real

def spectral_loss(x_hat_real, x_hat_imag, x_real, x_imag, eps=1e-7):
    """
    Computes spectral reconstruction loss.
    1. Linear Magnitude Loss
    2. Log Magnitude Loss
    """
    mag_hat = torch.sqrt(x_hat_real**2 + x_hat_imag**2 + eps)
    mag_real = torch.sqrt(x_real**2 + x_imag**2 + eps)
    
    log_mag_hat = torch.log(mag_hat)
    log_mag_real = torch.log(mag_real)
    
    loss_log_mag = F.l1_loss(log_mag_hat, log_mag_real)
    loss_mag = F.l1_loss(mag_hat, mag_real)
    
    return loss_log_mag, loss_mag

def save_audio(frames, sr, hop_len, path):
    """
    Reconstructs waveform from frames using Overlap-Add (OLA) and saves to disk.
    Applies Hanning window before OLA to ensure smooth transitions.
    """
    frame_len = frames.shape[-1]
    window = torch.hann_window(frame_len).to(frames.device)
    
    # Note: 'frames' here are model outputs (reconstructed frames).
    # We apply window here because the OLA process assumes windowed segments.
    # Updated: Now overlap_add applies Synthesis Window internally (frames * w)
    # So we pass frames (which are x*w) and window.
    wav = overlap_add(frames, hop_len, window=window)
    wav = wav.squeeze(0).detach().cpu().numpy()
    sf.write(path, wav, sr)

# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    # 1. Setup Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    
    # 2. Model & Optimizer
    freqs = custom_freqs.to(device)
    model = AudioCodec(
        sr=Config.SR, 
        freqs=freqs, 
        frame_ms=Config.FRAME_MS, 
        hop_ms=Config.HOP_MS,
        # Default architecture params (Non-Causal, 100M+) are used from modeling_audio_codec.py
    )
    discriminator = Discriminator(input_channels=2)
    
    opt_g = AdamW(model.parameters(), lr=Config.LR, betas=(0.5, 0.9))
    opt_d = AdamW(discriminator.parameters(), lr=Config.LR, betas=(0.5, 0.9))
    
    # 3. Data Preparation (Overfit on 4 samples)
    raw_dataset = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
    overfit_dataset = Subset(raw_dataset, list(range(4)))
    collator.max_frames = None # Disable truncation
    dataloader = DataLoader(overfit_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    
    # Losses
    ms_stft = MultiScaleSTFTLoss().to(device)

    # 4. Prepare with Accelerator
    model, discriminator, opt_g, opt_d, dataloader = accelerator.prepare(
        model, discriminator, opt_g, opt_d, dataloader
    )
    
    model.train()
    discriminator.train()
    
    # 5. Training Loop
    hop_len = int(Config.SR * Config.HOP_MS / 1000)
    
    # FETCH BATCH ONCE (Fixed Batch for Overfitting)
    data_iter = iter(dataloader)
    fixed_batch = next(data_iter)
    
    # Save GT ONCE (from the fixed batch)
    x_frames = fixed_batch["inputs_features"].to(device)
    # Also save input_values to verify data range if needed
    
    print("="*60)
    print("Starting Overfit Experiment")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Steps: {Config.NUM_STEPS}")
    print("="*60)
    
    # Save GT Audio (First sample)
    if True:
        gt_frames = x_frames[0]
        frame_len = gt_frames.shape[1]
        window = torch.hann_window(frame_len).to(gt_frames.device)
        gt_frames_windowed = gt_frames * window
        
        # Reconstruct and Trim GT
        gt_padded = overlap_add(gt_frames_windowed, hop_len, window=window)
        pad_len = frame_len // 2
        # gt_padded is [1, T], slice time dim then squeeze
        gt_trimmed = gt_padded[:, pad_len:-pad_len].squeeze(0)
        
        gt_path = os.path.join(Config.OUTPUT_DIR, "gt.wav")
        sf.write(gt_path, gt_trimmed.cpu().numpy(), Config.SR)
        print(f"Saved Ground Truth to {gt_path}")

    for step in range(1, Config.NUM_STEPS + 1):
        # Use FIXED Batch
        batch = fixed_batch
            
        x_frames = batch["inputs_features"].to(device) # [B, N, L]
        B, N, L = x_frames.shape
        
        # ------------------------------------------------------------------
        # Forward Pass
        # ------------------------------------------------------------------
        outputs = model(x_frames)
        x_hat = outputs["frames_hat"]
        commit_loss = outputs["commit_loss"]
        
        # Prepare Spectrograms for Discriminator & Loss
        # Flatten time dimension for batched processing
        flat_x = x_frames.view(-1, L)
        flat_x_hat = x_hat.view(-1, L)
        
        # Convert to Complex Spectrograms [B*N, F]
        spec_real = frame2vector(flat_x, Config.SR, freqs) 
        spec_hat = frame2vector(flat_x_hat, Config.SR, freqs)
        
        # Reshape back to [B, N, F] for Discriminator (Time=N, Freq=F)
        real_gt = spec_real.real.view(B, N, -1)
        imag_gt = spec_real.imag.view(B, N, -1)
        
        real_hat = spec_hat.real.view(B, N, -1)
        imag_hat = spec_hat.imag.view(B, N, -1)

        # ------------------------------------------------------------------
        # Train Discriminator
        # ------------------------------------------------------------------
        opt_d.zero_grad()
        
        logits_real = discriminator(real_gt, imag_gt)
        # Detach generator output for D training
        logits_fake = discriminator(real_hat.detach(), imag_hat.detach())
        
        loss_d = hinge_loss(logits_fake, logits_real)
        accelerator.backward(loss_d)
        opt_d.step()
        
        # ------------------------------------------------------------------
        # Train Generator
        # ------------------------------------------------------------------
        opt_g.zero_grad()
        
        # Recalculate Discriminator output for Generator (no detach)
        # Re-compute spectra for gradient flow
        # x_hat = outputs["frames_hat"] # Already computed
        flat_x_hat = x_hat.view(-1, L)
        spec_hat_g = frame2vector(flat_x_hat, Config.SR, freqs)
        real_hat_g = spec_hat_g.real.view(B, N, -1)
        imag_hat_g = spec_hat_g.imag.view(B, N, -1)
        
        logits_fake_g = discriminator(real_hat_g, imag_hat_g)
        loss_adv = -torch.mean(logits_fake_g)
        
        # --- New Loss Calculation ---
        # 1. Reconstruct Waveforms
        window = torch.hann_window(L).to(device)
        
        # Pred Waveform: OLA(x_hat, w)
        # x_hat from model is already implicitly windowed (Analysis window effect in reconstruction)
        wav_hat = overlap_add(x_hat, hop_len, window)
        
        # Target Waveform: OLA(x * w, w)
        # x_frames are raw, need to window them
        x_frames_windowed = x_frames * window.view(1, 1, -1)
        wav_target = overlap_add(x_frames_windowed, hop_len, window)
        
        min_len = min(wav_hat.shape[1], wav_target.shape[1])
        wav_hat = wav_hat[:, :min_len]
        wav_target = wav_target[:, :min_len]
        
        # 2. Multi-Scale STFT Loss
        loss_ms_mag, loss_ms_log_mag = ms_stft(wav_hat, wav_target)
        
        # 3. Time Domain Loss
        loss_time = F.l1_loss(wav_hat, wav_target)
        
        # 4. Complex Spectrogram Loss (Keep for local structure)
        # Using outputs["real_hat"] vs outputs["real"] (which are model-internal, compressed/uncompressed?)
        # modeling_audio_codec.py returns uncompressed real/imag as targets.
        loss_real = F.l1_loss(outputs["real_hat"], outputs["real"])
        loss_imag = F.l1_loss(outputs["imag_hat"], outputs["imag"])
        loss_complex = loss_real + loss_imag
        
        total_g_loss = (
            Config.W_ADV * loss_adv +
            Config.W_MAG * loss_ms_mag + 
            Config.W_LOG_MAG * loss_ms_log_mag +
            Config.W_COMPLEX * loss_complex +
            Config.W_COMMIT * commit_loss +
            Config.W_TIME * loss_time
        )
        
        accelerator.backward(total_g_loss)
        opt_g.step()
        
        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if step % Config.LOG_INTERVAL == 0:
            print(f"Step {step}: G Loss={total_g_loss.item():.4f} (Adv={loss_adv.item():.4f}, MS_Mag={loss_ms_mag.item():.4f}, Time={loss_time.item():.4f}), D Loss={loss_d.item():.4f}")
            
        if step % Config.SAVE_INTERVAL == 0:
            save_path = os.path.join(Config.OUTPUT_DIR, f"step_{step}.wav")
            # Save first sample
            save_audio(x_hat[0:1], Config.SR, hop_len, save_path)
            print(f"Saved audio to {save_path}")

    print("Overfit experiment complete.")

if __name__ == "__main__":
    main()
