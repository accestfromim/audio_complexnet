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
from AudioComplexNet.losses import MultiScaleSTFTLoss, MelSpectrogramLoss
from AudioComplexNet.metrics import calculate_sisdr
from prepare import dataset, collator, custom_freqs

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    # Training
    OUTPUT_DIR = "overfit_results_v5"
    NUM_STEPS = 3000
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 500
    BATCH_SIZE = 4
    LR = 3e-4
    
    # Audio
    SR = 16000
    FRAME_MS = 32.0
    HOP_MS = 10.0 # 100Hz STFT -> 25Hz Tokens
    
    # Loss Weights (Synced with train_audio_codec.py)
    W_ADV = 0.0      # Primary Driver
    W_FEAT = 0.0     # Feature Matching
    W_MEL = 1.0      # Standard HiFi-GAN Mel Loss Weight
    W_MAG = 0.0      # Replaced by Mel Spectrogram Loss
    W_LOG_MAG = 0.0   # Multi-Scale Log-Mag L1
    W_COMPLEX = 0.0   # Redundant with SI-SDR and MS-STFT
    W_TIME = 0.0      # Redundant with SI-SDR
    W_COMMIT = 1.0    # Increased for fresh Codebook learning
    W_SISDR = 1.0     # Disable SISDR for stability

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Helper Functions
# ==============================================================================

def power_law_compress(real, imag, alpha=0.3, eps=1e-5):
    """
    Compress complex spectrogram for Discriminator stability.
    Returns: (real_c, imag_c, mag_c)
    """
    # Safety: Clamp inputs to prevent extreme values causing instability
    real = torch.clamp(real, min=-1e5, max=1e5)
    imag = torch.clamp(imag, min=-1e5, max=1e5)

    mag = torch.sqrt(real**2 + imag**2 + eps)
    # Clamp mag to avoid gradient explosion in pow(mag, alpha) when mag -> 0
    mag = torch.clamp(mag, min=eps) 
    phase = torch.atan2(imag, real)
    
    mag_c = torch.pow(mag, alpha)
    real_c = mag_c * torch.cos(phase)
    imag_c = mag_c * torch.sin(phase)
    
    # Safety: Replace NaNs if any generated (though unlikely with clamp)
    if torch.isnan(real_c).any() or torch.isnan(imag_c).any():
        real_c = torch.nan_to_num(real_c)
        imag_c = torch.nan_to_num(imag_c)
        mag_c = torch.nan_to_num(mag_c)
    
    return real_c, imag_c, mag_c

def hinge_loss(fake, real):
    """GAN Hinge Loss."""
    loss_fake = torch.mean(F.relu(1.0 + fake))
    loss_real = torch.mean(F.relu(1.0 - real))
    return loss_fake + loss_real

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
    
    # RVQ Setup (Synced with train_audio_codec.py)
    n_quantizers = 8
    quantizer_weights = [1.0 * (0.8 ** i) for i in range(n_quantizers)]
    
    # Codebook sizes (Descending: Larger capacity for early layers, smaller for residuals)
    # 1024, 1024, 512, 512, 256, 256, 128, 128
    n_codebook = [1024, 1024, 512, 512, 256, 256, 128, 128]
    
    model = AudioCodec(
        sr=Config.SR, 
        freqs=freqs, 
        frame_ms=Config.FRAME_MS, 
        hop_ms=Config.HOP_MS,
        n_quantizers=n_quantizers,
        n_codebook=n_codebook,
        quantizer_weights=quantizer_weights
    )
    
    ENABLE_GAN = (Config.W_ADV > 0.0)
    discriminator = None
    if ENABLE_GAN:
        discriminator = Discriminator(input_channels=3) # Mag+Real+Imag
    
    opt_g = AdamW(model.parameters(), lr=Config.LR, betas=(0.5, 0.9))
    opt_d = None
    if ENABLE_GAN:
        opt_d = AdamW(discriminator.parameters(), lr=Config.LR, betas=(0.5, 0.9))
    
    # 3. Data Preparation (Overfit on 4 samples)
    raw_dataset = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
    overfit_dataset = Subset(raw_dataset, list(range(4)))
    collator.max_frames = None # Disable truncation
    # Update collator hop_ms
    collator.hop_ms = Config.HOP_MS
    
    dataloader = DataLoader(overfit_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    
    # Losses
    ms_stft = MultiScaleSTFTLoss().to(device)
    mel_loss_fn = MelSpectrogramLoss().to(device)

    # 4. Prepare with Accelerator
    if ENABLE_GAN:
        model, discriminator, opt_g, opt_d, dataloader = accelerator.prepare(
            model, discriminator, opt_g, opt_d, dataloader
        )
        discriminator.train()
    else:
        model, opt_g, dataloader = accelerator.prepare(
            model, opt_g, dataloader
        )
    
    model.train()
    
    # 5. Training Loop
    hop_len = int(Config.SR * Config.HOP_MS / 1000)
    
    # FETCH BATCH ONCE (Fixed Batch for Overfitting)
    data_iter = iter(dataloader)
    fixed_batch = next(data_iter)
    
    # Save GT ONCE (from the fixed batch)
    x_frames = fixed_batch["inputs_features"].to(device)
    
    print("="*60)
    print("Starting Overfit Experiment")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Steps: {Config.NUM_STEPS}")
    print(f"GAN Enabled: {ENABLE_GAN}")
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
        
        # Reconstruct Waveforms
        window = torch.hann_window(L).to(device)
        wav_hat = overlap_add(x_hat, hop_len, window)
        
        x_frames_windowed = x_frames * window.view(1, 1, -1)
        wav_target = overlap_add(x_frames_windowed, hop_len, window)
        
        min_len = min(wav_hat.shape[1], wav_target.shape[1])
        wav_hat = wav_hat[:, :min_len]
        wav_target = wav_target[:, :min_len]

        # ------------------------------------------------------------------
        # Train Discriminator (If Enabled)
        # ------------------------------------------------------------------
        loss_d = torch.tensor(0.0, device=device)
        if ENABLE_GAN:
            opt_d.zero_grad()
            
            # Compress Inputs
            flat_x = x_frames.view(-1, L)
            flat_x_hat = x_hat.detach().view(-1, L) 
            
            spec_real = frame2vector(flat_x, Config.SR, freqs) 
            spec_hat = frame2vector(flat_x_hat, Config.SR, freqs)
            
            real_r = spec_real.real.view(B, N, -1)
            real_i = spec_real.imag.view(B, N, -1)
            fake_r = spec_hat.real.view(B, N, -1)
            fake_i = spec_hat.imag.view(B, N, -1)
            
            real_r_c, real_i_c, real_m_c = power_law_compress(real_r, real_i)
            fake_r_c, fake_i_c, fake_m_c = power_law_compress(fake_r, fake_i)
            
            logits_real, _ = discriminator(real_r_c, real_i_c, real_m_c)
            logits_fake, _ = discriminator(fake_r_c, fake_i_c, fake_m_c)
            
            if isinstance(logits_real, list):
                loss_d = hinge_loss(logits_fake, logits_real)
            else:
                loss_d = hinge_loss(logits_fake, logits_real)
                
            accelerator.backward(loss_d)
            opt_d.step()
        
        # ------------------------------------------------------------------
        # Train Generator
        # ------------------------------------------------------------------
        opt_g.zero_grad()
        
        loss_adv = torch.tensor(0.0, device=device)
        if ENABLE_GAN:
             flat_x_hat_g = x_hat.view(-1, L)
             spec_hat_g = frame2vector(flat_x_hat_g, Config.SR, freqs)
             fake_r_g = spec_hat_g.real.view(B, N, -1)
             fake_i_g = spec_hat_g.imag.view(B, N, -1)
             
             fake_r_gc, fake_i_gc, fake_m_gc = power_law_compress(fake_r_g, fake_i_g)
             
             logits_fake_g, _ = discriminator(fake_r_gc, fake_i_gc, fake_m_gc)
             
             if isinstance(logits_fake_g, list):
                 for score in logits_fake_g:
                     loss_adv += -torch.mean(score)
                 loss_adv /= len(logits_fake_g)
             else:
                 loss_adv = -torch.mean(logits_fake_g)

        # Losses
        loss_ms_mag, loss_ms_log_mag = ms_stft(wav_hat, wav_target)
        loss_time = F.l1_loss(wav_hat, wav_target)
        loss_mel = mel_loss_fn(wav_hat, wav_target)
        
        sisdr_val = calculate_sisdr(wav_target, wav_hat)
        loss_sisdr = -torch.mean(sisdr_val)
        
        loss_real = F.l1_loss(outputs["real_hat"], outputs["real"])
        loss_imag = F.l1_loss(outputs["imag_hat"], outputs["imag"])
        loss_complex = loss_real + loss_imag
        
        total_g_loss = (
            Config.W_ADV * loss_adv +
            Config.W_MAG * loss_ms_mag + 
            Config.W_LOG_MAG * loss_ms_log_mag +
            Config.W_COMPLEX * loss_complex +
            Config.W_COMMIT * commit_loss +
            Config.W_TIME * loss_time +
            Config.W_MEL * loss_mel +
            Config.W_SISDR * loss_sisdr
        )
        
        accelerator.backward(total_g_loss)
        opt_g.step()
        
        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if step % Config.LOG_INTERVAL == 0:
            print(f"Step {step}: G={total_g_loss.item():.4f} "
                  f"(Mel={loss_mel.item():.4f}, SISDR={loss_sisdr.item():.4f}, "
                  f"Commit={commit_loss.item():.4f}, Adv={loss_adv.item():.4f})")
            
        if step % Config.SAVE_INTERVAL == 0:
            save_path = os.path.join(Config.OUTPUT_DIR, f"step_{step}.wav")
            save_audio(x_hat[0:1], Config.SR, hop_len, save_path)
            print(f"Saved audio to {save_path}")

    print("Overfit experiment complete.")

if __name__ == "__main__":
    main()
