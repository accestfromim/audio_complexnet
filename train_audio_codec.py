
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import DatasetDict
import os

from AudioComplexNet.modeling_audio_codec import AudioCodec, Discriminator
from AudioComplexNet.utils import overlap_add
from AudioComplexNet.losses import MultiScaleSTFTLoss
from AudioComplexNet.metrics import compute_metrics_batch, calculate_sisdr
from prepare import dataset, collator, custom_freqs

# Modify collator to not truncate
collator.max_frames = None

def get_train_dataset():
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        keys = list(dataset.keys())
        for name in keys:
            if "train" in name:
                return dataset[name]
        return dataset[keys[0]]
    return dataset

def get_validation_dataset():
    if isinstance(dataset, DatasetDict):
        if "validation" in dataset:
            return dataset["validation"]
        if "test" in dataset:
            return dataset["test"]
        keys = list(dataset.keys())
        for name in keys:
            if "valid" in name or "eval" in name:
                return dataset[name]
    return None

def hinge_loss(fake, real):
    if isinstance(fake, list):
        loss = 0
        for f, r in zip(fake, real):
            loss += torch.mean(F.relu(1.0 + f)) + torch.mean(F.relu(1.0 - r))
        return loss / len(fake)
    else:
        loss_fake = torch.mean(F.relu(1.0 + fake))
        loss_real = torch.mean(F.relu(1.0 - real))
        return loss_fake + loss_real

from torch.optim.lr_scheduler import CosineAnnealingLR

def feature_matching_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    
    # Normalize by number of layers to keep scale consistent regardless of depth
    num_layers = sum(len(d) for d in fmap_r)
    return loss / (num_layers + 1e-7)

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

def spectral_loss(x_hat_real, x_hat_imag, x_real, x_imag, eps=1e-5):
    # Force FP32
    x_hat_real = x_hat_real.float()
    x_hat_imag = x_hat_imag.float()
    x_real = x_real.float()
    x_imag = x_imag.float()

    mag_hat = torch.sqrt(x_hat_real**2 + x_hat_imag**2 + eps)
    mag_real = torch.sqrt(x_real**2 + x_imag**2 + eps)
    
    # Log Magnitude Loss (Critical for audio dynamic range)
    # Clamp before log for extra safety
    mag_hat = torch.clamp(mag_hat, min=eps)
    mag_real = torch.clamp(mag_real, min=eps)
    
    log_mag_hat = torch.log(mag_hat)
    log_mag_real = torch.log(mag_real)
    loss_log_mag = F.l1_loss(log_mag_hat, log_mag_real)
    
    # Linear Magnitude Loss
    loss_mag = F.l1_loss(mag_hat, mag_real)
    
    # Complex L1 (Already in main loop, but good to have here conceptually)
    # loss_complex = F.l1_loss(x_hat_real, x_real) + F.l1_loss(x_hat_imag, x_imag)
    
    return loss_log_mag, loss_mag

def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Config
    sr = 16000
    frame_ms = 32.0
    hop_ms = 8.0 # 125Hz STFT (128 samples) -> 62.5Hz Tokens (Perfect Alignment)
    freqs = custom_freqs.to(device)
    
    # Loss weights (Pure GAN Fine-tuning Strategy)
    # Rationale: Audio GANs (like HiFi-GAN) require very strong reconstruction weights 
    # to balance the gradients, as L1 Loss on spectrograms is numerically small (~0.02-0.05).
    # Standard HiFi-GAN config: Mel=45.0, Feat=2.0. 
    # We use slightly higher weights here to stabilize the previous exploding loss.
    W_ADV = 1.0       # Primary Driver
    W_FEAT = 10.0     # Feature Matching
    W_MEL = 5.0       # Reduced from 45.0 to 5.0 to prevent Gradient Explosion and Clipping dominance
    W_MAG = 0.0      # Replaced by Mel Spectrogram Loss
    W_LOG_MAG = 1.0   # Multi-Scale Log-Mag L1
    W_COMPLEX = 0.0   # Redundant with SI-SDR and MS-STFT, removed to reduce conflict
    W_TIME = 0.0      # Redundant with SI-SDR, removed to prevent L1 blurring
    W_COMMIT = 0.1    # Restored: Z detached in model, so this only updates Codebook now
    W_SISDR = 0.0     # Disable SISDR to prevent High Grad Norm and instability

    # Resume & Save Config
    # TODO: Modify these paths to your actual checkpoint paths
    resume_g_checkpoint = "./checkpoints_codec/epoch_40/codec_modelbkup.pth" 
    #resume_g_checkpoint = None
    resume_d_checkpoint = None # Start D from scratch
    
    save_dir_base = "checkpoints_codec_gan" # Save to a new directory
    
    # Pretrain D Config
    PRETRAIN_D_EPOCHS = 1 # Restore D pretraining as system proves resilient (Acc recovers to ~77%)
    
    # RVQ Layer Weights (Decaying to encourage early layers)
    # User requested different weights for different layers
    n_quantizers = 8
    quantizer_weights = [1.0 * (0.8 ** i) for i in range(n_quantizers)]
    # quantizer_weights = [1.0, 0.8, 0.64, 0.51, 0.41, 0.33, 0.26, 0.21]
    
    # Model
    model = AudioCodec(
        sr=sr, 
        freqs=freqs, 
        frame_ms=frame_ms, 
        hop_ms=hop_ms, 
        n_quantizers=n_quantizers,
        quantizer_weights=quantizer_weights
    )
    discriminator = Discriminator(input_channels=3) # Changed to 3 for Real+Imag+Mag
    
    # Freeze Encoder Config
    FREEZE_ENCODER = False # Changed to False to prevent NaN issues with broken gradient chains
    
    # Optimizers
    lr_g = 2e-5 # Decreased G LR for fine-tuning stability
    lr_d = 2e-5 # Balanced D LR with G
    
    # Soft Freeze Strategy: Set Encoder LR to 0.0 instead of requires_grad=False
    # This keeps the computation graph intact but prevents updates.
    
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
            
    opt_g = AdamW([
        {"params": encoder_params, "lr": 0.0}, # Freeze Encoder
        {"params": decoder_params, "lr": lr_g} # Train Decoder/Quantizer
    ], betas=(0.5, 0.9))
    opt_d = AdamW(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9), weight_decay=1e-4) # Weight decay to prevent D explosion

    # Scheduler
    num_epochs = 40
    sched_g = CosineAnnealingLR(opt_g, T_max=num_epochs, eta_min=1e-6)
    sched_d = CosineAnnealingLR(opt_d, T_max=num_epochs, eta_min=1e-6)

    # Data
    train_dataset = get_train_dataset()
    val_dataset = get_validation_dataset()
    
    # Batch size 4 per GPU
    batch_size = 4
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    else:
        val_dataloader = None

    # Prepare
    model, discriminator, opt_g, opt_d, train_dataloader, val_dataloader, sched_g, sched_d = accelerator.prepare(
        model, discriminator, opt_g, opt_d, train_dataloader, val_dataloader, sched_g, sched_d
    )
    
    # Load Checkpoints (After prepare is usually safer for DDP, but before is fine for state_dict)
    # Here we load into the unwrapped model if possible, or wrapped.
    # Accelerate handles loading transparently if we use load_state, but here we manually load weights.
    
    if resume_g_checkpoint and os.path.exists(resume_g_checkpoint):
        print(f"Loading Generator weights from {resume_g_checkpoint}")
        # Unwrap to load standard state_dict
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = torch.load(resume_g_checkpoint, map_location=device)
        # Allow missing keys (e.g. if structure changed slightly) or strict=True
        missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded G. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
    if resume_d_checkpoint and os.path.exists(resume_d_checkpoint):
        print(f"Loading Discriminator weights from {resume_d_checkpoint}")
        unwrapped_d = accelerator.unwrap_model(discriminator)
        state_dict = torch.load(resume_d_checkpoint, map_location=device)
        missing, unexpected = unwrapped_d.load_state_dict(state_dict, strict=False)
        print(f"Loaded D. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # Losses
    ms_stft = MultiScaleSTFTLoss().to(device)
    from AudioComplexNet.losses import MelSpectrogramLoss
    mel_loss_fn = MelSpectrogramLoss().to(device)

    num_epochs = 40 # Train longer for Codec
    total_batches = len(train_dataloader)
    
    # CSV Logging Setup
    import csv
    csv_file = os.path.join(save_dir_base, "validation_losses.csv")
    if accelerator.is_main_process:
        if not os.path.exists(save_dir_base):
            os.makedirs(save_dir_base, exist_ok=True)
        # Initialize CSV with header if not exists
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Total_Loss", "Adv_Loss", "Mag_Loss", "LogMag_Loss", "Complex_Loss", "Time_Loss", "Commit_Loss", "PESQ", "STOI", "SI-SDR", "D_Real_Acc", "D_Fake_Acc"])

    for epoch in range(num_epochs):
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        # Determine Warmup Status at start of epoch based on history
        if epoch < PRETRAIN_D_EPOCHS:
            is_warmup = True
        else:
            is_warmup = False
            
        epoch_d_real_acc = 0.0
        epoch_d_fake_acc = 0.0
        d_steps = 0
        
        if accelerator.is_main_process:
             status = "PRETRAIN D (Freeze G)" if is_warmup else "TRAIN (G+D)"
             print(f"\nEpoch {epoch+1} Start. Mode: {status}.")

        # Set Generator Mode based on Warmup Status
        if is_warmup:
            model.eval() # Freeze BN/Dropout
            # Also freeze G parameters to save memory/compute
            for p in model.parameters():
                p.requires_grad = False
        else:
            model.train()
            # Optimize: Keep Encoder frozen (requires_grad=False) to save compute/memory
            for name, p in model.named_parameters():
                if "encoder" in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        
        # Discriminator Update Frequency
        D_UPDATE_INTERVAL = 1 # Train D every step (1:1) to fix D collapse
        last_d_loss = 0.0 # For logging continuity
        
        for batch_idx, batch in enumerate(train_dataloader, start=1):
            inputs = batch["inputs_features"] # [B, T, Frame_Len]
            
            # Sanity Check Inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"CRITICAL: Input Batch {batch_idx} contains NaN/Inf. Skipping.")
                continue
                
            # No targets needed, autoencoder
            
            # --- Train Discriminator ---
            # Update D only every D_UPDATE_INTERVAL steps
            should_update_d = (batch_idx % D_UPDATE_INTERVAL == 0)
            
            # Initialize d_loss with last known value for logging
            d_loss = torch.tensor(last_d_loss, device=device)
            
            if should_update_d:
                opt_d.zero_grad()
                
                # Forward Generator (detach)
                with torch.no_grad():
                    if is_warmup:
                        model.eval() # Ensure eval mode during warmup
                        g_out = model(inputs)
                    else:
                        g_out = model(inputs)
                    # x_hat = torch.cat([g_out["real_hat"], g_out["imag_hat"]], dim=-1) # [B, T, 2F]
                    
                    # Real x
                    # x_real = torch.cat([g_out["real"], g_out["imag"]], dim=-1) # [B, T, 2F]
                
                # D Forward
                # Note: Discriminator now returns (score, features) tuple
                # Use Compressed Features for D Stability
                
                # Check for NaN in G output immediately
                if torch.isnan(g_out["real_hat"]).any() or torch.isnan(g_out["imag_hat"]).any():
                    print(f"CRITICAL: Generator produced NaN in output at batch {batch_idx}. Skipping Batch.")
                    # We must skip D update and G update for this batch
                    continue
                    
                fake_r, fake_i, fake_m = power_law_compress(g_out["real_hat"], g_out["imag_hat"])
                real_r, real_i, real_m = power_law_compress(g_out["real"], g_out["imag"])
                
                d_fake_out, _ = discriminator(fake_r.detach(), fake_i.detach(), fake_m.detach()) # Detach for D training
                d_real_out, _ = discriminator(real_r, real_i, real_m)
                
                d_loss_step = hinge_loss(d_fake_out, d_real_out)
                
                if torch.isnan(d_loss_step) or torch.isinf(d_loss_step):
                    print(f"Warning: D Loss is NaN/Inf ({d_loss_step.item()}). Skipping D step.")
                    opt_d.zero_grad()
                else:
                    accelerator.backward(d_loss_step)
                    accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)
                    opt_d.step()
                    
                    d_loss = d_loss_step # Update d_loss for logging
                    last_d_loss = d_loss.item()
                    epoch_d_loss += d_loss.item()
                
                # --- Calculate D Accuracy ---
                with torch.no_grad():
                    # Real Acc: Count logits > 0 (or > 1 for strict margin)
                    # Let's use > 0 as decision boundary for "Real"
                    if isinstance(d_real_out, list):
                        real_correct = 0
                        real_total = 0
                        fake_correct = 0
                        fake_total = 0
                        for dr, df in zip(d_real_out, d_fake_out):
                            real_correct += (dr > 0).float().sum().item()
                            real_total += dr.numel()
                            fake_correct += (df < 0).float().sum().item()
                            fake_total += df.numel()
                        
                        acc_real = real_correct / (real_total + 1e-7)
                        acc_fake = fake_correct / (fake_total + 1e-7)
                    else:
                        acc_real = (d_real_out > 0).float().mean().item()
                        acc_fake = (d_fake_out < 0).float().mean().item()
                    
                    epoch_d_real_acc += acc_real
                    epoch_d_fake_acc += acc_fake
                    d_steps += 1
            
            # --- Train Generator ---
            if not is_warmup:
                opt_g.zero_grad()
                
                # Forward Generator (grad)
                g_out = model(inputs)
                
                # Check for NaN immediately
                if torch.isnan(g_out["real_hat"]).any() or torch.isnan(g_out["imag_hat"]).any():
                     print(f"CRITICAL: G output NaN in Train Step (Batch {batch_idx}). Skipping.")
                     opt_g.zero_grad()
                     continue
                
                # --- Reconstruct Waveforms via OLA for Loss ---
                # Hop Length in Samples
                hop_length = int(sr * hop_ms / 1000)
                
                # Create Synthesis Window (Hanning)
                # Must match frame_len used in model
                frame_len = inputs.shape[-1]
                window = torch.hann_window(frame_len).to(device)
                
                # Reconstruct Pred Waveform
                # g_out["frames_hat"] are reconstructed frames (already effectively windowed by analysis)
                # So OLA(frames_hat, window) gives frames_hat * window -> x * w^2
                wav_hat = overlap_add(g_out["frames_hat"], hop_length, window)
                
                # Reconstruct Target Waveform (from input frames)
                # Inputs are RAW frames (unwindowed).
                # We must window them first to get correct OLA reconstruction (Weighted OLA)
                inputs_windowed = inputs * window.unsqueeze(0).unsqueeze(0)
                wav_target = overlap_add(inputs_windowed, hop_length, window)
                
                # Trim to match length (just in case)
                min_len = min(wav_hat.shape[1], wav_target.shape[1])
                wav_hat = wav_hat[:, :min_len]
                wav_target = wav_target[:, :min_len]
                
                # --- Losses ---
                
                # 1. Multi-Scale STFT Loss (Mag + LogMag) on Waveforms
                loss_ms_mag, loss_ms_log_mag = ms_stft(wav_hat, wav_target)
                
                # 2. Time Domain L1 Loss (Strong Phase Constraint)
                loss_time = F.l1_loss(wav_hat, wav_target)
                
                # 3. Complex Spectrogram L1 Loss (Frame-level, still useful for local structure)
                loss_real = F.l1_loss(g_out["real_hat"], g_out["real"])
                loss_imag = F.l1_loss(g_out["imag_hat"], g_out["imag"])
                loss_complex = loss_real + loss_imag

                # 4. SI-SDR Loss (Differentiable Metric)
                # Maximize SI-SDR => Minimize -SI-SDR
                # wav_hat and wav_target are [B, T]
                sisdr_val = calculate_sisdr(wav_target, wav_hat)
                loss_sisdr = -torch.mean(sisdr_val)

                # Commit Loss (Weighted in model)
                commit_loss = g_out["commit_loss"]
                
                # GAN Loss (Generator wants D to predict real)
                # D returns (scores, features)
                fake_r, fake_i, fake_m = power_law_compress(g_out["real_hat"], g_out["imag_hat"])
                # We need to re-compute real features for feature matching
                real_r, real_i, real_m = power_law_compress(g_out["real"], g_out["imag"])

                d_fake_for_g, d_fake_feats = discriminator(fake_r, fake_i, fake_m)
                d_real_for_g, d_real_feats = discriminator(real_r, real_i, real_m)
                
                loss_adv = 0
                loss_feat = 0
                
                if isinstance(d_fake_for_g, list):
                    # Multi-Scale
                    for score in d_fake_for_g:
                        loss_adv += -torch.mean(score)
                    loss_adv /= len(d_fake_for_g)                    
                    
                    # Feature Matching
                    loss_feat = torch.tensor(0.0, device=device)
                    if W_FEAT > 0:
                        loss_feat = feature_matching_loss(d_real_feats, d_fake_feats)
                    
                else:
                    loss_adv = -torch.mean(d_fake_for_g)
                    loss_feat = feature_matching_loss([d_real_feats], [d_fake_feats]) if W_FEAT > 0 else torch.tensor(0.0, device=device)
                
                # Check for NaN/Inf
                if torch.isnan(loss_adv) or torch.isinf(loss_adv):
                    print(f"Warning: loss_adv is {loss_adv}, clamping")
                    loss_adv = torch.zeros_like(loss_adv)
                
                # Debug Logits Range
                '''
                if batch_idx % 100 == 0:
                    d_min, d_max = 0.0, 0.0
                    if isinstance(d_fake_for_g, list):
                        d_min = min([s.min().item() for s in d_fake_for_g])
                        d_max = max([s.max().item() for s in d_fake_for_g])
                    else:
                        d_min = d_fake_for_g.min().item()
                        d_max = d_fake_for_g.max().item()
                    print(f"[Debug] D Logits Range: {d_min:.2f} to {d_max:.2f}")
                '''

                # Dynamic Weighting?
                # If Feat Loss is too big, scale it down dynamically?
                # No, normalization by layers fixed this.
                
                # Check reconstruction loss magnitude
                # If Rec Loss is 0.5 and Adv Loss is 2.0, G will focus on Adv.
                # We want Rec Loss to dominate slightly or be equal.
                
                # 5. Mel Spectrogram Loss (New! Key for PESQ)
                loss_mel = mel_loss_fn(wav_hat, wav_target)

                total_g_loss = (
                    W_ADV * loss_adv +
                    W_FEAT * loss_feat +          # Added Feature Matching
                    W_MAG * loss_ms_mag +        
                    W_LOG_MAG * loss_ms_log_mag + 
                    W_MEL * loss_mel +            # Mel Loss
                    W_COMPLEX * loss_complex +
                    W_COMMIT * commit_loss +
                    W_TIME * loss_time +            # Use W_TIME variable
                    W_SISDR * loss_sisdr            # SI-SDR Loss
                )
                
                # Check Total Loss
                if torch.isnan(total_g_loss) or torch.isinf(total_g_loss):
                    print(f"CRITICAL: G Loss NaN/Inf. Skip Step. Breakdown: "
                          f"Adv={loss_adv.item():.4f}, Feat={loss_feat.item():.4f}, "
                          f"Mel={loss_mel.item():.4f}, LogMag={loss_ms_log_mag.item():.4f}, "
                          f"Commit={commit_loss.item():.4f}, SISDR={loss_sisdr.item():.4f}")
                    opt_g.zero_grad() # Safety clear
                else:
                    accelerator.backward(total_g_loss)
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 10.0) # Relaxed clipping to 10.0
                    
                    # Print detailed breakdown if norm is huge
                    if grad_norm > 100:
                        print(f"High Grad Norm: {grad_norm:.2f}. Breakdown: "
                              f"Adv={loss_adv.item():.4f}, Feat={loss_feat.item():.4f}, "
                              f"Mel={loss_mel.item():.4f}, LogMag={loss_ms_log_mag.item():.4f}, "
                              f"Commit={commit_loss.item():.4f}, SISDR={loss_sisdr.item():.4f}")
                        
                        # Analyze Layer-wise Gradients
                        try:
                            param_norms = []
                            for name, p in model.named_parameters():
                                if p.grad is not None:
                                    param_norm = p.grad.detach().data.norm(2).item()
                                    param_norms.append((name, param_norm))
                            param_norms.sort(key=lambda x: x[1], reverse=True)
                            print("  Top 5 Gradient Norms:")
                            for name, norm in param_norms[:5]:
                                print(f"    {name}: {norm:.4f}")
                        except Exception as e:
                            print(f"  Error analyzing gradients: {e}")
                    
                    # Skip only if truly insane (like > 20000) or NaN
                    if torch.isnan(grad_norm) or grad_norm > 20000: 
                        print(f"Skipping G step due to exploding gradients: {grad_norm}")
                        if torch.isnan(grad_norm):
                             print("  Finding NaN Gradients:")
                             for name, p in model.named_parameters():
                                 if p.grad is not None and torch.isnan(p.grad).any():
                                     print(f"    NaN Grad in: {name}")
                    else:
                        opt_g.step()
                
                epoch_g_loss += total_g_loss.item()
            
            if accelerator.is_main_process:
                progress = batch_idx / total_batches
                bar_len = 30
                filled_len = int(bar_len * progress)
                bar = "=" * filled_len + "." * (bar_len - filled_len)
                
                if is_warmup:
                     print(
                        f"\rEpoch {epoch+1} [{bar}] {batch_idx}/{total_batches} "
                        f"G: Frozen D: {d_loss.item():.4f} ",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rEpoch {epoch+1} [{bar}] {batch_idx}/{total_batches} "
                        f"G: {total_g_loss.item():.4f}| "
                        f"Adv: {loss_adv.item():.4f} Feat: {loss_feat.item():.4f} "
                        f"Mel: {loss_mel.item():.4f} LogMag: {loss_ms_log_mag.item():.4f} "
                        f"SISDR: {loss_sisdr.item():.4f} ",
                        end="",
                        flush=True,
                    )

        # Track D Loss
        avg_epoch_d_loss = epoch_d_loss / total_batches
        
        # Calculate D Metrics for logging
        avg_d_real_acc = epoch_d_real_acc / d_steps if d_steps > 0 else 0.0
        avg_d_fake_acc = epoch_d_fake_acc / d_steps if d_steps > 0 else 0.0
        
        if accelerator.is_main_process:
            print() # Newline after epoch
            print(f"Epoch {epoch+1} finished. Avg D Loss: {avg_epoch_d_loss:.4f} Avg G Loss: {epoch_g_loss/total_batches:.4f}")
            print(f"    D Acc: Real {avg_d_real_acc:.2%} | Fake {avg_d_fake_acc:.2%}")

        # Step Scheduler
        sched_g.step()
        sched_d.step()
        
        # Validation
        if val_dataloader:
            model.eval()
            discriminator.eval()
            val_total_loss = 0.0
            val_loss_mag = 0.0
            val_loss_log_mag = 0.0
            val_loss_time = 0.0
            val_loss_complex = 0.0
            val_loss_adv = 0.0
            val_pesq = 0.0
            val_stoi = 0.0
            val_sisdr = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch["inputs_features"]
                    g_out = model(inputs)
                    
                    # --- Reconstruct Waveforms via OLA for Loss ---
                    hop_length = int(sr * hop_ms / 1000)
                    frame_len = inputs.shape[-1]
                    window = torch.hann_window(frame_len).to(device)
                    
                    # Reconstruct Pred Waveform
                    wav_hat = overlap_add(g_out["frames_hat"], hop_length, window)
                    
                    # Reconstruct Target Waveform
                    inputs_windowed = inputs * window.unsqueeze(0).unsqueeze(0)
                    wav_target = overlap_add(inputs_windowed, hop_length, window)
                    
                    # Trim
                    min_len = min(wav_hat.shape[1], wav_target.shape[1])
                    wav_hat = wav_hat[:, :min_len]
                    wav_target = wav_target[:, :min_len]
                    
                    # --- Metrics ---
                    # Compute objective metrics (PESQ, STOI, SI-SDR)
                    # We compute this for every batch in validation
                    batch_metrics = compute_metrics_batch(wav_target, wav_hat, sr)
                    val_pesq += batch_metrics["PESQ"]
                    val_stoi += batch_metrics["STOI"]
                    val_sisdr += batch_metrics["SI-SDR"]
                    
                    # --- Losses ---
                    loss_ms_mag, loss_ms_log_mag = ms_stft(wav_hat, wav_target)
                    loss_time = F.l1_loss(wav_hat, wav_target)
                    
                    loss_real = F.l1_loss(g_out["real_hat"], g_out["real"])
                    loss_imag = F.l1_loss(g_out["imag_hat"], g_out["imag"])
                    loss_complex = loss_real + loss_imag
                    
                    commit_loss = g_out["commit_loss"]
                    
                    # Calculate ADV Loss for consistency (using D in eval mode)
                    # D returns (scores, features)
                    fake_r, fake_i, fake_m = power_law_compress(g_out["real_hat"], g_out["imag_hat"])
                    real_r, real_i, real_m = power_law_compress(g_out["real"], g_out["imag"])
                    
                    d_fake_for_g, d_fake_feats = discriminator(fake_r, fake_i, fake_m)
                    d_real_for_g, d_real_feats = discriminator(real_r, real_i, real_m)
                    
                    loss_adv = 0
                    loss_feat = 0
                    
                    if isinstance(d_fake_for_g, list):
                        for score in d_fake_for_g:
                            loss_adv += -torch.mean(score)
                        loss_adv /= len(d_fake_for_g)
                        
                        loss_feat = feature_matching_loss(d_real_feats, d_fake_feats)
                    else:
                        loss_adv = -torch.mean(d_fake_for_g)
                        loss_feat = feature_matching_loss([d_real_feats], [d_fake_feats])
                    
                    # SI-SDR Loss (using metric value since no_grad)
                    # batch_metrics["SI-SDR"] is average dB
                    loss_sisdr = - batch_metrics["SI-SDR"]

                    total_g_loss = (
                        W_ADV * loss_adv +
                        W_FEAT * loss_feat +
                        W_MAG * loss_ms_mag +
                        W_LOG_MAG * loss_ms_log_mag +
                        W_COMPLEX * loss_complex +
                        W_COMMIT * commit_loss +
                        W_TIME * loss_time +
                        W_SISDR * loss_sisdr
                    )
                    
                    if torch.isnan(total_g_loss) or torch.isinf(total_g_loss):
                        continue

                    val_total_loss += total_g_loss.item()
                    val_loss_mag += loss_ms_mag.item()
                    val_loss_log_mag += loss_ms_log_mag.item()
                    val_loss_time += loss_time.item()
                    val_loss_complex += loss_complex.item()
                    val_loss_adv += loss_adv.item()
                    val_count += 1
            
            avg_commit = 0.0 # Not tracked per batch above, but less critical
            
            avg_val_loss = val_total_loss / val_count if val_count > 0 else 0.0
            avg_mag = val_loss_mag / val_count if val_count > 0 else 0.0
            avg_time = val_loss_time / val_count if val_count > 0 else 0.0
            avg_log_mag = val_loss_log_mag / val_count if val_count > 0 else 0.0
            avg_complex = val_loss_complex / val_count if val_count > 0 else 0.0
            avg_adv = val_loss_adv / val_count if val_count > 0 else 0.0
            
            avg_pesq = val_pesq / val_count if val_count > 0 else 0.0
            avg_stoi = val_stoi / val_count if val_count > 0 else 0.0
            avg_sisdr = val_sisdr / val_count if val_count > 0 else 0.0
            
            if accelerator.is_main_process:
                print(f"=== Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} (Mag: {avg_mag:.4f}, Time: {avg_time:.4f}) ===")
                print(f"    Metrics: PESQ={avg_pesq:.4f}, STOI={avg_stoi:.4f}, SI-SDR={avg_sisdr:.2f} dB")
                
                # CSV Log
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1, 
                        f"{avg_val_loss:.4f}", 
                        f"{avg_adv:.4f}", 
                        f"{avg_mag:.4f}", 
                        f"{avg_log_mag:.4f}", 
                        f"{avg_complex:.4f}", 
                        f"{avg_time:.4f}", 
                        f"{avg_commit:.4f}",
                        f"{avg_pesq:.4f}",
                        f"{avg_stoi:.4f}",
                        f"{avg_sisdr:.4f}",
                        f"{avg_d_real_acc:.4f}",
                        f"{avg_d_fake_acc:.4f}"
                    ])

            accelerator.log({
                "val_loss": avg_val_loss,
                "val_mag_loss": avg_mag,
                "val_time_loss": avg_time,
                "val_pesq": avg_pesq,
                "val_stoi": avg_stoi,
                "val_sisdr": avg_sisdr
            }, step=epoch)
            
            # Save Checkpoint
            if accelerator.is_main_process:
                save_path = f"{save_dir_base}/epoch_{epoch+1}"
                os.makedirs(save_path, exist_ok=True)
                accelerator.save_state(save_path) # Saves everything including optim/sched
                # Also save model only for easier loading
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "codec_model.pth"))
                
                # Save Discriminator too
                unwrapped_d = accelerator.unwrap_model(discriminator)
                torch.save(unwrapped_d.state_dict(), os.path.join(save_path, "discriminator_model.pth"))
                
                print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
