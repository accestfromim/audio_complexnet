
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import DatasetDict
import os

from AudioComplexNet.modeling_audio_codec import AudioCodec, CombinedDiscriminator
from AudioComplexNet.utils import overlap_add
from AudioComplexNet.losses import MultiScaleSTFTLoss
from AudioComplexNet.metrics import compute_metrics_batch, calculate_sisdr, calculate_pesq, calculate_stoi
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
            # Check if rl/gl are tuples (Complex Features: (real, imag))
            if isinstance(rl, (tuple, list)):
                rl_real, rl_imag = rl
                gl_real, gl_imag = gl
                # Detach targets (real) to ensure no gradient flow back to D and save memory
                loss += torch.mean(torch.abs(rl_real.detach() - gl_real) + torch.abs(rl_imag.detach() - gl_imag))
            # Check if standard Complex Tensor
            elif torch.is_complex(rl):
                loss += torch.mean(torch.abs(rl.real.detach() - gl.real) + torch.abs(rl.imag.detach() - gl.imag))
            else:
                loss += torch.mean(torch.abs(rl.detach() - gl))
    
    # Normalize by number of layers to keep scale consistent regardless of depth
    num_layers = sum(len(d) for d in fmap_r)
    return loss / (num_layers + 1e-7)

def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Config
    sr = 16000
    frame_ms = 32.0
    hop_ms = 10.0 # 100Hz STFT -> 25Hz Tokens
    freqs = custom_freqs.to(device)
    
    # Loss weights (Stable GAN Strategy)
    W_ADV = 1.0      # GAN Loss
    W_FEAT = 2.0    # Feature Matching (Increased to 10.0 for better stability)
    W_MEL = 10.0     # Strong Mel Loss (Increased to 45.0 for better spectral quality/PESQ)
    W_MAG = 0.0      # Enable Linear Mag (Fix Spectral Convergence Error)
    W_LOG_MAG = 0.0  # Enable Log Mag (Fix Spectral Convergence Error)
    W_COMPLEX = 0.0  
    W_TIME = 0.0     
    W_COMMIT = 5.0  # Enable Codebook Learning
    W_SISDR = 1.0    # Disable SI-SDR for GAN training (Conflicts with Adversarial Phase)

    # Resume & Save Config
    resume_g_checkpoint = "checkpoints_codec_new_arch_warmup_just_SISDR_20_quantizers_stage2_gan_again/epoch_33/codec_model.pth"
    resume_d_checkpoint = "checkpoints_codec_new_arch_warmup_just_SISDR_20_quantizers_stage2_gan_again/epoch_33/discriminator_model.pth"#"checkpoints_codec_new_arch_warmup_just_SISDR_20_quantizers_stage2/epoch_32/discriminator_model.pth"
    
    save_dir_base = "checkpoints_codec_new_arch_warmup_just_SISDR_20_quantizers_stage3_gan"
    
    # Pretrain D Config
    PRETRAIN_D_EPOCHS = 0
    
    # D Update Interval (Weaken D by updating less frequently)
    D_UPDATE_INTERVAL = 1
    
    # RVQ Layer Weights (Decaying to encourage early layers)
    # User requested different weights for different layers
    n_quantizers = 20
    quantizer_weights = [1.0 * (0.8 ** i) for i in range(n_quantizers)]
    
    # Codebook sizes (Descending: Larger capacity for early layers, smaller for residuals)
    # 1024, 1024, 512, 512, 256, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 32, 32, 32, 32
    n_codebook = [1024, 1024, 512, 512, 256, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 32, 32, 32, 32]
    
    # Model
    model = AudioCodec(
        sr=sr, 
        freqs=freqs, 
        frame_ms=frame_ms, 
        hop_ms=hop_ms, 
        n_quantizers=n_quantizers,
        n_codebook=n_codebook,
        quantizer_weights=quantizer_weights
    )
    
    ENABLE_GAN = True # Set to True to enable Discriminator training
    
    if ENABLE_GAN:
        discriminator = CombinedDiscriminator()
    else:
        discriminator = None
    
    # Freeze Encoder Config
    FREEZE_ENCODER = True # Changed to False for Full Training to improve metrics
    
    # Optimizers
    lr_g = 2e-4 # Increased back to standard 2e-4 now that we have stability fixes
    lr_d = 2e-4 # Increased to standard HiFi-GAN value (was 5e-5) to help D learn Real distribution faster
    
    # Train Full Model
    
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
            
    if FREEZE_ENCODER:
        # Freeze Encoder Parameters
        for p in encoder_params:
            p.requires_grad = False
            
        opt_g = AdamW([
            # {"params": encoder_params, "lr": lr_g}, # Train Encoder
            {"params": decoder_params, "lr": lr_g} # Train Decoder/Quantizer
        ], betas=(0.5, 0.9))
    else:
        opt_g = AdamW([
            {"params": encoder_params, "lr": lr_g}, # Train Encoder
            {"params": decoder_params, "lr": lr_g} # Train Decoder/Quantizer
        ], betas=(0.5, 0.9))
    
    num_epochs = 80

    if ENABLE_GAN:
        opt_d = AdamW(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9), weight_decay=1e-4)
        sched_d = CosineAnnealingLR(opt_d, T_max=num_epochs, eta_min=1e-6)
    else:
        opt_d = None
        sched_d = None

    # Scheduler
    sched_g = CosineAnnealingLR(opt_g, T_max=num_epochs, eta_min=1e-6)
    # sched_d handled above

    # Data
    train_dataset = get_train_dataset()
    val_dataset = get_validation_dataset()
    
    # Batch size 5 per GPU -> Global Batch Size = 25
    batch_size = 3
    
    # Optimize Data Loading for Multi-GPU
    # num_workers per GPU. 5 GPUs * 4 workers = 20 workers total (ensure CPU has enough cores)
    # pin_memory speeds up host-to-device transfer
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collator, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
        )
    else:
        val_dataloader = None

    # Prepare
    if ENABLE_GAN:
        model, discriminator, opt_g, opt_d, train_dataloader, val_dataloader, sched_g, sched_d = accelerator.prepare(
            model, discriminator, opt_g, opt_d, train_dataloader, val_dataloader, sched_g, sched_d
        )
    else:
         model, opt_g, train_dataloader, val_dataloader, sched_g = accelerator.prepare(
            model, opt_g, train_dataloader, val_dataloader, sched_g
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
        
    if ENABLE_GAN and resume_d_checkpoint and os.path.exists(resume_d_checkpoint):
        print(f"Loading Discriminator weights from {resume_d_checkpoint}")
        unwrapped_d = accelerator.unwrap_model(discriminator)
        state_dict = torch.load(resume_d_checkpoint, map_location=device)
        missing, unexpected = unwrapped_d.load_state_dict(state_dict, strict=False)
        print(f"Loaded D. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # Losses
    ms_stft = MultiScaleSTFTLoss().to(device)
    from AudioComplexNet.losses import MelSpectrogramLoss
    mel_loss_fn = MelSpectrogramLoss().to(device)

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

    import gc

    for epoch in range(num_epochs):
        if ENABLE_GAN:
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
            # Optimize: Train Everything (Except Encoder if Frozen)
            for name, p in model.named_parameters():
                if FREEZE_ENCODER and "encoder" in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        
        # Discriminator Update Frequency
        # Train D every 3 steps to balance training (Prevent D from overpowering G)
        last_d_loss = 0.0 # For logging continuity
        
        for batch_idx, batch in enumerate(train_dataloader, start=1):
            inputs = batch["inputs_features"] # [B, T, Frame_Len]
            inputs_wav = batch["inputs_wav"].to(device).unsqueeze(1) # [B, 1, T_wav]
            
            # Sanity Check Inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"CRITICAL: Input Batch {batch_idx} contains NaN/Inf. Skipping.")
                continue
                
            # Prepare Waveform Reconstruction Parameters
            hop_length = int(sr * hop_ms / 1000)
            frame_len = inputs.shape[-1]
            window = torch.hann_window(frame_len).to(device)
            
            # Helper to reconstruct waveform
            def reconstruct_waveform(g_output, input_frames, hop_len, win):
                # Reconstruct Pred Waveform
                # Apply window to frames_hat to match wav_target's WOLA reconstruction (x * w^2)
                # wav_target is derived from inputs * window, so it becomes inputs * window^2 after overlap_add
                # So we must window frames_hat too.
                wav_hat = overlap_add(g_output["frames_hat"] * win.view(1, 1, -1), hop_len, win)
                
                # Reconstruct Target Waveform (from input frames) - Kept for reference/debug, but we use RAW WAV now
                # Inputs are RAW frames (unwindowed).
                # We must window them first to get correct OLA reconstruction (Weighted OLA)
                inputs_windowed = input_frames * win.unsqueeze(0).unsqueeze(0)
                wav_target_wola = overlap_add(inputs_windowed, hop_len, win)
                
                return wav_hat, wav_target_wola
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            # Train D more frequently if needed, or based on interval
            # Initialize d_loss for logging even if not updated
            d_loss = torch.tensor(last_d_loss, device=device)
            
            if ENABLE_GAN and (batch_idx % D_UPDATE_INTERVAL == 0 or is_warmup):
                opt_d.zero_grad()
                
                # Forward G (No Grad for G here)
                with torch.no_grad():
                    g_out = model(inputs, quantize=True) # Always Quantize for GAN training
                    
                    # Reconstruct Waveform
                    wav_hat = overlap_add(g_out["frames_hat"] * window.view(1, 1, -1), hop_length, window)
                    
                    # Use Raw Waveform as Target (Trim to match length)
                    min_len = min(wav_hat.shape[1], inputs_wav.shape[2])
                    wav_hat = wav_hat[:, :min_len]
                    wav_target = inputs_wav[:, 0, :min_len] # [B, T]
                    
                # Forward D
                # Detach wav_hat to prevent G update
                d_fake_out, _ = discriminator(wav_hat.detach())
                d_real_out, _ = discriminator(wav_target)
                
                # Calculate Hinge Loss
                d_loss_step = hinge_loss(d_fake_out, d_real_out)
                
                # Backward D
                accelerator.backward(d_loss_step)
                
                # Clip Gradients
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                
                opt_d.step()
                
                d_loss = d_loss_step
                epoch_d_loss += d_loss.item()
                last_d_loss = d_loss.item()
                
                # Calculate Accuracy
                with torch.no_grad():
                    # Flatten list of outputs
                    fake_logits = torch.cat([x.flatten() for x in d_fake_out])
                    real_logits = torch.cat([x.flatten() for x in d_real_out])
                    
                    epoch_d_real_acc += (real_logits > 0).float().mean().item()
                    epoch_d_fake_acc += (fake_logits < 0).float().mean().item()
                    d_steps += 1
            
            # ---------------------
            # Train Generator
            # ---------------------
            # Only train G if not in warmup
            if not is_warmup:
                opt_g.zero_grad()
                
                # Forward G
                g_out = model(inputs, quantize=True)
                
                # Reconstruct Pred Waveform
                # Apply window to frames_hat to match wav_target's WOLA reconstruction (x * w^2)
                # wav_target is derived from inputs * window, so it becomes inputs * window^2 after overlap_add
                # So we must window frames_hat too.
                wav_hat = overlap_add(g_out["frames_hat"] * window.view(1, 1, -1), hop_length, window)
                
                # Use Raw Waveform as Target (Trim to match length)
                min_len = min(wav_hat.shape[1], inputs_wav.shape[2])
                wav_hat = wav_hat[:, :min_len]
                wav_target = inputs_wav[:, 0, :min_len] # [B, T]
                
                # --- Losses ---
                total_g_loss = torch.tensor(0.0, device=device)
                
                # Helper to add loss if weight > 0
                def add_loss(loss_val, weight):
                    if weight > 0:
                        return loss_val * weight
                    return torch.tensor(0.0, device=device)

                # 1. Multi-Scale STFT Loss (Mag + LogMag) on Waveforms
                loss_ms_mag = torch.tensor(0.0, device=device)
                loss_ms_log_mag = torch.tensor(0.0, device=device)
                if W_MAG > 0 or W_LOG_MAG > 0:
                    loss_ms_mag, loss_ms_log_mag = ms_stft(wav_hat, wav_target)
                    total_g_loss += add_loss(loss_ms_mag, W_MAG)
                    total_g_loss += add_loss(loss_ms_log_mag, W_LOG_MAG)
                
                # 2. Time Domain L1 Loss (Strong Phase Constraint)
                loss_time = torch.tensor(0.0, device=device)
                if W_TIME > 0:
                    loss_time = F.l1_loss(wav_hat, wav_target)
                    total_g_loss += add_loss(loss_time, W_TIME)
                
                # 3. Complex Spectrogram L1 Loss (Frame-level, still useful for local structure)
                loss_complex = torch.tensor(0.0, device=device)
                if W_COMPLEX > 0:
                    loss_real = F.l1_loss(g_out["real_hat"], g_out["real"])
                    loss_imag = F.l1_loss(g_out["imag_hat"], g_out["imag"])
                    loss_complex = loss_real + loss_imag
                    total_g_loss += add_loss(loss_complex, W_COMPLEX)

                # 4. SI-SDR Loss (Differentiable Metric)
                # Maximize SI-SDR => Minimize -SI-SDR
                # wav_hat and wav_target are [B, T]
                loss_sisdr = torch.tensor(0.0, device=device)
                if W_SISDR > 0:
                    sisdr_val = calculate_sisdr(wav_target, wav_hat)
                    loss_sisdr = -torch.mean(sisdr_val)
                    total_g_loss += add_loss(loss_sisdr, W_SISDR)

                # Commit Loss (Weighted in model)
                commit_loss = g_out["commit_loss"]
                total_g_loss += add_loss(commit_loss, W_COMMIT)
                
                # GAN Loss (Generator wants D to predict real)
                loss_adv = torch.tensor(0.0, device=device)
                loss_feat = torch.tensor(0.0, device=device)

                if ENABLE_GAN and (W_ADV > 0 or W_FEAT > 0):
                    # Wavs are already reconstructed as wav_hat, wav_target
                    
                    # D returns (scores, features)
                    # Note: wav_hat here has gradients from G
                    d_fake_for_g, d_fake_feats = discriminator(wav_hat)
                    
                    # Real Features (No Grad for D, No Grad for Real)
                    with torch.no_grad():
                        d_real_for_g, d_real_feats = discriminator(wav_target)
                    
                    # CombinedDiscriminator always returns list
                    loss_adv = 0.0
                    for score in d_fake_for_g:
                        loss_adv += -torch.mean(score)
                    loss_adv /= len(d_fake_for_g)
                    
                    # Feature Matching
                    if W_FEAT > 0:
                        loss_feat = feature_matching_loss(d_real_feats, d_fake_feats)
                
                total_g_loss += add_loss(loss_adv, W_ADV)
                total_g_loss += add_loss(loss_feat, W_FEAT)
                
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
                loss_mel = torch.tensor(0.0, device=device)
                if W_MEL > 0:
                    loss_mel = mel_loss_fn(wav_hat, wav_target)
                    total_g_loss += add_loss(loss_mel, W_MEL)
                
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
                        '''
                        # Skip only if truly insane (like > 20000) or NaN
                        if torch.isnan(grad_norm) or grad_norm > 20000: 
                            print(f"Skipping G step due to exploding gradients: {grad_norm}")
                            if torch.isnan(grad_norm):
                                print("  Finding NaN Gradients...")
                                # 1. Identify which parameter has NaN grad
                                nan_params = []
                                for name, p in model.named_parameters():
                                    if p.grad is not None and torch.isnan(p.grad).any():
                                        nan_params.append(name)
                                        print(f"    NaN Grad in: {name}")
                                
                                # 2. Diagnose which LOSS caused it (Expensive but useful)
                                print("  --- DIAGNOSING NAN SOURCE (Re-Forwarding...) ---")
                                opt_g.zero_grad()
                                
                                # Re-run forward pass to rebuild graph for individual loss backward
                                # (We can't reuse previous graph because it's gone)
                                try:
                                    g_out_debug = model(inputs)
                                    
                                    # Reconstruct wav_hat (same as main loop)
                                    wav_hat_debug = overlap_add(g_out_debug["frames_hat"] * window.view(1, 1, -1), hop_length, window)
                                    
                                    # Reconstruct wav_target (same as main loop)
                                    inputs_windowed_debug = inputs * window.unsqueeze(0).unsqueeze(0)
                                    wav_target_debug = overlap_add(inputs_windowed_debug, hop_length, window)
                                    
                                    # Trim
                                    min_len = min(wav_hat_debug.shape[1], wav_target_debug.shape[1])
                                    wav_hat_debug = wav_hat_debug[:, :min_len]
                                    wav_target_debug = wav_target_debug[:, :min_len]
                                    
                                    debug_losses = {}
                                    
                                    # Re-calculate individual losses
                                    if W_MAG > 0 or W_LOG_MAG > 0:
                                        l_mag, l_log = ms_stft(wav_hat_debug, wav_target_debug)
                                        if W_MAG > 0: debug_losses["Mag"] = l_mag
                                        if W_LOG_MAG > 0: debug_losses["LogMag"] = l_log
                                        
                                    if W_MEL > 0:
                                        debug_losses["Mel"] = mel_loss_fn(wav_hat_debug, wav_target_debug)
                                        
                                    if W_TIME > 0:
                                        debug_losses["Time"] = F.l1_loss(wav_hat_debug, wav_target_debug)
                                        
                                    if W_SISDR > 0:
                                        # Careful with SI-SDR stability
                                        sisdr_val = calculate_sisdr(wav_target_debug, wav_hat_debug)
                                        debug_losses["SISDR"] = -torch.mean(sisdr_val)
                                    
                                    debug_losses["Commit"] = g_out_debug["commit_loss"]
                                    
                                    if ENABLE_GAN and (W_ADV > 0 or W_FEAT > 0):
                                        # real_hat_debug = g_out_debug["real_hat"].permute(0, 2, 1).unsqueeze(1)
                                        # imag_hat_debug = g_out_debug["imag_hat"].permute(0, 2, 1).unsqueeze(1)
                                        # real_target_debug = g_out_debug["real"].permute(0, 2, 1).unsqueeze(1)
                                        # imag_target_debug = g_out_debug["imag"].permute(0, 2, 1).unsqueeze(1)

                                        d_fake, d_fake_f = discriminator(wav_hat_debug)
                                        d_real, d_real_f = discriminator(wav_target_debug)
                                        
                                        if isinstance(d_fake, list):
                                            if W_ADV > 0:
                                                l_adv = 0
                                                for s in d_fake: l_adv += -torch.mean(s)
                                                debug_losses["Adv"] = l_adv / len(d_fake)
                                            if W_FEAT > 0:
                                                debug_losses["Feat"] = feature_matching_loss(d_real_f, d_fake_f)
                                        else:
                                            if W_ADV > 0: debug_losses["Adv"] = -torch.mean(d_fake)
                                            if W_FEAT > 0: debug_losses["Feat"] = feature_matching_loss([d_real_f], [d_fake_f])

                                    # Check each loss gradient
                                    print("  Checking Gradients for each Loss:")
                                    for name, loss in debug_losses.items():
                                        if loss.requires_grad:
                                            opt_g.zero_grad()
                                            # Backward specific loss
                                            # retain_graph=True in case we share graph (though here we loop sequentially)
                                            # Actually we need retain_graph=True for all except last, 
                                            # but since we might break early, let's just use it.
                                            loss.backward(retain_graph=True)
                                            
                                            # Check gradient of a known NaN param (from step 1) or just first layer
                                            found_nan = False
                                            
                                            # Check specifically the params that were NaN before
                                            check_list = nan_params if nan_params else [n for n,p in model.named_parameters() if p.requires_grad]
                                            
                                            for param_name in check_list:
                                                # Find param by name
                                                param = dict(model.named_parameters())[param_name]
                                                if param.grad is not None and torch.isnan(param.grad).any():
                                                    print(f"    !!! NAN DETECTED from Loss: [{name}] in param [{param_name}] !!!")
                                                    found_nan = True
                                                    break
                                            
                                            if not found_nan:
                                                print(f"    Loss [{name}] Gradients OK.")
                                                
                                    opt_g.zero_grad() # Clean up
                                    
                                except Exception as e:
                                    print(f"  Debug Diagnosis Failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                        '''
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
                        f"G: {total_g_loss.item():.4f}| Commit: {commit_loss.item():.4f} "
                        f"Mel: {loss_mel.item():.4f} ADV: {loss_adv.item():.4f} "
                        f"SISDR: {loss_sisdr.item():.4f} ",
                        end="",
                        flush=True,
                    )

        # Track D Loss
        avg_epoch_d_loss = epoch_d_loss / total_batches
        
        # Cleanup End of Loop Variables to prevent OOM
        # Use try-except to avoid errors if variables weren't created (e.g. skipped batch)
        try:
            del g_out, inputs
        except: pass
        
        try:
            del wav_hat, wav_target, inputs_windowed
        except: pass
        
        try:
            del d_fake_for_g, d_fake_feats, d_real_for_g, d_real_feats
        except: pass
        
        try:
            del total_g_loss, loss_adv, loss_feat, loss_mel, loss_ms_mag, loss_ms_log_mag, loss_time, loss_complex, loss_sisdr, commit_loss
        except: pass
        
        # Calculate D Metrics for logging
        avg_d_real_acc = epoch_d_real_acc / d_steps if d_steps > 0 else 0.0
        avg_d_fake_acc = epoch_d_fake_acc / d_steps if d_steps > 0 else 0.0
        
        if accelerator.is_main_process:
            print() # Newline after epoch
            print(f"Epoch {epoch+1} finished. Avg D Loss: {avg_epoch_d_loss:.4f} Avg G Loss: {epoch_g_loss/total_batches:.4f}")
            print(f"    D Acc: Real {avg_d_real_acc:.2%} | Fake {avg_d_fake_acc:.2%}")

        # Force GC at end of epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Step Scheduler
        sched_g.step()
        if ENABLE_GAN:
            sched_d.step()
        
        # Validation
        if val_dataloader:
            # Important: Keep model in training mode for BatchNorm/Dropout consistency if stats are poor
            # However, standard practice is eval(). 
            # If silence occurs in eval() but not train(), it usually means BatchNorm stats are off (running_mean/var not updated correctly or diverse enough).
            # Given the custom complex architecture and potential for instability, let's try force eval() but monitor.
            # actually, let's stick to eval() but if the user reports silence, it's a huge hint.
            
            model.eval() 
            if ENABLE_GAN:
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
            metric_count = 0 # Track batches with expensive metrics
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch["inputs_features"]
                    inputs_wav = batch["inputs_wav"].to(device).unsqueeze(1) # [B, 1, T]

                    # Use same inference mode as training (Continuous or Quantized)
                    # For stage2, we use quantize=False
                    # But if we are in normal training, we might use True. 
                    # Let's align with the training loop's setting.
                    # Since we hardcoded quantize=False in training loop for this stage, we should do same here.
                    g_out = model(inputs, quantize=True)
                    
                    # --- Reconstruct Waveforms via OLA for Loss ---
                    hop_length = int(sr * hop_ms / 1000)
                    frame_len = inputs.shape[-1]
                    window = torch.hann_window(frame_len).to(device)
                    
                    # Reconstruct Pred Waveform
                    wav_hat = overlap_add(g_out["frames_hat"] * window.view(1, 1, -1), hop_length, window)
                    
                    # Use Raw Waveform as Target
                    min_len = min(wav_hat.shape[1], inputs_wav.shape[2])
                    wav_hat = wav_hat[:, :min_len]
                    wav_target = inputs_wav[:, 0, :min_len]
                    
                    # --- Metrics ---
                    # 1. SI-SDR (Fast, GPU-friendly, needed for Loss)
                    sisdr_batch = calculate_sisdr(wav_target, wav_hat)
                    val_sisdr += sisdr_batch.mean().item()
                    
                    # 2. Expensive Metrics (PESQ, STOI) - Limit to first 10 batches to save time
                    if val_count < 10:
                        # Move to CPU for PESQ/STOI
                        refs_np = wav_target.detach().cpu().numpy()
                        ests_np = wav_hat.detach().cpu().numpy()
                        
                        p_sum = 0
                        s_sum = 0
                        batch_sz = len(refs_np)
                        for i in range(batch_sz):
                            p_sum += calculate_pesq(refs_np[i], ests_np[i], sr)
                            s_sum += calculate_stoi(refs_np[i], ests_np[i], sr)
                        
                        val_pesq += p_sum / batch_sz
                        val_stoi += s_sum / batch_sz
                        metric_count += 1
                    
                    # --- Losses ---
                    loss_ms_mag, loss_ms_log_mag = ms_stft(wav_hat, wav_target)
                    loss_time = F.l1_loss(wav_hat, wav_target)
                    
                    loss_real = F.l1_loss(g_out["real_hat"], g_out["real"])
                    loss_imag = F.l1_loss(g_out["imag_hat"], g_out["imag"])
                    loss_complex = loss_real + loss_imag
                    
                    commit_loss = g_out["commit_loss"]
                    
                    # Calculate ADV Loss for consistency (using D in eval mode)
                    loss_adv = torch.tensor(0.0, device=device)
                    loss_feat = torch.tensor(0.0, device=device)

                    if ENABLE_GAN:
                        # Ensure correct shape for Discriminator
                        # real_hat_val = g_out["real_hat"].permute(0, 2, 1).unsqueeze(1)
                        # imag_hat_val = g_out["imag_hat"].permute(0, 2, 1).unsqueeze(1)
                        # real_target_val = g_out["real"].permute(0, 2, 1).unsqueeze(1)
                        # imag_target_val = g_out["imag"].permute(0, 2, 1).unsqueeze(1)

                        # CombinedDiscriminator takes waveform [B, 1, T]
                        # wav_hat and wav_target are already computed above [B, T]
                        
                        d_fake_for_g, d_fake_feats = discriminator(wav_hat)
                        d_real_for_g, d_real_feats = discriminator(wav_target)
                        
                        loss_adv = 0
                        loss_feat = 0
                        
                        # CombinedDiscriminator always returns list
                        for score in d_fake_for_g:
                            loss_adv += -torch.mean(score)
                        loss_adv /= len(d_fake_for_g)
                        
                        if W_FEAT > 0:
                            loss_feat = feature_matching_loss(d_real_feats, d_fake_feats)
                    
                    # SI-SDR Loss (using metric value since no_grad)
                    # batch_metrics["SI-SDR"] is average dB
                    loss_sisdr = -torch.mean(sisdr_batch)

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
            
            avg_pesq = val_pesq / metric_count if metric_count > 0 else 0.0
            avg_stoi = val_stoi / metric_count if metric_count > 0 else 0.0
            avg_sisdr = val_sisdr / val_count if val_count > 0 else 0.0
            
            if accelerator.is_main_process:
                print(f"=== Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} (Mag: {avg_mag:.4f}, Time: {avg_time:.4f}, ADV: {avg_adv:.4f}) ===")
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
                if ENABLE_GAN and discriminator is not None:
                    unwrapped_d = accelerator.unwrap_model(discriminator)
                    torch.save(unwrapped_d.state_dict(), os.path.join(save_path, "discriminator_model.pth"))
                
                print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
