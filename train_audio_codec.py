
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import DatasetDict
import os

from AudioComplexNet.modeling_audio_codec import AudioCodec, Discriminator
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
    loss_fake = torch.mean(F.relu(1.0 + fake))
    loss_real = torch.mean(F.relu(1.0 - real))
    return loss_fake + loss_real

def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Config
    sr = 16000
    frame_ms = 25.0
    hop_ms = 10.0
    freqs = custom_freqs.to(device)
    
    # Model
    model = AudioCodec(sr=sr, freqs=freqs, frame_ms=frame_ms, hop_ms=hop_ms)
    discriminator = Discriminator(input_channels=1) 
    # Discriminator expects [B, 1, T, 2F]
    
    # Optimizers
    lr = 3e-4
    opt_g = AdamW(model.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    # Data
    train_dataset = get_train_dataset()
    val_dataset = get_validation_dataset()
    
    # Batch size: User said 5 GPUs. 
    # Codec training is usually on waveforms. 
    # Batch size 4 per GPU?
    batch_size = 4
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    else:
        val_dataloader = None

    # Prepare
    model, discriminator, opt_g, opt_d, train_dataloader, val_dataloader = accelerator.prepare(
        model, discriminator, opt_g, opt_d, train_dataloader, val_dataloader
    )

    num_epochs = 20 # Train longer for Codec
    total_batches = len(train_dataloader)
    
    # Loss weights
    lambda_recon = 1.0
    lambda_commit = 1.0 # Beta is inside model, this is global weight
    lambda_gan = 1.0

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader, start=1):
            inputs = batch["inputs_features"] # [B, T, Frame_Len]
            # No targets needed, autoencoder
            
            # --- Train Discriminator ---
            opt_d.zero_grad()
            
            # Forward Generator (detach)
            with torch.no_grad():
                g_out = model(inputs)
                # x_hat = torch.cat([g_out["real_hat"], g_out["imag_hat"]], dim=-1) # [B, T, 2F]
                
                # Real x
                # x_real = torch.cat([g_out["real"], g_out["imag"]], dim=-1) # [B, T, 2F]

            # D Forward
            d_fake = discriminator(g_out["real_hat"], g_out["imag_hat"])
            d_real = discriminator(g_out["real"], g_out["imag"])
            
            d_loss = hinge_loss(d_fake, d_real)
            accelerator.backward(d_loss)
            opt_d.step()
            
            # --- Train Generator ---
            opt_g.zero_grad()
            
            # Forward Generator (grad)
            g_out = model(inputs)
            # x_hat = torch.cat([g_out["real_hat"], g_out["imag_hat"]], dim=-1) # [B, T, 2F]
            # x_real = torch.cat([g_out["real"], g_out["imag"]], dim=-1)
            
            # Recon Loss (Spectrogram L1 + L2 on Real/Imag parts separately)
            recon_loss = F.l1_loss(g_out["real_hat"], g_out["real"]) + F.l1_loss(g_out["imag_hat"], g_out["imag"]) + \
                         F.mse_loss(g_out["real_hat"], g_out["real"]) + F.mse_loss(g_out["imag_hat"], g_out["imag"])
            
            # Commit Loss
            commit_loss = g_out["commit_loss"]
            
            # GAN Loss (Generator wants D to predict real)
            d_fake_for_g = discriminator(g_out["real_hat"], g_out["imag_hat"])
            g_gan_loss = -torch.mean(d_fake_for_g)
            
            total_g_loss = lambda_recon * recon_loss + lambda_commit * commit_loss + lambda_gan * g_gan_loss
            
            accelerator.backward(total_g_loss)
            opt_g.step()
            
            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += d_loss.item()

            if accelerator.is_main_process:
                progress = batch_idx / total_batches
                bar_len = 30
                filled_len = int(bar_len * progress)
                bar = "=" * filled_len + "." * (bar_len - filled_len)
                print(
                    f"\rEpoch {epoch} [{bar}] {batch_idx}/{total_batches} "
                    f"G_Loss: {total_g_loss.item():.4f} D_Loss: {d_loss.item():.4f} "
                    f"Recon: {recon_loss.item():.4f} GAN: {g_gan_loss.item():.4f}",
                    end="",
                    flush=True,
                )
        
        if accelerator.is_main_process:
            print()
            print(f"Epoch {epoch} finished.")

        # Validation
        if val_dataloader:
            model.eval()
            val_recon_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch["inputs_features"]
                    g_out = model(inputs)
                    # x_hat = torch.cat([g_out["real_hat"], g_out["imag_hat"]], dim=-1)
                    # x_real = torch.cat([g_out["real"], g_out["imag"]], dim=-1)
                    loss = F.l1_loss(g_out["real_hat"], g_out["real"]) + F.l1_loss(g_out["imag_hat"], g_out["imag"])
                    val_recon_loss += loss.item()
                    val_count += 1
            
            if accelerator.is_main_process:
                avg_val_loss = val_recon_loss / val_count if val_count > 0 else 0
                print(f"Epoch {epoch} Validation Recon Loss: {avg_val_loss:.6f}")
        
        # Save
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_dir = "checkpoints_codec"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save Model
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"codec_epoch_{epoch}.pt"))
            
            # Save Discriminator (optional)
            unwrapped_disc = accelerator.unwrap_model(discriminator)
            torch.save(unwrapped_disc.state_dict(), os.path.join(save_dir, f"disc_epoch_{epoch}.pt"))
            
            print(f"Saved checkpoint to {save_dir}")

if __name__ == "__main__":
    main()
