import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import DatasetDict

from AudioComplexNet.configuration_complexnet import ComplexNetConfig
from AudioComplexNet.modeling_complexnet_dummy_new import ComplexNetLM
from prepare import dataset, collator, custom_freqs
from accelerate import Accelerator, DistributedDataParallelKwargs
import os


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
        # 尝试查找包含 'valid' 或 'eval' 的键
        keys = list(dataset.keys())
        for name in keys:
            if "valid" in name or "eval" in name:
                return dataset[name]
    return None

def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    
    sr = collator.sr
    frame_ms = 25.0
    hop_ms = 10.0
    
    # Ensure freqs are on the correct device
    freqs = custom_freqs.to(device)
    
    hidden_size = 2 * freqs.numel()
    config = ComplexNetConfig(hidden_size=hidden_size)
    model = ComplexNetLM(config=config, sr=sr, freqs=freqs, frame_ms=frame_ms, hop_ms=hop_ms)
    # model.to(device) # Handled by accelerator.prepare

    train_dataset = get_train_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator)

    val_dataset = get_validation_dataset()
    val_dataloader = None
    if val_dataset:
        if accelerator.is_main_process:
            print(f"Validation dataset found with {len(val_dataset)} samples.")
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collator)
    else:
        if accelerator.is_main_process:
            print("No validation dataset found. Skipping validation.")

    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Prepare with accelerator
    if val_dataloader:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    
    model.train()

    num_epochs = 10
    total_batches = len(train_dataloader)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_dataloader, start=1):
            # inputs = batch["inputs_features"].to(device) # Handled by accelerator
            inputs = batch["inputs_features"]
            attention_mask = batch["attention_mask"]
            target_frames = batch["target_frames"]

            outputs = model(input_ids=inputs, attention_mask=attention_mask, use_cache=False)
            preds = outputs.logits
            loss = F.mse_loss(preds, target_frames)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_main_process:
                progress = batch_idx / total_batches
                bar_len = 30
                filled_len = int(bar_len * progress)
                bar = "=" * filled_len + "." * (bar_len - filled_len)
                print(
                    f"\rEpoch {epoch} [{bar}] {batch_idx}/{total_batches} loss {loss.item():.6f}",
                    end="",
                    flush=True,
                )
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if accelerator.is_main_process:
            print()
            print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.6f}")
        
        # Validation Loop
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch["inputs_features"]
                    attention_mask = batch["attention_mask"]
                    target_frames = batch["target_frames"]

                    outputs = model(input_ids=inputs, attention_mask=attention_mask, use_cache=False)
                    preds = outputs.logits
                    loss = F.mse_loss(preds, target_frames)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            # Note: This validation loss is per-device. For strict global loss, use accelerator.gather
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            if accelerator.is_main_process:
                print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.6f}")
            model.train()

        # Save model checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            save_path = os.path.join(checkpoint_dir, f"complexnet_checkpoint_epoch_{epoch}.pt")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        save_path = os.path.join(checkpoint_dir, "complexnet_final.pt")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), save_path)
        print(f"Training complete. Final model saved to {save_path}")


if __name__ == "__main__":
    main()
