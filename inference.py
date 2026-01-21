import torch
import torchaudio
import soundfile as sf
import argparse
import os
from AudioComplexNet.configuration_complexnet import ComplexNetConfig
from AudioComplexNet.modeling_complexnet_dummy_new import ComplexNetLM
from utils import frame_audio, overlap_add

custom_freqs = torch.linspace(0, 8000, 512)
def main():
    parser = argparse.ArgumentParser(description="Inference for ComplexNet Audio Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output_audio", type=str, default="output.wav", help="Path to save output audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_gen_frames", type=int, default=50, help="Number of frames to autoregressively generate")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 1. Model Configuration
    # Ensure these match training parameters
    sr = 16000
    frame_ms = 25.0
    hop_ms = 10.0
    freqs = custom_freqs.to(device)
    
    hidden_size = 2 * freqs.numel()
    config = ComplexNetConfig(hidden_size=hidden_size)
    
    # 2. Load Model
    print("Loading model...")
    model = ComplexNetLM(config=config, sr=sr, freqs=freqs, frame_ms=frame_ms, hop_ms=hop_ms)
    
    # Load state dict
    # Handle potential "module." prefix from accelerator/distributed training
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Check if state_dict has keys
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # 3. Load and Preprocess Audio
    print(f"Loading audio from {args.input_audio}...")
    wav, original_sr = sf.read(args.input_audio)
    wav = torch.tensor(wav).float()
    
    # Handle channels
    if wav.ndim > 1:
        wav = wav.transpose(0, 1) # [Channels, Time]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0) # Mix to mono
        else:
            wav = wav.squeeze(0)
            
    # Resample if necessary
    if original_sr != sr:
        print(f"Resampling from {original_sr} to {sr}...")
        resampler = torchaudio.transforms.Resample(original_sr, sr)
        wav = resampler(wav)
    
    wav = wav.to(device)
    
    frames = frame_audio(wav, sr, frame_ms, hop_ms).unsqueeze(0).to(device)
    
    print(f"Input shape (frames): {frames.shape}")
    
    # 4. Autoregressive Generation (next-frame prediction)
    print("Running autoregressive generation...")
    with torch.no_grad():
        context_len = 512
        generated = frames[:, -context_len:, :]
        attention = torch.ones(
            generated.shape[:2], dtype=torch.long, device=device
        )
        
        all_frames = frames
        for _ in range(args.num_gen_frames):
            outputs = model(
                input_ids=generated, attention_mask=attention, use_cache=False
            )
            next_frame = outputs.logits[:, -1, :].unsqueeze(1)
            
            all_frames = torch.cat([all_frames, next_frame], dim=1)
            generated = torch.cat([generated, next_frame], dim=1)
            if generated.size(1) > context_len:
                generated = generated[:, -context_len:, :]
            attention = torch.ones(
                generated.shape[:2], dtype=torch.long, device=device
            )
    predicted_frames = all_frames
    print(f"Output shape (frames): {predicted_frames.shape}")
    
    # 5. Reconstruction (Overlap-Add)
    hop_length = int(sr * hop_ms / 1000)
    output_wav = overlap_add(predicted_frames, hop_length)
    
    # Remove batch dimension and move to CPU
    output_wav = output_wav.squeeze(0).cpu().numpy()
    
    # 6. Save Output
    print(f"Saving output to {args.output_audio}...")
    sf.write(args.output_audio, output_wav, sr)
    print("Done!")

if __name__ == "__main__":
    main()
