
import torch
import soundfile as sf
import os
import argparse
from AudioComplexNet.modeling_audio_codec import AudioCodec
from prepare import custom_freqs

def overlap_add(frames, hop_length):
    """
    frames: [B, Num_Frames, Frame_Len]
    """
    batch_size, num_frames, frame_len = frames.shape
    total_len = (num_frames - 1) * hop_length + frame_len
    
    output = torch.zeros(batch_size, total_len, device=frames.device)
    count = torch.zeros(batch_size, total_len, device=frames.device)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_len
        output[:, start:end] += frames[:, i, :]
        count[:, start:end] += 1.0
        
    # Avoid division by zero
    count = count.clamp(min=1.0)
    return output / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output_file", type=str, default="reconstructed.wav", help="Path to save reconstructed audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Config (Must match training)
    sr = 16000
    frame_ms = 25.0
    hop_ms = 10.0
    freqs = custom_freqs.to(args.device)
    
    # Load Model
    model = AudioCodec(sr=sr, freqs=freqs, frame_ms=frame_ms, hop_ms=hop_ms)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()
    
    # Load Audio
    wav, file_sr = sf.read(args.input_file)
    wav = torch.tensor(wav).float().to(args.device)
    
    # Resample if needed
    if file_sr != sr:
        import torchaudio
        resampler = torchaudio.transforms.Resample(file_sr, sr).to(args.device)
        wav = resampler(wav)
        
    # Handle Stereo
    if wav.ndim > 1:
        wav = wav.mean(dim=1) # Mix to mono
        
    # Add Batch Dim
    wav = wav.unsqueeze(0) # [1, T]
    
    # Frame
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    num_frames = 1 + (wav.shape[-1] - frame_length) // hop_length
    frames = wav.unfold(-1, frame_length, hop_length) # [1, N, L]
    
    print(f"Input shape: {wav.shape}, Frames: {frames.shape}")
    
    # Inference
    with torch.no_grad():
        output = model(frames)
        frames_hat = output["frames_hat"] # [1, N, L]
        
    # Reconstruct Waveform (Overlap-Add)
    wav_hat = overlap_add(frames_hat, hop_length)
    
    # Save
    wav_hat = wav_hat.squeeze(0).cpu().numpy()
    sf.write(args.output_file, wav_hat, sr)
    print(f"Reconstructed audio saved to {args.output_file}")

if __name__ == "__main__":
    main()
