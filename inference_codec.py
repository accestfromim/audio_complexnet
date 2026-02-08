import torch
import torchaudio
import os
import argparse
from AudioComplexNet.modeling_audio_codec import AudioCodec
from AudioComplexNet.utils import vector2frame, overlap_add
from prepare import NUM_FREQS, custom_freqs

def load_model(checkpoint_path, device):
    # Config matches training
    sr = 16000
    frame_ms = 32.0
    hop_ms = 8.0
    freqs = custom_freqs.to(device)
    
    model = AudioCodec(sr=sr, freqs=freqs, frame_ms=frame_ms, hop_ms=hop_ms)
    
    # Load weights
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def infer(model, input_wav_path, output_dir):
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Audio
    wav, sr = torchaudio.load(input_wav_path)
    if sr != model.sr:
        wav = torchaudio.functional.resample(wav, sr, model.sr)
    
    # Ensure 1D [Time]
    if wav.dim() == 2:
        wav = wav.mean(dim=0)

    # Pad audio to match training preprocessing (Centered Padding)
    frame_len = int(model.sr * model.frame_ms / 1000)
    pad_len = frame_len // 2
    wav = torch.nn.functional.pad(wav, (pad_len, pad_len))

    from AudioComplexNet.utils import frame_audio
    
    # Frame the audio
    # wav is [L]
    frames = frame_audio(wav, model.sr, model.frame_ms, model.hop_ms) # [T, Frame_Len]
    frames = frames.unsqueeze(0).to(device) # [1, T, Frame_Len]
    
    print(f"Input frames shape: {frames.shape}")
    
    with torch.no_grad():
        out = model(frames)
        
    # Reconstruct Waveform from Spectrogram
    # out["real_hat"]: [B, T, F]
    # out["imag_hat"]: [B, T, F]
    
    real_hat = out["real_hat"]
    imag_hat = out["imag_hat"]
    
    # Convert back to time domain frames
    # vector2frame returns windowed frames if training used windowing
    
    recon_frames = vector2frame(real_hat, imag_hat, model.sr, model.freqs, frames.shape[-1]) # [B, T, Frame_Len]
    
    # Overlap-Add with Window Correction
    hop_length = int(model.sr * model.hop_ms / 1000)
    frame_len = frames.shape[-1]
    
    # We need to divide by the sum of squared windows (COLA)
    # Since we used Hanning window in prepare.py
    window = torch.hann_window(frame_len, device=device)
    
    recon_wav = overlap_add(recon_frames[0], hop_length, window)
    recon_wav = recon_wav.squeeze(0)
    
    # Trim padding
    if pad_len > 0:
        recon_wav = recon_wav[pad_len:-pad_len]
    
    input_filename = os.path.splitext(os.path.basename(input_wav_path))[0]
    output_path = os.path.join(output_dir, f"{input_filename}_recon.wav")
    
    torchaudio.save(output_path, recon_wav.unsqueeze(0).cpu(), model.sr)
    print(f"Saved reconstructed audio to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input wav file")
    parser.add_argument("--out_dir", type=str, default="inference_results", help="Output directory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(args.ckpt, device)
    # Needed for OLA helper
    model.hop_length_samples = int(model.sr * model.hop_ms / 1000)
    
    infer(model, args.input, args.out_dir)
