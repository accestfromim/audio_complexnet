
import torch
from AudioComplexNet.modeling_audio_codec import AudioCodec
from prepare import custom_freqs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("Initializing AudioCodec with updated dimensions...")
    # Use default parameters (now updated in modeling_audio_codec.py)
    # hidden_dim=256, latent_dim=256, n_codebook=4096, n_quantizers=8
    
    model = AudioCodec(
        sr=16000, 
        freqs=custom_freqs, 
        causal=False
    )
    
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Total Trainable Parameters (Millions): {total_params/1e6:.2f}M")
    
    # Breakdown
    enc_params = count_parameters(model.encoder)
    dec_params = count_parameters(model.decoder)
    quant_params = count_parameters(model.quantizer)
    
    print("-" * 30)
    print(f"Encoder: {enc_params:,} ({enc_params/1e6:.2f}M)")
    print(f"Decoder: {dec_params:,} ({dec_params/1e6:.2f}M)")
    print(f"Quantizer: {quant_params:,} ({quant_params/1e6:.2f}M)")

if __name__ == "__main__":
    main()
