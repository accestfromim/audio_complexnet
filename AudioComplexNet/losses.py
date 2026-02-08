import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class MelSpectrogramLoss(nn.Module):
    """
    Mel-Spectrogram Loss for Perceptual Quality.
    Standard in HiFi-GAN, MelGAN, etc.
    """
    def __init__(self, 
                 sample_rate=16000, 
                 n_fft=1024, 
                 hop_length=256, 
                 win_length=1024, 
                 n_mels=80, 
                 f_min=0.0, 
                 f_max=None):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=1.0, # Magnitude
            normalized=True
        )
        
    def forward(self, x_hat, x):
        """
        x_hat: Reconstructed Waveform [B, T]
        x: Target Waveform [B, T]
        """
        # Force FP32
        x_hat = x_hat.float()
        x = x.float()
        
        mel_hat = self.mel_transform(x_hat)
        mel = self.mel_transform(x)
        
        # Log Mel Loss (Dynamic Range)
        # Add epsilon and clamp. Using 1e-4 for better gradient stability.
        log_mel_hat = torch.log(torch.clamp(mel_hat, min=1e-4))
        log_mel = torch.log(torch.clamp(mel, min=1e-4))
        
        return F.l1_loss(log_mel_hat, log_mel)

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, 
                 fft_sizes=[512, 1024, 2048], 
                 hop_sizes=[128, 256, 512], 
                 win_lengths=[512, 1024, 2048],
                 window="hann_window"):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window
        
    def forward(self, x_hat, x):
        """
        x_hat: Reconstructed Waveform [B, T]
        x: Target Waveform [B, T]
        """
        # Force FP32 for stability
        x_hat = x_hat.float()
        x = x.float()
        
        loss_mag = 0.0
        loss_log_mag = 0.0
        
        for i in range(len(self.fft_sizes)):
            n_fft = self.fft_sizes[i]
            hop = self.hop_sizes[i]
            win = self.win_lengths[i]
            
            window = getattr(torch, self.window)(win).to(x.device)
            
            # STFT
            x_stft = torch.stft(x, n_fft, hop, win, window, return_complex=True)
            x_hat_stft = torch.stft(x_hat, n_fft, hop, win, window, return_complex=True)
            
            # Magnitude (add epsilon for safety)
            mag = torch.abs(x_stft) + 1e-7
            mag_hat = torch.abs(x_hat_stft) + 1e-7
            
            # Log Magnitude (with clamp)
            # Use 1e-5 for safer log in mixed precision contexts (though we cast to float above)
            log_mag = torch.log(torch.clamp(mag, min=1e-5))
            log_mag_hat = torch.log(torch.clamp(mag_hat, min=1e-5))
            
            # L1 Losses
            loss_mag += F.l1_loss(mag_hat, mag)
            loss_log_mag += F.l1_loss(log_mag_hat, log_mag)
            
        # Average
        loss_mag /= len(self.fft_sizes)
        loss_log_mag /= len(self.fft_sizes)
        
        return loss_mag, loss_log_mag
