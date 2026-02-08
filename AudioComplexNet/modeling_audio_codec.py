
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.nn.utils import spectral_norm
from .utils import frame2vector, vector2frame

class ComplexConv1d(nn.Module):
    """
    Complex Convolution 1D
    Performs convolution with complex weights:
    W = W_r + i W_i
    Input x = x_r + i x_i
    
    Output y = (x_r * W_r - x_i * W_i) + i (x_r * W_i + x_i * W_r)
    * represents convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Weights for real and imaginary parts
        self.weight_real = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.weight_imag = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_real, -bound, bound)
            init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input_real, input_imag):
        
        
        # Reference Logic:
        # Real_out = Conv(Real_in, Real_W) + Conv(Imag_in, Imag_W)
        # Imag_out = Conv(Real_in, Imag_W) - Conv(Imag_in, Real_W)
        
        real_real = F.conv1d(input_real, self.weight_real, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        imag_imag = F.conv1d(input_imag, self.weight_imag, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        
        real_imag = F.conv1d(input_real, self.weight_imag, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        imag_real = F.conv1d(input_imag, self.weight_real, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        
        out_real = real_real + imag_imag
        out_imag = real_imag - imag_real
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real.view(1, -1, 1)
            out_imag = out_imag + self.bias_imag.view(1, -1, 1)
            
        return out_real, out_imag

class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        self.weight_real = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.weight_imag = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, kernel_size))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            init.zeros_(self.bias_real)
            init.zeros_(self.bias_imag)

    def forward(self, input_real, input_imag):
        # Consistent with ComplexConv1d logic (Conjugate logic)
        # Real_out = ConvT(Real_in, Real_W) + ConvT(Imag_in, Imag_W)
        # Imag_out = ConvT(Real_in, Imag_W) - ConvT(Imag_in, Real_W)
        
        real_real = F.conv_transpose1d(input_real, self.weight_real, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        imag_imag = F.conv_transpose1d(input_imag, self.weight_imag, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        
        real_imag = F.conv_transpose1d(input_real, self.weight_imag, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        imag_real = F.conv_transpose1d(input_imag, self.weight_real, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        
        out_real = real_real + imag_imag
        out_imag = real_imag - imag_real
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real.view(1, -1, 1)
            out_imag = out_imag + self.bias_imag.view(1, -1, 1)
            
        return out_real, out_imag

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight_real = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.weight_imag = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_real, -bound, bound)
            init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input_real, input_imag):
        real_real = F.conv2d(input_real, self.weight_real, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        imag_imag = F.conv2d(input_imag, self.weight_imag, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        
        real_imag = F.conv2d(input_real, self.weight_imag, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        imag_real = F.conv2d(input_imag, self.weight_real, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        
        out_real = real_real + imag_imag
        out_imag = real_imag - imag_real
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real.view(1, -1, 1, 1)
            out_imag = out_imag + self.bias_imag.view(1, -1, 1, 1)
            
        return out_real, out_imag

class ComplexBatchNorm1d(nn.Module):
    """
    Magnitude-based Complex Batch Normalization.
    Normalizes the input by its magnitude variance, preserving phase distribution shape relative to the center.
    
    Logic:
    1. Center the data: z = z - E[z]
    2. Normalize by magnitude: z = z / sqrt(E[|z|^2] + eps)
    3. Affine transform: z = z * gamma + beta (gamma is real scaling, beta is complex shift)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features)) # Gamma (Real scaling)
            self.bias_real = nn.Parameter(torch.Tensor(num_features)) # Beta Real
            self.bias_imag = nn.Parameter(torch.Tensor(num_features)) # Beta Imag
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.register_buffer('running_mean_real', torch.zeros(num_features))
        self.register_buffer('running_mean_imag', torch.zeros(num_features))
        self.register_buffer('running_var_mag', torch.ones(num_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias_real)
            init.zeros_(self.bias_imag)
        # Running stats are already init to 0/1

    def forward(self, input_real, input_imag):
        # Input: [B, C, T]
        
        if self.training:
            # Calculate mean across Batch and Time
            # mean: [1, C, 1]
            mean_r = input_real.mean([0, 2], keepdim=True)
            mean_i = input_imag.mean([0, 2], keepdim=True)
            
            # Center
            centered_r = input_real - mean_r
            centered_i = input_imag - mean_i
            
            # Calculate magnitude variance: E[|z-mu|^2] = E[r^2 + i^2]
            # var: [1, C, 1]
            mag_sq = centered_r ** 2 + centered_i ** 2
            var_mag = mag_sq.mean([0, 2], keepdim=True)
            
            # Update running stats
            with torch.no_grad():
                self.running_mean_real.mul_(1 - self.momentum).add_(mean_r.squeeze() * self.momentum)
                self.running_mean_imag.mul_(1 - self.momentum).add_(mean_i.squeeze() * self.momentum)
                self.running_var_mag.mul_(1 - self.momentum).add_(var_mag.squeeze() * self.momentum)
        else:
            # Use running stats
            mean_r = self.running_mean_real.view(1, -1, 1)
            mean_i = self.running_mean_imag.view(1, -1, 1)
            var_mag = self.running_var_mag.view(1, -1, 1)
            
            centered_r = input_real - mean_r
            centered_i = input_imag - mean_i

        # Normalize
        std_mag = torch.sqrt(var_mag + self.eps)
        inv_std = 1.0 / std_mag
        
        norm_r = centered_r * inv_std
        norm_i = centered_i * inv_std
        
        # Affine
        if self.affine:
            # gamma is real scalar per channel
            weight = self.weight.view(1, -1, 1)
            bias_r = self.bias_real.view(1, -1, 1)
            bias_i = self.bias_imag.view(1, -1, 1)
            
            out_r = norm_r * weight + bias_r
            out_i = norm_i * weight + bias_i
        else:
            out_r = norm_r
            out_i = norm_i
            
        return out_r, out_i

class ComplexReLU(nn.Module):
    """
    Custom Complex Activation.
    Logic: If (Real < 0 AND Imag < 0), set to 0. Else identity.
    "Masked" activation for the 3rd quadrant.
    """
    def forward(self, real, imag):
        # Condition: Keep if NOT (real < 0 AND imag < 0)
        # Equivalent: Keep if (real >= 0 OR imag >= 0)
        
        # Note: We want to set to 0 ONLY if BOTH are negative.
        # mask = 1 if (r>=0 or i>=0), mask = 0 if (r<0 and i<0)
        
        mask = (real >= 0) | (imag >= 0)
        mask = mask.float()
        
        return real * mask, imag * mask

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, causal=False):
        super().__init__()
        self.conv1 = CausalComplexConv1d(channels, channels, kernel_size, dilation=dilation, causal=causal)
        self.bn1 = ComplexBatchNorm1d(channels)
        self.act1 = ComplexReLU()
        self.conv2 = CausalComplexConv1d(channels, channels, kernel_size, dilation=dilation, causal=causal) 
        self.bn2 = ComplexBatchNorm1d(channels)
        self.act2 = ComplexReLU()

    def forward(self, x_real, x_imag):
        residual_real, residual_imag = x_real, x_imag
        
        y_real, y_imag = self.conv1(x_real, x_imag)
        y_real, y_imag = self.bn1(y_real, y_imag)
        y_real, y_imag = self.act1(y_real, y_imag)
        
        y_real, y_imag = self.conv2(y_real, y_imag)
        y_real, y_imag = self.bn2(y_real, y_imag)
        y_real, y_imag = self.act2(y_real, y_imag)
        
        return residual_real + y_real, residual_imag + y_imag

class CausalComplexConv1d(nn.Module):
    """
    Causal Complex Convolution.
    If causal=True, Pads only on the left.
    If causal=False, Uses symmetric padding (Same).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, causal=False):
        super().__init__()
        self.causal = causal
        self.padding_val = (kernel_size - 1) * dilation
        
        if causal:
            self.conv = ComplexConv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation)
        else:
            # Symmetric padding
            pad = self.padding_val // 2
            self.conv = ComplexConv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation)
        
    def forward(self, x_real, x_imag):
        if self.causal:
            # Pad left
            x_real = F.pad(x_real, (self.padding_val, 0))
            x_imag = F.pad(x_imag, (self.padding_val, 0))
        return self.conv(x_real, x_imag)

class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantization module.
    """
    def __init__(self, n_e, e_dim, beta=10.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e) # Too small for large n_e
        self.embedding.weight.data.normal_(mean=0.0, std=0.02) # Standard initialization

    def forward(self, z):
        # z: [b, c, t] -> [b, t, c]
        # FORCE DETACH Z for GAN Fine-tuning
        # This prevents gradients flowing back to Encoder and High Grad Norm
        # Also allows Codebook to update (via Codebook Loss) without dragging Encoder
        z = z.detach() 
        
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2ze
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        # Modified for Frozen Encoder Fine-tuning:
        # z is already detached at start of forward.
        # So we only compute Codebook Loss (update e towards z).
        # We remove beta scaling because this is the primary loss for Codebook now.
        loss = torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to [b, c, t]
        z_q = z_q.permute(0, 2, 1).contiguous()
        
        return z_q, loss, min_encoding_indices.view(z.shape[0], z.shape[1])

class ResidualVQ(nn.Module):
    """
    Residual Vector Quantizer.
    """
    def __init__(self, num_quantizers, n_e, e_dim, beta=10.0):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim, beta) for _ in range(num_quantizers)
        ])

    def forward(self, x):
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, loss, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_indices.append(indices)
        
        # Stack indices: [B, T, N_layers]
        all_indices = torch.stack(all_indices, dim=-1)
        # Return individual losses for weighting
        # total_loss = torch.stack(all_losses).sum()
        
        return quantized_out, all_losses, all_indices

    def from_codes(self, codes):
        """
        Reconstruct quantized latents from codes.
        codes: [B, T, N_quantizers]
        """
        quantized_out = 0.0
        # codes is [B, T, N_q]
        # We need to iterate over layers
        
        for i, layer in enumerate(self.layers):
            # layer codes: [B, T]
            idx = codes[..., i]
            # get embeddings: [B, T, D]
            z_q = layer.embedding(idx)
            # permute to [B, D, T]
            z_q = z_q.permute(0, 2, 1)
            quantized_out = quantized_out + z_q
            
        return quantized_out

class ComplexEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.causal = causal
        
        # Initial Conv (Causal)
        self.conv_in = CausalComplexConv1d(in_channels, hidden_dim, 7, causal=causal)
        self.bn_in = ComplexBatchNorm1d(hidden_dim)
        self.act_in = ComplexReLU()
        
        # Downsampling blocks with dilation for large receptive field
        self.blocks = nn.ModuleList([
            # Stride 2 (120 -> 60)
            CausalComplexConv1d(hidden_dim, hidden_dim, 3, stride=2, causal=causal), # Down 1
            ResBlock(hidden_dim, 3, dilation=3, causal=causal),
            ResBlock(hidden_dim, 3, dilation=9, causal=causal),
            
            # Stride 1 (60 -> 60)
            CausalComplexConv1d(hidden_dim, hidden_dim, 3, stride=1, causal=causal), # Down 2 (Stride 1)
            ResBlock(hidden_dim, 3, dilation=3, causal=causal),
            ResBlock(hidden_dim, 3, dilation=9, causal=causal),
        ])
        
        self.conv_out = CausalComplexConv1d(hidden_dim, latent_dim, 3, causal=causal)
        self.act_block = ComplexReLU()

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.conv_in(x_real, x_imag)
        x_real, x_imag = self.bn_in(x_real, x_imag)
        x_real, x_imag = self.act_in(x_real, x_imag)
        
        for block in self.blocks:
            if isinstance(block, CausalComplexConv1d):
                x_real, x_imag = block(x_real, x_imag)
                x_real, x_imag = self.act_block(x_real, x_imag)
            else:
                x_real, x_imag = block(x_real, x_imag)
                
        x_real, x_imag = self.conv_out(x_real, x_imag)
        return x_real, x_imag

class ComplexDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_channels, causal=False):
        super().__init__()
        self.causal = causal
        
        self.conv_in = CausalComplexConv1d(latent_dim, hidden_dim, 7, causal=causal)
        self.act_in = ComplexReLU()
        
        # Upsampling blocks (Mirror Encoder)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, 3, dilation=9, causal=causal),
            ResBlock(hidden_dim, 3, dilation=3, causal=causal),
            ComplexConvTranspose1d(hidden_dim, hidden_dim, 3, stride=1, padding=1, output_padding=0), # Up 1 (Stride 1)
            
            ResBlock(hidden_dim, 3, dilation=9, causal=causal),
            ResBlock(hidden_dim, 3, dilation=3, causal=causal),
            ComplexConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1), # Up 2
        ])
        
        self.conv_out = CausalComplexConv1d(hidden_dim, out_channels, 7, causal=causal)
        self.act_block = ComplexReLU()

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.conv_in(x_real, x_imag)
        x_real, x_imag = self.act_in(x_real, x_imag)
        
        for block in self.blocks:
            if isinstance(block, ComplexConvTranspose1d):
                x_real, x_imag = block(x_real, x_imag)
                x_real, x_imag = self.act_block(x_real, x_imag)
            else:
                x_real, x_imag = block(x_real, x_imag)
                
        x_real, x_imag = self.conv_out(x_real, x_imag)
        return x_real, x_imag

class SingleScaleDiscriminator(nn.Module):
    """
    PatchGAN Discriminator on Concatenated Real/Imag Spectrograms.
    Input: Complex Spectrogram [B, F, T] (Real, Imag) separated.
    Output: Real score map.
    """
    def __init__(self, input_channels=2, hidden_dim=64): # Reduced base dim for Multi-Scale
        super().__init__()
        
        self.conv_in = spectral_norm(nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1))
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
             nn.Sequential(
                spectral_norm(nn.Conv2d(hidden_dim * 8, hidden_dim * 16, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        
        self.conv_out = spectral_norm(nn.Conv2d(hidden_dim * 16, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, x):
        # Input: [B, 2, F, T]
        features = []
        
        x = self.conv_in(x)
        x = F.leaky_relu(x, 0.2)
        features.append(x)
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
            
        x = self.conv_out(x)
        features.append(x)
        
        return x, features

class Discriminator(nn.Module):
    """
    Multi-Scale Discriminator (3 scales)
    """
    def __init__(self, input_channels=2, hidden_dim=64):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SingleScaleDiscriminator(input_channels, hidden_dim),
            SingleScaleDiscriminator(input_channels, hidden_dim),
            SingleScaleDiscriminator(input_channels, hidden_dim)
        ])
        
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, real, imag, mag=None):
        # Input: [B, T, F]
        # Treat Freq as Height (H), Time as Width (W).
        # We want [B, C, H, W] = [B, 2, F, T] (or 3 if mag provided)
        
        if real.dim() == 3:
            # [B, T, F] -> [B, F, T]
            real = real.permute(0, 2, 1)
            imag = imag.permute(0, 2, 1)
            if mag is not None:
                mag = mag.permute(0, 2, 1)
        
        # Stack to [B, C, F, T]
        if mag is not None:
            x = torch.stack([real, imag, mag], dim=1)
        else:
            x = torch.stack([real, imag], dim=1)
        
        outputs = []
        features_list = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            score, feats = d(x)
            outputs.append(score)
            features_list.append(feats)
            
        return outputs, features_list

class AudioCodec(nn.Module):
    def __init__(self, sr=16000, freqs=None, frame_ms=32.0, hop_ms=8.0, 
                 hidden_dim=512, latent_dim=1536, n_codebook=4096, n_quantizers=8, causal=False,
                 quantizer_weights=None):
        super().__init__()
        self.sr = sr
        self.register_buffer("freqs", freqs)
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        # With hop_ms=8.0ms -> 125Hz STFT
        # Encoder Downsample x2 -> 62.5Hz Tokens
        self.causal = causal
        
        # RVQ Weights
        if quantizer_weights is None:
            # Default: Uniform weights
            quantizer_weights = [1.0] * n_quantizers
        assert len(quantizer_weights) == n_quantizers, "quantizer_weights must match n_quantizers"
        self.register_buffer("quantizer_weights", torch.tensor(quantizer_weights, dtype=torch.float32))
        
        num_freqs = freqs.numel()
        # Input channels for Complex Conv is num_freqs (Real part) + num_freqs (Imag part)
        # But ComplexConv1d expects in_channels to mean "Complex Channels".
        # Since our input is [Freq_1, Freq_2, ...], we have num_freqs "channels".
        
        self.encoder = ComplexEncoder(num_freqs, hidden_dim, latent_dim, causal=causal)
        
        # RVQ is real-valued. We need to project Complex Latent -> Real Latent for Quantization
        # Or quantize Real/Imag separately? 
        # User accepted "Joint Quantization" previously.
        # So we flatten Complex Latent [B, Latent, T] -> [B, 2*Latent, T] -> RVQ -> [B, 2*Latent, T] -> Reshape back.
        
        self.quantizer = ResidualVQ(n_quantizers, n_codebook, latent_dim * 2) # *2 for Real+Imag
        
        self.decoder = ComplexDecoder(latent_dim, hidden_dim, num_freqs, causal=causal)

    def forward(self, frames):
        """
        frames: [B, T, Frame_Len]
        """
        spec_vector = frame2vector(frames, self.sr, self.freqs) # [B, T, F] Complex
        
        # --- Power-Law Compression (Alpha=0.3) ---
        # Helps with High Freq reconstruction by compressing dynamic range
        # X_compressed = |X|^0.3 * exp(j * angle(X))
        mag = torch.abs(spec_vector)
        phase = torch.angle(spec_vector)
        # Avoid zero division or nan
        mag = torch.clamp(mag, min=1e-7)
        mag_compressed = torch.pow(mag, 0.3)
        spec_compressed = torch.polar(mag_compressed, phase)
        
        real = spec_compressed.real
        imag = spec_compressed.imag
        
        # Prepare for Conv1d: [B, F, T]
        real_in = real.permute(0, 2, 1)
        imag_in = imag.permute(0, 2, 1)
        
        # Encode
        z_real, z_imag = self.encoder(real_in, imag_in) # [B, Latent, T']
        
        # Flatten for Quantization
        z = torch.cat([z_real, z_imag], dim=1) # [B, 2*Latent, T']
        
        # Quantize
        z_q, all_losses, codes = self.quantizer(z)
        
        # Compute Weighted Commitment Loss
        commit_loss = 0.0
        for i, loss in enumerate(all_losses):
            commit_loss += loss * self.quantizer_weights[i]
        
        # Unflatten
        z_q_real, z_q_imag = torch.chunk(z_q, 2, dim=1)
        
        # Decode
        x_hat_real, x_hat_imag = self.decoder(z_q_real, z_q_imag) # [B, F, T]
        
        # Post-process
        real_hat = x_hat_real.permute(0, 2, 1) # [B, T, F]
        imag_hat = x_hat_imag.permute(0, 2, 1)
        
        # --- Inverse Power-Law Compression ---
        # X_recon = |X_hat|^(1/0.3) * exp(j * angle(X_hat))
        spec_hat_compressed = torch.complex(real_hat, imag_hat)
        mag_hat_compressed = torch.abs(spec_hat_compressed)
        phase_hat = torch.angle(spec_hat_compressed)
        
        # Safety: Clamp compressed magnitude to avoid explosion during inverse power (x^3.33)
        # 50^3.33 is ~450,000, which is large but safe for float32. 
        # 100^3.33 is ~4.6 million.
        # If the model outputs garbage > 100, we get explosion.
        mag_hat_compressed = torch.clamp(mag_hat_compressed, max=50.0)
        
        mag_hat = torch.pow(mag_hat_compressed, 1.0/0.3)
        # Re-construct complex
        spec_hat = torch.polar(mag_hat, phase_hat)
        
        real_hat = spec_hat.real
        imag_hat = spec_hat.imag
        
        # Note: We return both the "compressed" output (for latent loss?) 
        # No, usually we compute loss on the final decompressed output.
        
        # Align shapes if needed (due to Conv/Deconv padding differences)
        if real_hat.shape[1] != real.shape[1]:
            # Crop to min length (Reference is original Real/Imag derived from Spec)
            # But wait, 'real' variable above is COMPRESSED.
            # We need original uncompressed 'spec_vector' for reference if we want to return it.
            # Let's realign everything to 'spec_vector' shape
            min_len = min(real_hat.shape[1], spec_vector.shape[1])
            real_hat = real_hat[:, :min_len, :]
            imag_hat = imag_hat[:, :min_len, :]
            # real = real[:, :min_len, :] # This is compressed
            # imag = imag[:, :min_len, :] # This is compressed
            
            # Update spec_vector crop too for consistency in return
            # spec_vector = spec_vector[:, :min_len, :]
            
            # Actually, let's just crop outputs.
        
        frames_hat = vector2frame(real_hat, imag_hat, self.sr, self.freqs, frames.shape[-1])
        
        return {
            "frames_hat": frames_hat,
            "real_hat": real_hat,
            "imag_hat": imag_hat,
            "real": spec_vector.real, # Return UNCOMPRESSED original for Loss
            "imag": spec_vector.imag, # Return UNCOMPRESSED original for Loss
            "commit_loss": commit_loss,
            "codes": codes
        }

    def decode_from_codes(self, codes):
        """
        Decode from discrete codes.
        codes: [B, T, N_quantizers]
        """
        # Get quantized latents
        z_q = self.quantizer.from_codes(codes) # [B, 2*Latent, T]
        
        # Unflatten
        z_q_real, z_q_imag = torch.chunk(z_q, 2, dim=1)
        
        # Decode
        x_hat_real, x_hat_imag = self.decoder(z_q_real, z_q_imag) # [B, F, T]
        
        # Post-process
        real_hat = x_hat_real.permute(0, 2, 1) # [B, T, F]
        imag_hat = x_hat_imag.permute(0, 2, 1)
        
        # We can estimate frame_len from sr and frame_ms.
        frame_len = int(self.sr * self.frame_ms / 1000)
        
        frames_hat = vector2frame(real_hat, imag_hat, self.sr, self.freqs, frame_len)
        
        return frames_hat
