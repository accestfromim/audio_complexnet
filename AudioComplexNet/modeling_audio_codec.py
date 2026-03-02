
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.nn.utils import spectral_norm, weight_norm
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
        from torch.nn.modules.utils import _pair
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight_real = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.weight_imag = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        
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
    """
    Bottleneck Residual Block
    Structure:
    1. 1x1 Conv (Expansion): Channel Mixing / Up-projection
    2. 3x3 Conv (Temporal): Contextual Processing (with Dilation)
    3. 1x1 Conv (Reduction): Channel Mixing / Down-projection
    """
    def __init__(self, channels, kernel_size=3, dilation=1, causal=False, expansion=2):
        super().__init__()
        hidden_channels = channels * expansion
        
        # 1. Expansion (1x1 Linear-like mixing)
        self.conv1 = ComplexConv1d(channels, hidden_channels, kernel_size=1)
        self.bn1 = ComplexBatchNorm1d(hidden_channels)
        self.act1 = ComplexReLU()
        
        # 2. Temporal Processing (3x3 with Dilation)
        self.conv2 = CausalComplexConv1d(hidden_channels, hidden_channels, kernel_size, dilation=dilation, causal=causal) 
        self.bn2 = ComplexBatchNorm1d(hidden_channels)
        self.act2 = ComplexReLU()
        
        # 3. Reduction (1x1 Linear-like mixing)
        self.conv3 = ComplexConv1d(hidden_channels, channels, kernel_size=1)
        self.bn3 = ComplexBatchNorm1d(channels)
        # No activation after final projection, similar to Transformer FFN / ResNet Bottleneck

    def forward(self, x_real, x_imag):
        residual_real, residual_imag = x_real, x_imag
        
        # 1. Expansion
        y_real, y_imag = self.conv1(x_real, x_imag)
        y_real, y_imag = self.bn1(y_real, y_imag)
        y_real, y_imag = self.act1(y_real, y_imag)
        
        # 2. Temporal
        y_real, y_imag = self.conv2(y_real, y_imag)
        y_real, y_imag = self.bn2(y_real, y_imag)
        y_real, y_imag = self.act2(y_real, y_imag)
        
        # 3. Reduction
        y_real, y_imag = self.conv3(y_real, y_imag)
        y_real, y_imag = self.bn3(y_real, y_imag)
        
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
        # Codebook Loss: update embeddings to match encoder outputs
        loss_codebook = torch.mean((z_q - z.detach())**2)
        # Commitment Loss: update encoder to produce outputs close to embeddings
        loss_commit = torch.mean((z_q.detach() - z)**2)
        
        loss = loss_codebook + self.beta * loss_commit

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
        
        if isinstance(n_e, int):
            n_e = [n_e] * num_quantizers
        assert len(n_e) == num_quantizers, "n_e list must match num_quantizers"
        
        self.layers = nn.ModuleList([
            VectorQuantizer(n_e[i], e_dim, beta) for i in range(num_quantizers)
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
        # Increased capacity: [1, 3, 9] dilation pattern
        self.blocks = nn.ModuleList([
            # Stride 2 (120 -> 60)
            CausalComplexConv1d(hidden_dim, hidden_dim, 3, stride=2, causal=causal), # Down 1
            ResBlock(hidden_dim, 3, dilation=1, causal=causal),
            ResBlock(hidden_dim, 3, dilation=3, causal=causal),
            ResBlock(hidden_dim, 3, dilation=9, causal=causal),
            
            # Stride 2 (60 -> 30)
            CausalComplexConv1d(hidden_dim, hidden_dim, 3, stride=2, causal=causal), # Down 2 (Stride 2)
            ResBlock(hidden_dim, 3, dilation=1, causal=causal),
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
        # Increased capacity: [1, 3, 9] dilation pattern
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, 3, dilation=9, causal=causal),
            ResBlock(hidden_dim, 3, dilation=3, causal=causal),
            ResBlock(hidden_dim, 3, dilation=1, causal=causal),
            ComplexConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1), # Up 1 (Stride 2)
            
            ResBlock(hidden_dim, 3, dilation=9, causal=causal),
            ResBlock(hidden_dim, 3, dilation=3, causal=causal),
            ResBlock(hidden_dim, 3, dilation=1, causal=causal),
            ComplexConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1), # Up 2 (Stride 2)
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

class ComplexSingleScaleDiscriminator(nn.Module):
    """
    Complex PatchGAN Discriminator.
    Input: Complex Spectrogram [B, 1, F, T] (Real, Imag).
    Output: Real score map.
    """
    def __init__(self, input_channels=1, hidden_dim=64):
        super().__init__()
        
        # Helper to apply SN to ComplexConv2d
        def complex_sn(layer):
            layer = spectral_norm(layer, name='weight_real')
            layer = spectral_norm(layer, name='weight_imag')
            return layer

        self.conv_in = complex_sn(ComplexConv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1))
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                complex_sn(ComplexConv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)),
                ComplexReLU(),
            ),
            # Removed the 2nd block to weaken the discriminator
            # nn.Sequential(
            #     complex_sn(ComplexConv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)),
            #     ComplexReLU(),
            # ),
            # nn.Sequential(
            #     complex_sn(ComplexConv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)),
            #     ComplexReLU(),
            # ),
        ])
        
        # Reduced output channel depth to match the removed block (hidden_dim * 2)
        # Restore Spectral Norm to stabilize output range and prevent exploding gradients
        self.conv_out = complex_sn(ComplexConv2d(hidden_dim * 2, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x_real, x_imag):
        # Input: [B, C, F, T]
        features = []
        
        x_real, x_imag = self.conv_in(x_real, x_imag)
        x_real, x_imag = ComplexReLU()(x_real, x_imag) # Leaky ReLU equivalent? ComplexReLU masks.
        # Standard ComplexReLU is like ReLU. For D we usually want LeakyReLU.
        # But ComplexLeakyReLU is tricky. 
        # User said "Reference G's implementation". G uses ComplexReLU.
        # I will use ComplexReLU.
        
        features.append((x_real, x_imag))
        
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, ComplexConv2d):
                    x_real, x_imag = layer(x_real, x_imag)
                else:
                    x_real, x_imag = layer(x_real, x_imag)
            features.append((x_real, x_imag))
            
        x_real, x_imag = self.conv_out(x_real, x_imag)
        features.append((x_real, x_imag))
        
        # Return Real part as score (Hermitian projection concept)
        return x_real, features

class ComplexMultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Complex Discriminator.
    Consists of 3 ComplexSingleScaleDiscriminators operating on:
    1. Original Resolution
    2. x2 Downsampled
    3. x4 Downsampled
    """
    def __init__(self, input_channels=1, hidden_dim=64, n_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ComplexSingleScaleDiscriminator(input_channels, hidden_dim) for _ in range(n_scales)
        ])
        # Average Pooling for downsampling (preserves complex structure linearly)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x_real, x_imag):
        # x_real, x_imag: [B, C, F, T]
        outputs = []
        features = []
        
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x_real = self.downsample(x_real)
                x_imag = self.downsample(x_imag)
            
            score, feats = d(x_real, x_imag)
            outputs.append(score)
            features.append(feats)
            
        return outputs, features

class DiscriminatorR(nn.Module):
    """
    STFT-based Discriminator (Resolution Discriminator).
    Operates on (Real, Imag) of STFT.
    """
    def __init__(self, n_fft, hop_length, win_length, use_spectral_norm=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(2, 16, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(16, 16, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(16, 16, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(16, 16, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, x):
        # x: [B, 1, T]
        # STFT
        x_stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length, 
                            window=torch.hann_window(self.win_length, device=x.device),
                            return_complex=True)
        # x_stft: [B, F, T] (Complex)
        
        # Concat Real/Imag: [B, 2, F, T]
        # Need to permute F and T for Conv2d? Conv2d is (B, C, H, W).
        # Usually H=Freq, W=Time.
        # x_stft is (B, F, T).
        # We want (B, 2, F, T).
        
        x_in = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        
        fmap = []
        for l in self.convs:
            x_in = l(x_in)
            x_in = F.leaky_relu(x_in, 0.1)
            fmap.append(x_in)
        
        x_in = self.conv_post(x_in)
        fmap.append(x_in)
        x_in = torch.flatten(x_in, 1, -1)
        
        return x_in, fmap

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        # Resolutions: (n_fft, hop_length, win_length)
        # Reduced resolutions to save memory (Removed 1024)
        self.resolutions = [
            (512, 50, 240),
            (256, 30, 120),
            (128, 15, 60)
        ]
        
        self.discriminators = nn.ModuleList([
            DiscriminatorR(n_fft, hop, win, use_spectral_norm=use_spectral_norm)
            for (n_fft, hop, win) in self.resolutions
        ])

    def forward(self, x):
        outputs = []
        features = []
        for d in self.discriminators:
            score, feats = d(x)
            outputs.append(score)
            features.append(feats)
        return outputs, features

class Discriminator(nn.Module):
    # Deprecated: Kept alias for compatibility if needed, but we will switch CombinedDiscriminator to use MRD.
    def __init__(self, input_channels=1, hidden_dim=64):
        super().__init__()
        self.mrd = MultiResolutionDiscriminator()
    def forward(self, x):
        return self.mrd(x)

class DiscriminatorP(nn.Module):
    """
    Period Discriminator (MPD) from HiFi-GAN.
    Operates on raw waveform by reshaping 1D -> 2D with specific period.
    """
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, channel_multiplier=1.0):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Reduced channels by default if channel_multiplier < 1
        ch = lambda x: int(x * channel_multiplier)
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, ch(32), (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(ch(32), ch(128), (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(ch(128), ch(512), (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(ch(512), ch(1024), (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(ch(1024), ch(1024), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(ch(1024), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) container.
    """
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        # Standard HiFi-GAN periods are [2, 3, 5, 7, 11]
        # Reduced to [2, 3, 5, 7, 11] but with 0.5x channels to save memory
        # Or reduce periods to [3, 7, 11]
        self.periods = [3, 7, 11] # Reduced number of periods
        
        # Use 0.25x channels to prevent overpowering G and save memory
        self.discriminators = nn.ModuleList([
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm, channel_multiplier=0.25) for p in self.periods
        ])

    def forward(self, x):
        # x: [B, 1, T]
        outputs = []
        features_list = []
        for d in self.discriminators:
            score, feats = d(x)
            outputs.append(score)
            features_list.append(feats)
        return outputs, features_list

class CombinedDiscriminator(nn.Module):
    """
    Combines Spectral Discriminator (MRD) and Waveform Discriminator (MPD).
    Provides orthogonal views for GAN training.
    """
    def __init__(self, input_channels=1, hidden_dim=64):
        super().__init__()
        # 1. Spectral Discriminator (New - MRD)
        self.spectral_discriminator = MultiResolutionDiscriminator()
        
        # 2. Waveform Discriminator (MPD)
        self.waveform_discriminator = MultiPeriodDiscriminator()

    def forward(self, waveform):
        """
        waveform: Raw Waveform [B, 1, T]
        """
        # Ensure waveform is [B, 1, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # 1. Spectral Branch (MRD takes waveform)
        spec_scores, spec_feats = self.spectral_discriminator(waveform)
        
        # 2. Waveform Branch
        wave_scores, wave_feats = self.waveform_discriminator(waveform)
            
        # Combine results
        return spec_scores + wave_scores, spec_feats + wave_feats

class AudioCodec(nn.Module):
    def __init__(self, sr=16000, freqs=None, frame_ms=32.0, hop_ms=10.0, 
                 hidden_dim=512, latent_dim=256, n_codebook=None, n_quantizers=16, causal=False,
                 quantizer_weights=None):
        super().__init__()
        self.sr = sr
        self.register_buffer("freqs", freqs)
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        # With hop_ms=10.0ms -> 100Hz STFT
        # Encoder Downsample x4 -> 25Hz Tokens
        self.causal = causal
        
        # RVQ Weights
        if quantizer_weights is None:
            # Default: Decaying weights (0.8^i) to encourage early layers, consistent with training
            quantizer_weights = [1.0 * (0.8 ** i) for i in range(n_quantizers)]
        assert len(quantizer_weights) == n_quantizers, "quantizer_weights must match n_quantizers"
        self.register_buffer("quantizer_weights", torch.tensor(quantizer_weights, dtype=torch.float32))

        # Default Codebook Sizes (Descending)
        if n_codebook is None:
            if n_quantizers == 16:
                n_codebook = [1024, 1024, 512, 512, 256, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
            elif n_quantizers == 8:
                n_codebook = [1024, 1024, 512, 512, 256, 256, 128, 128]
            else:
                n_codebook = 1024 # Fallback to uniform 1024 if not 8 layers
        
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

    def forward(self, frames, quantize=True):
        """
        frames: [B, T, Frame_Len]
        quantize: Whether to use VQ or pass continuous latents (with Tanh)
        """
        spec_vector = frame2vector(frames, self.sr, self.freqs) # [B, T, F] Complex
        
        # --- Power-Law Compression (Alpha=0.3) ---
        # Helps with High Freq reconstruction by compressing dynamic range
        # X_compressed = |X|^0.3 * exp(j * angle(X))
        # Use Algebraic Scaling to avoid NaN gradients in atan2/polar
        # z_c = z * |z|^(alpha-1)
        
        mag_sq = spec_vector.real**2 + spec_vector.imag**2
        mag = torch.sqrt(mag_sq + 1e-5) # eps=1e-5 for safety
        
        # scale = mag^(0.3 - 1) = mag^(-0.7)
        # Avoid division by zero
        scale = torch.pow(mag, 0.3 - 1.0)
        # If mag is very small, scale becomes huge. 
        # But z is small. z * scale -> 0 * inf = NaN.
        # Better: scale = mag^0.3 / mag
        # z_c = z * (mag^0.3 / (mag + eps))
        
        mag_compressed = torch.pow(mag, 0.3)
        scale_safe = mag_compressed / (mag + 1e-6)
        
        real = spec_vector.real * scale_safe
        imag = spec_vector.imag * scale_safe
        
        # Prepare for Conv1d: [B, F, T]
        real_in = real.permute(0, 2, 1)
        imag_in = imag.permute(0, 2, 1)
        
        # Encode
        z_real, z_imag = self.encoder(real_in, imag_in) # [B, Latent, T']
        
        # Flatten for Quantization
        z = torch.cat([z_real, z_imag], dim=1) # [B, 2*Latent, T']
        
        if quantize:
            # Quantize
            # Continuous Training Mode used Tanh, so we must apply Tanh here too
            # to match the distribution of the embeddings trained with Tanh-clamped inputs.
            # z = torch.tanh(z)
            z_q, all_losses, codes = self.quantizer(z)
            
            # Compute Weighted Commitment Loss
            commit_loss = 0.0
            for i, loss in enumerate(all_losses):
                commit_loss += loss * self.quantizer_weights[i]
        else:
            # Continuous Training Mode
            # Apply Tanh to constrain latent space to [-1, 1] for future clustering
            z_q = torch.tanh(z)
            commit_loss = torch.tensor(0.0, device=z.device)
            codes = None
            all_losses = []
        
        # Unflatten
        z_q_real, z_q_imag = torch.chunk(z_q, 2, dim=1)
        
        # Decode
        x_hat_real, x_hat_imag = self.decoder(z_q_real, z_q_imag) # [B, F, T]
        
        # Post-process
        real_hat = x_hat_real.permute(0, 2, 1) # [B, T, F]
        imag_hat = x_hat_imag.permute(0, 2, 1)
        
        # --- Inverse Power-Law Compression ---
        # X_recon = |X_hat|^(1/0.3) * exp(j * angle(X_hat))
        # Algebraic: z = z_c * |z_c|^(1/alpha - 1)
        
        mag_sq_hat = real_hat**2 + imag_hat**2
        mag_hat_compressed = torch.sqrt(mag_sq_hat + 1e-5)
        
        # Safety clamp to prevent explosion
        mag_hat_compressed = torch.clamp(mag_hat_compressed, max=50.0)
        
        # scale_inv = mag_hat^(1/0.3 - 1) = mag_hat^(2.33)
        inv_alpha = 1.0 / 0.3
        scale_inv = torch.pow(mag_hat_compressed, inv_alpha - 1.0)
        
        # spec_hat = z_hat * scale_inv
        real_hat_decomp = real_hat * scale_inv
        imag_hat_decomp = imag_hat * scale_inv

        
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
            real_hat_decomp = real_hat_decomp[:, :min_len, :]
            imag_hat_decomp = imag_hat_decomp[:, :min_len, :]
            
            # Crop targets (Compressed)
            real = real[:, :min_len, :]
            imag = imag[:, :min_len, :]
            
            # Crop targets (Uncompressed) - For completeness if needed, though we use compressed for loss
            spec_vector = spec_vector[:, :min_len, :]
            
            # Actually, let's just crop outputs.
        
        # Use DECOMPRESSED spectrum for waveform reconstruction
        frames_hat = vector2frame(real_hat_decomp, imag_hat_decomp, self.sr, self.freqs, frames.shape[-1])
        
        return {
            "frames_hat": frames_hat,
            "real_hat": real_hat,         # Compressed Prediction
            "imag_hat": imag_hat,         # Compressed Prediction
            "real": real,                 # Compressed Target
            "imag": imag,                 # Compressed Target
            "real_raw": spec_vector.real, # Uncompressed Target (Optional)
            "imag_raw": spec_vector.imag, # Uncompressed Target (Optional)
            "commit_loss": commit_loss,
            "codes": codes
        }

    def decode_from_codes(self, codes):
        """
        Decode from discrete codes.
        codes: [B, T, N_quantizers]
        """
        # Get quantized latents
        # IMPORTANT: The model was trained with Tanh output from encoder.
        # But here we are decoding from codes which are embeddings.
        # The embeddings were learned to match Tanh(z).
        # So the embeddings themselves already represent values in range [-1, 1].
        # We don't need to apply Tanh here.
        
        z_q = self.quantizer.from_codes(codes) # [B, 2*Latent, T]
        
        # Unflatten
        z_q_real, z_q_imag = torch.chunk(z_q, 2, dim=1)
        
        # Decode
        x_hat_real, x_hat_imag = self.decoder(z_q_real, z_q_imag) # [B, F, T]
        
        # Post-process
        real_hat = x_hat_real.permute(0, 2, 1) # [B, T, F]
        imag_hat = x_hat_imag.permute(0, 2, 1)
        
        # --- Inverse Power-Law Compression ---
        # Same as in forward()
        
        mag_sq_hat = real_hat**2 + imag_hat**2
        mag_hat_compressed = torch.sqrt(mag_sq_hat + 1e-5)
        mag_hat_compressed = torch.clamp(mag_hat_compressed, max=50.0)
        
        inv_alpha = 1.0 / 0.3
        scale_inv = torch.pow(mag_hat_compressed, inv_alpha - 1.0)
        
        real_hat_decomp = real_hat * scale_inv
        imag_hat_decomp = imag_hat * scale_inv
        
        # We can estimate frame_len from sr and frame_ms.
        frame_len = int(self.sr * self.frame_ms / 1000)
        
        # Use DECOMPRESSED spectrum
        # Ensure we pass the correct frame length for the given model config
        # If the input was padded, this frame_len might be slightly off if not careful,
        # but typically it's fixed by frame_ms.
        frames_hat = vector2frame(real_hat_decomp, imag_hat_decomp, self.sr, self.freqs, frame_len)
        
        return frames_hat
