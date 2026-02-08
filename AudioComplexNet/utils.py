import torch 
import math
from torchkbnufft import KbNufft,KbNufftAdjoint
def frame_audio(waveform: torch.Tensor, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 25.0):
    """
    将语音信号分帧 (无窗或后续可加窗)
    waveform: [num_samples] 或 [1, num_samples]
    sample_rate: 采样率 (Hz)
    frame_ms: 每帧长度（毫秒）
    hop_ms: 帧移（毫秒）
    返回: [num_frames, frame_length]
    """
    # if waveform.dim() == 2:
    #     waveform = waveform.squeeze(0)
    
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)
    
    # 使用 unfold 实现滑动窗口
    num_frames = 1 + (waveform.numel() - frame_length) // hop_length
    frames = waveform.unfold(-1, frame_length, hop_length)  # [num_frames, frame_length]
    
    return frames
def frame2vector(frames,sr,freqs):
    """
    对每一帧计算指定频率下的傅里叶变换结果
    frames: [num_frames, frame_len]
    freqs: list or tensor of real frequencies (Hz)
    return: [num_frames, len(freqs)] 复数结果
    """
    device = frames.device
    frame_len = frames.shape[-1]
    t = torch.arange(frame_len, device=device) / sr  # 时间轴
    
    # Apply Hanning Window to reduce spectral leakage
    # We apply window here because this is the Analysis step (Forward)
    window = torch.hann_window(frame_len, device=device)
    frames_windowed = frames * window.unsqueeze(0) # Broadcast to [N, L]
    
    # freqs = torch.tensor(freqs, device=device).float()
    # 计算复指数矩阵: exp(-j 2π f t)
    exp_matrix = torch.exp(-2j * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0))  # [num_freqs, frame_len]
    # 矩阵乘法实现DFT
    vector = torch.matmul(frames_windowed.to(torch.complex64), exp_matrix.T)  # [num_frames, num_freqs]
    return vector
def frame2vector_nufft(frames, sr, freqs):
    """
    使用 KbNufft 计算指定频率的傅里叶变换
    frames: [num_frames, frame_len] 实数或复数
    freqs: list or tensor of frequencies (Hz)
    return: [num_frames, num_freqs] 复数谱
    """
    device = frames.device
    frame_len = frames.shape[-1]
    frames = frames.to(torch.complex64)

    # 转为 tensor
    # freqs = torch.tensor(freqs, device=device, dtype=torch.float32)

    # 映射到 [-pi, pi] 弧度
    omega = 2 * torch.pi * freqs / sr
    # omega = (omega + torch.pi) % (2*torch.pi) - torch.pi
    # omega = omega.view(1, 1, -1)  # [1, num_freqs, 1]
    omega=omega.unsqueeze(0).unsqueeze(0)
    # 初始化 1D NUFFT
    nufft_ob = KbNufft(im_size=(frame_len,))

    # reshape frames: [num_frames, frame_len] -> [num_frames, 1, frame_len]
    # frames_input = frames.unsqueeze(-2)
    # print(frames_input.shape)
  
    
    spectrum = nufft_ob(frames, omega)  # [num_frames, 1, num_freqs]
    
    # spectrum = spectrum.squeeze(1)  # [num_frames, num_freqs]
    return spectrum

def overlap_add(frames, hop_length, window=None):
    """
    Overlap-Add with optional Synthesis Window (Weighted OLA)
    frames: [Batch, Num_Frames, Frame_Len] or [Num_Frames, Frame_Len]
    """
    if frames.dim() == 2:
        frames = frames.unsqueeze(0) # [1, N, L]
        
    batch_size, num_frames, frame_len = frames.shape
    device = frames.device
    
    total_len = (num_frames - 1) * hop_length + frame_len
    
    output = torch.zeros(batch_size, total_len, device=device)
    overlap_count = torch.zeros(total_len, device=device)
    
    if window is None:
        window = torch.ones(frame_len, device=device)
        
    # Apply Synthesis Window: x * w
    # If Analysis was x * w, then now we have x * w^2
    frames = frames * window.view(1, 1, -1)
    weight = window ** 2
    
    # Vectorized OLA? Hard with variable strides. Loop is fine for training if not too slow.
    # Actually, we can use F.fold but it's for constant stride. Here stride is constant.
    # But F.fold is 2D/3D. 
    # Let's stick to loop for clarity, it's fast enough on GPU for audio lengths.
    
    # But we need to handle Batch.
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_len
        output[:, start:end] += frames[:, i, :]
        overlap_count[start:end] += weight
        
    # Normalize
    mask = overlap_count > 1e-4
    output[:, mask] /= overlap_count[mask]
    
    return output

# def vector2frame(vector,sr,freqs,frame_len):
#     """
#     任意频率数组下的逆傅里叶变换。
#     输入:
#         spectrum: [B, num_frames, num_freqs] 复数张量
#         sr: 采样率
#         freqs: [num_freqs] 频率数组 (Hz)
#         frame_len: 每帧长度
#     输出:
#         reconstructed: [B, num_frames, frame_len] 实数张量
#     """
#     device = vector.device
#     num_freqs = vector.shape[-1]
#     t = torch.arange(frame_len, device=device).float() / sr  # [frame_len]


   
#     exp_matrix = torch.exp(2j * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0))  # [num_freqs, frame_len]

 
#     reconstructed = torch.matmul(vector, exp_matrix) / num_freqs  # [B, num_frames, frame_len]

#     # 取实部作为最终信号
#     reconstructed = reconstructed.real
#     return reconstructed
def vector2frame(vector_real, vector_imag, sr, freqs, frame_len):
    """
    任意频率数组下的逆傅里叶变换（实部/虚部分别输入版）

    输入:
        vector_real: [B, num_frames, num_freqs]  实部
        vector_imag: [B, num_frames, num_freqs]  虚部
        sr: 采样率
        freqs: [num_freqs] 频率数组 (Hz)
        frame_len: 每帧长度

    输出:
        reconstructed: [B, num_frames, frame_len] 实数张量
    """
    device = vector_real.device
    num_freqs = vector_real.shape[-1]

    # Ensure freqs is on the correct device
    if not isinstance(freqs, torch.Tensor):
        freqs = torch.tensor(freqs, device=device)
    elif freqs.device != device:
        freqs = freqs.to(device)

    # 时间轴 t
    t = torch.arange(frame_len, device=device).float() / sr   # [frame_len]

    '''
    # Original implementation using Transpose (Assuming Orthogonality)
    angle = 2 * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0)   # [num_freqs, frame_len]
    cos_term = torch.cos(angle)
    sin_term = torch.sin(angle)

    out_real = torch.matmul(vector_real, cos_term) - torch.matmul(vector_imag, sin_term)

    reconstructed = out_real / num_freqs
    return reconstructed
    '''

    # New implementation using Pseudo-Inverse (Least Squares) on Real System
    # Solves for Real X such that X @ A.T = V (complex)
    # Equivalent to minimizing || X @ Re(A).T - Re(V) ||^2 + || X @ Im(A).T - Im(V) ||^2
    
    exp_matrix = torch.exp(-2j * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0)) # [F, L]
    
    # Construct System Matrix M [2F, L]
    # We want to solve: X @ [Re(A).T, Im(A).T] = [Re(V), Im(V)]
    # Let M_T = [Re(A).T, Im(A).T] (shape L x 2F)
    # Then M = [Re(A); Im(A)] (shape 2F x L)
    # Solution: X = [Re(V), Im(V)] @ pinv(M_T) = [Re(V), Im(V)] @ pinv(M).T
    
    # Force FP32 for Matrix Construction and PINV to avoid underflow/overflow/singularity in BF16
    M = torch.cat([exp_matrix.real, exp_matrix.imag], dim=0).float() # [2F, L]
    
    # Compute pseudo-inverse
    # Use rcond to improve stability
    M_pinv = torch.linalg.pinv(M, rcond=1e-5) # [L, 2F]
    
    # Prepare target vector
    y = torch.cat([vector_real, vector_imag], dim=-1).float() # [B, N, 2F]
    
    # Reconstruct
    reconstructed = torch.matmul(y, M_pinv.T) # [B, N, L]
    
    # Cast back to original dtype if needed, or keep float32 for safety
    reconstructed = reconstructed.to(vector_real.dtype)
    
    return reconstructed
def vector2frame_nufft(vector,sr,freqs,frame_len):
    omega = 2 * torch.pi * freqs / sr
  
    omega = omega.view(1, 1, -1)  
    nufft_adj = KbNufftAdjoint(im_size=(frame_len,))
    reconstructed_frames = nufft_adj(vector, omega)
    return reconstructed_frames.real
if __name__=="__main__":
    x=torch.randn((2,10000))
    sr=10000
    frame=frame_audio(x,sr)
    print(frame.shape)
    freqs=torch.arange(20,8001,20)
    v=frame2vector(frame,sr,freqs)
    v_nufft=frame2vector_nufft(frame,sr,freqs)
    print(v_nufft.shape)
    print((v-v_nufft).abs().max())
    print(v.shape)
    recontruct=vector2frame(v,sr,freqs,frame.shape[-1])
    re_nufft=vector2frame_nufft(v_nufft,sr,freqs,frame.shape[-1])
    print(recontruct.shape)
    print(re_nufft.shape)
    diff=(frame-recontruct).abs()
    print(diff.max())
    print(diff.mean())
    print((frame-re_nufft).abs().max())