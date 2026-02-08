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
    # freqs = torch.tensor(freqs, device=device).float()
    # 计算复指数矩阵: exp(-j 2π f t)
    exp_matrix = torch.exp(-2j * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0))  # [num_freqs, frame_len]
    # 矩阵乘法实现DFT
    vector = torch.matmul(frames.to(torch.complex64), exp_matrix.T)  # [num_frames, num_freqs]
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

def vector2frame_(vector,sr,freqs,frame_len):
    """
    任意频率数组下的逆傅里叶变换。
    输入:
        spectrum: [B, num_frames, num_freqs] 复数张量
        sr: 采样率
        freqs: [num_freqs] 频率数组 (Hz)
        frame_len: 每帧长度
    输出:
        reconstructed: [B, num_frames, frame_len] 实数张量
    """
    device = vector.device
    num_freqs = vector.shape[-1]
    t = torch.arange(frame_len, device=device).float() / sr  # [frame_len]


   
    exp_matrix = torch.exp(2j * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0))  # [num_freqs, frame_len]

 
    reconstructed = torch.matmul(vector, exp_matrix) / num_freqs  # [B, num_frames, frame_len]

    # 取实部作为最终信号
    reconstructed = reconstructed.real
    return reconstructed
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

    # 时间轴 t
    t = torch.arange(frame_len, device=device).float() / sr   # [frame_len]

    # 构造 e^{j 2π f t} = cos(2πft) + j sin(2πft)
    angle = 2 * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0)   # [num_freqs, frame_len]
    cos_term = torch.cos(angle)
    sin_term = torch.sin(angle)

    # ====== 实数域复数乘法 ======
    # vector * exp(jθ) = (a + jb)(cosθ + j sinθ)
    # real = a*cosθ - b*sinθ
    # imag = a*sinθ + b*cosθ

    out_real = torch.matmul(vector_real, cos_term) - torch.matmul(vector_imag, sin_term)
    out_imag = torch.matmul(vector_real, sin_term) + torch.matmul(vector_imag, cos_term)

    # 只取实部，并做归一化
    reconstructed = out_real / num_freqs
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
    recontruct=vector2frame(v.real,v.imag,sr,freqs,frame.shape[-1])
    r=vector2frame_(v,sr,freqs,frame.shape[-1])
    re_nufft=vector2frame_nufft(v_nufft,sr,freqs,frame.shape[-1])
    print(recontruct.shape)
    print(re_nufft.shape)
    diff=(frame-recontruct).abs()
    print(diff.max())
    print(diff.mean())
    print((frame-re_nufft).abs().max())
    print((recontruct-r).abs().max())

def overlap_add(frames: torch.Tensor, hop_length: int) -> torch.Tensor:
    """
    Overlap-Add method to reconstruct waveform from frames.
    frames: [Batch, Num_Frames, Frame_Len] or [Num_Frames, Frame_Len]
    hop_length: int
    """
    if frames.dim() == 2:
        frames = frames.unsqueeze(0)  # [1, Num_Frames, Frame_Len]
        
    batch_size, num_frames, frame_length = frames.shape
    
    # Calculate output length
    total_length = (num_frames - 1) * hop_length + frame_length
    
    output_wav = torch.zeros((batch_size, total_length), device=frames.device)
    norm_wav = torch.zeros((batch_size, total_length), device=frames.device)
    
    window = torch.ones(frame_length, device=frames.device) # Rectangular window implicitly used in framing
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        output_wav[:, start:end] += frames[:, i, :]
        norm_wav[:, start:end] += window
        
    # Avoid division by zero
    mask = norm_wav > 1e-5
    output_wav[mask] /= norm_wav[mask]
    
    return output_wav