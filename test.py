# import torch

# from torchkbnufft import KbNufft, KbNufftAdjoint
# import math

# class AudioNUFFTProcessor:
#     def __init__(self, sr, frame_ms=25.0, hop_ms=None):
#         """
#         sr: 采样率
#         frame_ms: 帧长（毫秒）
#         hop_ms: 帧移（毫秒），默认 frame_ms/2
#         device: 'cuda' 或 'cpu'
#         """
#         self.sr = sr
#         self.frame_ms = frame_ms
#         self.hop_ms = hop_ms if hop_ms is not None else frame_ms / 2


#     def frame_audio(self, waveforms):
#         """
#         waveforms: [B, T] 实数波形
#         return: [B, num_frames, frame_len]
#         """
#         B, T = waveforms.shape
#         frame_len = int(self.sr * self.frame_ms / 1000)
#         hop_len = int(self.sr * self.hop_ms / 1000)
#         num_frames = 1 + (T - frame_len) // hop_len
#         frames = waveforms.unfold(-1, frame_len, hop_len)  # [B, num_frames, frame_len]
#         return frames

#     def frames2spectrum(self, frames, freqs):
#         """
#         frames: [B, num_frames, frame_len]
#         freqs: [num_freqs] Hz
#         return: [B, num_frames, num_freqs] 复数谱
#         """
#         B, num_frames, frame_len = frames.shape
#         frames = frames.to(torch.complex64)
#         # freqs = torch.tensor(freqs, dtype=torch.float32)
#         omega = 2 * math.pi * freqs / self.sr  # 弧度
#         omega = omega.view(1, 1, -1)  # [1, num_freqs, 1]
#         print(f"omega:{omega.shape}")
#         nufft = KbNufft(im_size=(frame_len,))
#         frames_input = frames.reshape(B * num_frames, 1, frame_len)  # [B*num_frames,1,frame_len]
#         print(f"frames_input:{frames_input.shape}")
#         spectrum = nufft(frames_input, omega)  # [B*num_frames,1,num_freqs]
#         spectrum = spectrum.reshape(B, num_frames, -1)
#         return spectrum

#     def spectrum2frames(self, spectrum, freqs, frame_len):
#         """
#         spectrum: [B, num_frames, num_freqs] 复数谱
#         freqs: [num_freqs] Hz
#         frame_len: 每帧长度
#         return: [B, num_frames, frame_len] 复数时域帧
#         """
#         B, num_frames, num_freqs = spectrum.shape
#         # freqs = torch.tensor(freqs, dtype=torch.float32)
#         omega = 2 * math.pi * freqs / self.sr  # 弧度
#         omega = omega.view(1, -1,1)

#         nufft_adj = KbNufftAdjoint(im_size=(frame_len,))
#         spectrum_input = spectrum.reshape(B * num_frames, 1, num_freqs)
#         reconstructed_frames = nufft_adj(spectrum_input, omega).reshape(B, num_frames, frame_len)
#         return reconstructed_frames.real  # 返回实部

# # ================= 示例使用 =================
# if __name__ == "__main__":
#     waveform=torch.randn(10000)
#     sr=10000
#     B = 4
#     waveforms = waveform.repeat(B,1)  # 模拟 batch
#     processor = AudioNUFFTProcessor(sr, frame_ms=25, hop_ms=25)

#     # 分帧
#     frames = processor.frame_audio(waveforms)  # [B, num_frames, frame_len]
#     print("Frames:", frames.shape)

#     # 指定频率
#     freqs = torch.arange(20, 8001, 20)  # 20Hz~8000Hz

#     # 正变换
#     spectrum = processor.frames2spectrum(frames, freqs)
#     print("Spectrum:", spectrum.shape)

#     # 逆变换回帧
#     reconstructed_frames = processor.spectrum2frames(spectrum, freqs, frames.shape[-1])
#     print("Reconstructed frames:", reconstructed_frames.shape)
#     print((reconstructed_frames-frames).abs().max())
import torch
import torchkbnufft as tkbn

# 1. 准备音频信号
audio = torch.randn(320)  # 模拟一个随机音频信号，长度 320
sr = 20000  # 采样频率，例如 16kHz

# 将音频信号转换为复数格式，并调整形状为 (batch_size, num_coils, length)
audio_complex = audio.to(torch.complex64)
audio_complex = audio_complex.unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, 320]

# 2. 准备目标频率坐标
target_freqs = torch.arange(20,8001,20)
# target_freqs=torch.tensor([20,30])
# 将物理频率归一化到 [-π, π] 区间
normalized_freqs = (2 * torch.pi * target_freqs) / sr - torch.pi 
# 调整坐标张量的形状为 (batch_size, ndims, n_samples)
coord = normalized_freqs.unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, 5]

# 3. 初始化 NUFFT 对象
nufft_ob = tkbn.KbNufft(im_size=(audio_complex.shape[-1],))

# 4. 执行非均匀傅里叶变换
spectrum_at_target_freqs = nufft_ob(audio_complex, coord)
print("频谱形状:", spectrum_at_target_freqs.shape)

# 5. 初始化逆变换对象
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=(audio_complex.shape[-1],))

# 6. 执行逆变换
reconstructed_signal = adjnufft_ob(spectrum_at_target_freqs, coord)

# 7. 处理逆变换结果
audio_reconstructed = reconstructed_signal.real.squeeze()  # 取实部并压缩维度
print("重构音频信号形状:", audio_reconstructed.shape)
print(f"diff:{(audio-audio_reconstructed).abs().mean()}")