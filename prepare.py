import torch
import os
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Union, Any
from datasets import load_dataset, Audio, DatasetDict

SAMPLE_RATE = 16000
TARGET_FREQS = torch.linspace(0, SAMPLE_RATE/2, 256) 

def batch_frame_audio(waveforms, sr, frame_ms=25.0, hop_ms=10.0):
    """
    waveforms: [Batch, Time] (已经 Pad 过的波形)
    """
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    
    # unfold 会在最后一个维度操作，自动输出 [Batch, Num_Frames, Frame_Len]
    # 注意：如果 waveform 只有一维，unfold 输出 [Num_Frames, Frame_Len]
    # 这里我们保证输入是二维 [B, T]
    frames = waveforms.unfold(-1, frame_length, hop_length)
    return frames

def batch_frame2vector(frames, sr, freqs):
    """
    frames: [Batch, Num_Frames, Frame_Len]
    freqs: [Num_Freqs]
    """
    device = frames.device
    frame_len = frames.shape[-1]
    
    # 生成时间轴 [Frame_Len]
    t = torch.arange(frame_len, device=device).float() / sr 
    
    # 生成复指数矩阵 [Num_Freqs, Frame_Len]
    # exp(-j 2pi f t)
    exp_matrix = torch.exp(-2j * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0))
    
    # 矩阵乘法:
    # [B, N, L] (complex) @ [L, F] (complex) -> [B, N, F]
    # PyTorch matmul 支持这种广播
    vector = torch.matmul(frames.to(torch.complex64), exp_matrix.T)
    return vector


@dataclass
class CustomAudioDataCollator:
    sr: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    freqs: torch.Tensor = None  
    return_complex: bool = False # False=返回实虚拼接, True=返回复数Tensor
    max_frames: int = 512

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        features: List of dataset items. 
                  Example item: {'input_values': tensor([...]), 'label': ...}
        """
        
        # 1. 提取波形 (List of Tensors)
        # 假设 dataset map 之后字段叫 'input_values'
        waveforms = [f["input_values"] for f in features]
        
        # 记录原始波形长度，用于计算 mask
        original_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        
        # 2. Pad 波形 -> [Batch, Max_Time]
        # batch_first=True 使得输出为 [B, T]
        padded_waves = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        
        # 3. 批量分帧 -> [Batch, Num_Frames, Frame_Len]
        frames = batch_frame_audio(padded_waves, self.sr, self.frame_ms, self.hop_ms)

        # 可选：截断过长序列，控制显存开销
        if self.max_frames is not None and frames.shape[1] > self.max_frames:
            frames = frames[:, : self.max_frames]
        
        # 4. 批量自定义 DFT (已移入模型内部，此处直接返回 Frames)
        # 确保 freqs 在正确的 device 上 (通常 collator 在 CPU 上运行)
        # if self.freqs.device != frames.device:
        #     self.freqs = self.freqs.to(frames.device)
            
        # spectrums = batch_frame2vector(frames, self.sr, self.freqs)
        
        # 5. 处理输入输出
        # 修改为：inputs_features 直接为 frames (Float Tensor)
        inputs = frames

        # 6. 生成 Attention Mask
        # 因为波形被 Pad 过，频谱后面几帧也是无效的
        # 计算每一条音频对应的有效帧数
        frame_len = int(self.sr * self.frame_ms / 1000)
        hop_len = int(self.sr * self.hop_ms / 1000)
        
        # 公式: num_frames = 1 + (len - frame_len) // hop_len
        valid_frames = 1 + (original_lengths - frame_len) // hop_len
        valid_frames = torch.clamp(valid_frames, min=1) # 至少保留1帧
        
        max_frames = inputs.shape[1]
        attention_mask = torch.zeros((len(features), max_frames), dtype=torch.long)
        
        for i, valid_len in enumerate(valid_frames.clamp(max=max_frames)):
            attention_mask[i, :valid_len] = 1
            
        batch = {
            "inputs_features": inputs.float(),
            "attention_mask": attention_mask,
            "target_frames": frames.float(),
        }
        
        if "labels" in features[0]:
             labels = [f["labels"] for f in features]
             batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
             
        return batch
    
NUM_FREQS = 257 # Standard STFT size for n_fft=512 (Nyquist at 8kHz)
# 这里可以是 Mel 频率，或者是线性频率
custom_freqs = torch.linspace(0, 8000, NUM_FREQS) 

# 模式选择：
# "dummy_file": 加载单个 dummy_train.parquet
# "dummy_folder": 加载生成的 dummy_sharded_dataset 文件夹（模拟分片数据集）
# "hf_librispeech": 加载 Hugging Face 的 LibriSpeech (需要网络)
# "custom": 加载你自己指定的数据集目录
DATASET_MODE = "hf_librispeech" 

# 你的真实数据集路径配置
CUSTOM_DATASET_PATH = "/path/to/your/dataset/folder" 

try:
    if DATASET_MODE == "dummy_file":
        data_files = {"train": "/mnt/d/Desktop/audio_working/dummy_train.parquet"} 
        dataset = load_dataset("parquet", data_files=data_files, split="train")
        
    elif DATASET_MODE == "dummy_folder":
        # 加载本地分片数据集
        base_path = "/mnt/d/Desktop/audio_working/dummy_sharded_dataset"
        train_path = os.path.join(base_path, "train")
        val_path = os.path.join(base_path, "validation")
        
        data_dict = {}
        
        if os.path.exists(train_path):
            print(f"Loading train set from {train_path}...")
            data_dict["train"] = load_dataset("parquet", data_dir=train_path, split="train")
            
        if os.path.exists(val_path):
            print(f"Loading validation set from {val_path}...")
            # split="train" 是因为 load_dataset 对单文件夹默认行为，我们将其放入 validation 键
            data_dict["validation"] = load_dataset("parquet", data_dir=val_path, split="train")
            
        if len(data_dict) == 0:
            raise FileNotFoundError(f"No dataset found in {base_path}")
            
        dataset = DatasetDict(data_dict) if len(data_dict) > 1 else list(data_dict.values())[0]

    elif DATASET_MODE == "hf_librispeech":
        # 加载官方 LibriSpeech：同时构造 train 和 validation
        dataset = DatasetDict(
            {
                "train": load_dataset(
                    "openslr/librispeech_asr", "clean", split="train.100"
                ),
                "validation": load_dataset(
                    "openslr/librispeech_asr", "clean", split="validation"
                ),
            }
        )
        # 显式关闭内部解码，避免依赖 torchcodec，仅保留路径/bytes 供自定义解码使用
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))
        
    elif DATASET_MODE == "custom":
        # 加载自定义目录
        # 如果是 parquet 文件夹
        dataset = load_dataset("parquet", data_dir=CUSTOM_DATASET_PATH, split="train")
        # 如果是 Hugging Face Hub 上的数据集
        # dataset = load_dataset("your_org/your_dataset", split="train")

    else:
        raise ValueError(f"Unknown mode: {DATASET_MODE}")

    print(f"成功加载数据集 (Mode: {DATASET_MODE})，样本数: {len(dataset)}")

except Exception as e:
    print(f"加载数据集失败 ({e})，尝试回退到默认 LibriSpeech 数据集用于演示...")
    dataset = DatasetDict(
        {
            "train": load_dataset(
                "openslr/librispeech_asr", "clean", split="train.100"
            ),
            "validation": load_dataset(
                "openslr/librispeech_asr", "clean", split="validation"
            ),
        }
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))


# 移除 cast_column，避免 datasets 内部解码依赖问题 (如 torchcodec/ffmpeg)
# if "audio" in dataset.column_names:
#     dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

import io
import soundfile as sf

def preprocess_function(example):
    # 提取数组，转为 Tensor
    # 兼容多种情况：
    # 1. 原始路径字符串 (手动加载)
    # 2. 已解码的字典 (LibriSpeech fallback)
    # 3. 二进制数据 (Parquet bytes)
    audio_data = example["audio"]
    
    wav = None
    sr = None
    
    try:
        # 情况1: 字典 (可能是已解码的array，或者是包含bytes的字典)
        if isinstance(audio_data, dict):
            if "array" in audio_data:
                # datasets 已解码
                return {"input_values": torch.tensor(audio_data["array"], dtype=torch.float32)}
            elif "bytes" in audio_data:
                # 二进制数据
                audio_bytes = audio_data["bytes"]
                if audio_bytes is not None:
                    # 使用 soundfile 读取 bytes
                    wav_numpy, sr = sf.read(io.BytesIO(audio_bytes))
                    wav = torch.tensor(wav_numpy).float()
                elif "path" in audio_data and audio_data["path"] is not None:
                    # 只有path
                    wav_numpy, sr = sf.read(audio_data["path"])
                    wav = torch.tensor(wav_numpy).float()
        
        # 情况2: 字符串 (文件路径)
        elif isinstance(audio_data, str):
            wav_numpy, sr = sf.read(audio_data)
            wav = torch.tensor(wav_numpy).float()
            
        # 情况3: 直接是 bytes
        elif isinstance(audio_data, bytes):
            wav_numpy, sr = sf.read(io.BytesIO(audio_data))
            wav = torch.tensor(wav_numpy).float()

        if wav is not None:
            # soundfile load: [Time, Channels] or [Time]
            # PyTorch expects: [Channels, Time] or [Time]
            # Handle shape
            if wav.ndim > 1:
                # Transpose to [Channels, Time]
                wav = wav.transpose(0, 1)
                # Mix to mono if needed
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0)
                else:
                    wav = wav.squeeze(0)
            
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = resampler(wav)
                
            return {"input_values": wav}
            
    except Exception as e:
        print(f"加载音频失败: {audio_data}, error: {e}")

    # Fallback
    return {"input_values": torch.zeros(16000)}

# 移除不必要的列，只保留 input_values
# 动态获取列名以兼容不同数据集
if isinstance(dataset, DatasetDict):
    column_names = dataset[list(dataset.keys())[0]].column_names
else:
    column_names = dataset.column_names

cols_to_remove = [col for col in column_names if col != "input_values"]
'''
dataset = dataset.map(
    preprocess_function,
    remove_columns=cols_to_remove,
    load_from_cache_file=False,
    num_proc=4,
)
'''
dataset = dataset.map(preprocess_function, remove_columns=cols_to_remove, load_from_cache_file=True)
# 确保输出为 PyTorch Tensor
dataset.set_format(type="torch", columns=["input_values"])

collator = CustomAudioDataCollator(
    sr=16000,
    frame_ms=25.0,
    hop_ms=10.0,
    freqs=custom_freqs,
    return_complex=False # 大模型通常选 False，输入实数向量
)
