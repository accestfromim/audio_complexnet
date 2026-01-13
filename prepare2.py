from datasets import Audio
from transformers import AutoProcessor
from datasets import load_dataset

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

dataset= load_dataset("/root/working/librispeech_data", "all")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

model_checkpoint = "facebook/hubert-base-ls960"
processor = AutoProcessor.from_pretrained(model_checkpoint)

def prepare_dataset(batch):
    """
    预处理单样本函数：将音频数组转换为 input_values
    """
    audio = batch["audio"]
    
    # 校验采样率
    if audio["sampling_rate"]!= 16000:
        # 在这里可以抛出异常，或者调用 librosa.resample
        # 推荐依赖 datasets 的 cast_column 机制
        pass

    # 调用 Processor 进行特征提取
    # input_values 是 Wav2Vec2/HuBERT 期望的输入键名
    # sampling_rate 参数是必须的，用于处理器内部校验
    inputs = processor(
        audio["array"], 
        sampling_rate=16000,
        return_tensors="pt" # 直接请求 PyTorch 张量
    )
    
    # 提取张量并移除 Batch 维度
    # processor 默认返回 (1, T) 的形状，但 datasets.map 期望的是 (T,)
    # 后续的 DataCollator 会重新组装 Batch
    batch["input_values"] = inputs.input_values
    
    # 如果进行有监督训练，还需要处理文本
    # batch["labels"] = processor(text=batch["text"]).input_ids
    
    # 计算长度，用于后续的分组排序（Group By Length）优化
    batch["input_length"] = len(batch["input_values"])
    
    return batch

encoded_dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names, # 移除原始的 audio, file, text 列
    num_proc=4, # 启用多进程，根据 CPU 核心数调整
    batched=False # 这里演示逐个处理；若改为批量处理需调整函数逻辑
)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict:
        # features 是一个列表，包含 Batch Size 个样本
        # 每个样本是 {'input_values': tensor(...), 'labels':...}

        # 分离音频输入和标签
        # input_values 是 float32，需要 padding_value=0.0 (通常)
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        
        # 只有在存在标签时才处理标签
        label_features = [{"input_ids": feature["labels"]} for feature in features] if "labels" in features else None

        # 使用 Processor 进行动态填充
        # processor.pad 会自动调用 feature_extractor.pad
        # 这会自动处理 input_values 的填充，使其长度一致
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # 处理标签 (如果是 ASR 任务)
        if label_features:
            with self.processor.as_target_processor(): # 兼容旧版 API
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )
            
            # 关键技巧：将 Padding 部分的 Label 设为 -100
            # CTC Loss 会自动忽略值为 -100 的目标
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels

        return batch