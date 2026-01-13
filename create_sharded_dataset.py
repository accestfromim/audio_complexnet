import os
import torch
import numpy as np
import soundfile as sf
import pandas as pd
from datasets import Dataset, Audio
import shutil

def create_sharded_dataset():
    # Base directory
    base_dir = os.path.abspath("dummy_sharded_dataset")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Audio storage (shared)
    audio_dir = os.path.join(base_dir, "audio_files")
    os.makedirs(audio_dir, exist_ok=True)

    print(f"Generating dummy dataset in {base_dir}...")
    
    splits = {
        "train": 20,  # 20 samples for training
        "validation": 5 # 5 samples for validation
    }
    
    sr = 16000
    duration = 1.0 # seconds
    
    for split_name, num_samples in splits.items():
        print(f"Processing split: {split_name}...")
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Generate samples
        data_list = []
        for i in range(num_samples):
            # Generate random audio
            t = np.linspace(0, duration, int(sr * duration))
            # Different freq for each sample to make them distinct
            freq = 220 + (i * 10) 
            audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.normal(0, 1, len(t))
            
            file_name = f"{split_name}_{i}.wav"
            file_path = os.path.join(audio_dir, file_name)
            sf.write(file_path, audio, sr)
            
            # Mimic Librispeech columns
            data_list.append({
                "file": file_path, # librispeech often has 'file' or 'audio'
                "audio": file_path, 
                "text": f"THIS IS SAMPLE {i} IN SPLIT {split_name.upper()}",
                "speaker_id": i % 5,
                "chapter_id": i % 2,
                "id": f"{split_name}_{i}"
            })
            
        # Sharding: Save into multiple parquet files if samples > 10
        # For this dummy set, we force splitting to demonstrate multi-file loading
        df = pd.DataFrame(data_list)
        
        shard_size = 10
        num_shards = (len(df) + shard_size - 1) // shard_size
        
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, len(df))
            
            sub_df = df.iloc[start_idx:end_idx]
            
            # Convert to HF Dataset to save as parquet easily
            # We DON'T cast to Audio() here to keep paths as strings, 
            # simulating the case where we load audio manually in prepare.py
            # or simulating a "path-only" dataset which is common for large local datasets.
            ds = Dataset.from_pandas(sub_df)
            
            output_file = os.path.join(split_dir, f"shard_{shard_idx}.parquet")
            ds.to_parquet(output_file)
            print(f"  Saved {output_file} ({len(sub_df)} samples)")

    print("Done! Sharded dataset created.")
    print(f"Structure:")
    print(f"  {base_dir}/")
    print(f"    train/ (contains .parquet shards)")
    print(f"    validation/ (contains .parquet shards)")
    print(f"    audio_files/ (contains .wav files)")

if __name__ == "__main__":
    create_sharded_dataset()
