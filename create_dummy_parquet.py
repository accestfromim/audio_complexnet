import torch
import numpy as np
import soundfile as sf
import os
import pandas as pd
from datasets import Dataset, Audio

def create_dummy_parquet():
    print("Generating dummy audio files...")
    # Create a temporary directory for dummy wavs
    os.makedirs("dummy_data", exist_ok=True)
    
    audio_paths = []
    sr = 16000
    duration = 2.0 # seconds
    
    # Generate 10 dummy audio files
    for i in range(10):
        # Generate random noise or sine wave
        t = np.linspace(0, duration, int(sr * duration))
        # Simple sine wave + noise
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.normal(0, 1, len(t))
        
        file_path = os.path.abspath(f"dummy_data/sample_{i}.wav")
        sf.write(file_path, audio, sr)
        audio_paths.append(file_path)
        
    print("Creating HuggingFace Dataset...")
    # Create a dictionary with 'audio' column pointing to files
    data = {"audio": audio_paths}
    
    # Create dataset
    ds = Dataset.from_dict(data)
    
    # Skip casting to Audio to avoid datasets internal decoding issues (missing torchcodec/ffmpeg)
    # We will load audio manually in prepare.py
    # ds = ds.cast_column("audio", Audio(sampling_rate=sr))
    
    output_file = os.path.abspath("dummy_train.parquet")
    print(f"Saving to {output_file}...")
    
    # Saving to parquet. embed_external_files=True will store the audio data inside the parquet
    ds.to_parquet(output_file)
    
    print("Done! Dummy parquet file created.")
    return output_file

if __name__ == "__main__":
    create_dummy_parquet()
