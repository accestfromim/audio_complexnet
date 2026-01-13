from datasets import load_dataset
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("/root/working/librispeech_data", "all")

print(ds["train.clean.100"][0])