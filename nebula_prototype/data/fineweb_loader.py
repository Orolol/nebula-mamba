import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class FineWebStreamingDataset(IterableDataset):
    def __init__(self, split="train", seq_len=1024):
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", split=split, streaming=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len

    def __iter__(self):
        iterator = iter(self.dataset)
        for item in iterator:
            text = item["text"]
            encodings = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.seq_len,
                return_tensors="pt"
            )
            # Yield single example: (input_ids, attention_mask)
            # Squeeze to remove batch dim added by tokenizer
            yield encodings.input_ids.squeeze(0), encodings.attention_mask.squeeze(0)

def get_dataloader(split="train", seq_len=1024, batch_size=4, num_workers=4):
    dataset = FineWebStreamingDataset(split=split, seq_len=seq_len)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
