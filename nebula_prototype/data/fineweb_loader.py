import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class FineWebStreamingDataset(IterableDataset):
    def __init__(self, split="train", seq_len=1024, batch_size=4):
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", split=split, streaming=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            batch_texts = []
            try:
                for _ in range(self.batch_size):
                    batch_texts.append(next(iterator)["text"])
            except StopIteration:
                if not batch_texts:
                    break
            
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.seq_len,
                return_tensors="pt"
            )
            
            yield encodings.input_ids, encodings.attention_mask

def get_dataloader(split="train", seq_len=1024, batch_size=4):
    dataset = FineWebStreamingDataset(split=split, seq_len=seq_len, batch_size=batch_size)
    return DataLoader(dataset, batch_size=None) # Batching is handled in dataset
