from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def causal_mask(size):
  mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
  return mask == 0

class BilingualDataset(Dataset):
  def __init__(self, ds, seq_len)->None:
    super().__init__()

    self.ds = ds
    self.seq_len = seq_len

    
  def __len__(self):
    return len(self.ds)

  def __getitem__(self, index: Any)->Any:
    
    input_ids = self.ds[index]["input_ids"]
    inputs = {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "labels": torch.tensor(input_ids.copy(), dtype = torch.int64),
        "mask": torch.tensor(causal_mask(len(input_ids)), dtype = torch.int64),
    }

    

    assert inputs["input_ids"].size(0) == self.seq_len
    assert inputs["labels"].size(0) == self.seq_len
    assert inputs["mask"].size(-1) == self.seq_len
    
    
    return inputs
