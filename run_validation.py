import torch
from pres_gpt2 import PresGPT2, GPTConfig 
from Dataset import PresidentDataset

import pickle
import tiktoken
from torch.utils.data import DataLoader

from const import block_size

with open("./tokenizer/pres_tokenizer.pkl", "rb") as f:
    pres_enc: tiktoken.Encoding = pickle.load(f)
    
with open("./data/validation.pkl", "rb") as f:
    validation = pickle.load(f)

validation_dataset = PresidentDataset(validation)

config: GPTConfig = GPTConfig(
    block_size,
    len(pres_enc._mergeable_ranks) + len(pres_enc._special_tokens),
    12,
    12,
    768,
    0.0
)

model: PresGPT2 = PresGPT2(config)

checkpoint = torch.load('./model/checkpoint.pt', map_location=torch.device("cuda"))  # Adjust device if needed
model.load_state_dict(checkpoint)

validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=True)

model.eval()
    
val_loss = 0

with torch.no_grad():
    for X, Y in validation_dataloader:
        _, loss = model(X, Y)
        
        val_loss += loss.item()

print(val_loss.item() / len(validation))