import torch
from pres_gpt2 import PresGPT2, GPTConfig 
from Dataset import PresidentDataset

import pickle
import tiktoken
import tiktoken
from torch.utils.data import DataLoader


with open("./tokenizer/pres_tokenizer.pkl", "rb") as f:
    pres_enc: tiktoken.Encoding = pickle.load(f)
    
with open("./data/train.pkl", "rb") as f:
    train = pickle.load(f)
    
with open("./data/validation.pkl", "rb") as f:
    validation = pickle.load(f)

train_dataset = PresidentDataset(train)
validation_dataset = PresidentDataset(validation)

config: GPTConfig = GPTConfig(
    1024,
    len(pres_enc._mergeable_ranks) + len(pres_enc._special_tokens),
    12,
    12,
    768
)

model: PresGPT2 = PresGPT2(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

train_input, train_output = next(iter(train_dataloader))

for i in range(5):
    optimizer.zero_grad()
    logits, loss = model(train_input, train_output)
    loss.backward()
    optimizer.step()
    print(f'{i} loss: {loss.item()}')