from collections import defaultdict
from pathlib import Path
import random
import sys
import pandas as pd
import pickle as pkl

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from const import presidents, block_size

with open("../tokenizer/pres_tokenizer.pkl", "rb") as f:
    pres_enc = pkl.load(f)

tokenizer = pres_enc
samples = defaultdict(list)

pres_df = pd.read_csv('./presidential_speeches.csv', sep=';', encoding='utf-8', quotechar="'")

# ignore everything in between an open and close square bracket
pres_df['speech'] = pres_df['speech'].str.replace(r"\[.*?\]", " ", regex=True)

# ignore everything in between an open and close parentheses
pres_df['speech'] = pres_df['speech'].str.replace(r"\(.*?\)", " ", regex=True)

# remove duplicate single quotes
pres_df['speech'] = pres_df['speech'].str.replace(r"'{2,}", "'", regex=True)

# remove duplicate dash
pres_df['speech'] = pres_df['speech'].str.replace(r"-{2,}", "-", regex=True)

# add space after every punctuation
pres_df['speech'] = pres_df['speech'].str.replace(r"([.,;:!?])", r"\1 ", regex=True)

# remove duplicate spaces
pres_df['speech'] = pres_df['speech'].str.replace(r'\s+', ' ', regex=True)

pres_df['speech_tokenized'] = pres_df['speech'].apply(lambda x: tokenizer.encode(x))

president_encodings = {}

max_len = block_size

for _, row in pres_df.iterrows():
    # Get the president's name
    president_name = row['President']
    
    if president_name not in presidents:
        continue
    
    # Retrieve or compute the encoding for the president
    if president_name not in president_encodings:
        president_encoding = tokenizer.encode(f'<President: {president_name}>', allowed_special='all')[0]
        president_encodings[president_name] = president_encoding
    else:
        president_encoding = president_encodings[president_name]
        
    speech_tokens = row['speech_tokenized']
    
    pres_tensors = samples[president_name]
    
    if len(speech_tokens) <= max_len:
        continue
    
    i = 0
    
    once = False
        
    while i < len(speech_tokens):
        input = [president_encoding, *speech_tokens[i:min(i+max_len-1, len(speech_tokens))]]
        output = speech_tokens[i:min(i+max_len, len(speech_tokens))]
        
        pres_tensors.append([torch.tensor(input), torch.tensor(output)])
        
        i += (max_len-1)
        
        assert len(input) == len(output) and len(input) == max_len
                
        if len(speech_tokens) - i < max_len:
            if once:
                break
            
            i -= (max_len - (len(speech_tokens) - i))
            once = True
            
        
train = []
validation = []

for president, tensor_list in samples.items():
    random.shuffle(tensor_list)
    train.extend(tensor_list[:int(0.8*len(tensor_list))])
    validation.extend(tensor_list[int(0.8*len(tensor_list)):])
    
print('train len', len(train))
print('val len', len(validation))
    
with open('train.pkl', 'wb') as f:
    pkl.dump(train, f)
    
with open('validation.pkl', 'wb') as f:
    pkl.dump(validation, f)