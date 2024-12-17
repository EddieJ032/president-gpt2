import tiktoken
import pickle as pkl

from const import presidents

gpt2_base = tiktoken.get_encoding("gpt2")

special_tokens = dict()

for i, pres in enumerate(presidents):
    special_tokens[f'<President: {pres}>'] = len(gpt2_base._mergeable_ranks)+len(gpt2_base._special_tokens)+i
    
pres_enc = tiktoken.Encoding(
    name="pres_encoding",
    pat_str=gpt2_base._pat_str,
    mergeable_ranks=gpt2_base._mergeable_ranks,
    special_tokens={**gpt2_base._special_tokens, **special_tokens},
    explicit_n_vocab=50277
)

# pickle the tokenizer obj
with open('pres_tokenizer.pkl', 'wb') as f:
    pkl.dump(pres_enc, f)