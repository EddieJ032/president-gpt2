import torch
import torch.nn as nn
from torch.nn import functional as F

# parameters for GPT-2 
class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layers, n_heads, n_embd):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
    
""" one head of self-attention """
class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        head_size = config.n_embd // config.n_heads
        
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))  
        
    def forward(self, x):
        _,T,_ = x.shape

        k = self.key(x)
        q = self.query(x)
        
        # transpose last two dimensions
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)

        # mask out bottom half
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax each column
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)

        return wei @ v
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd,config. n_embd)
        self.proj.STD_SCALE_INIT = 1
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.STD_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
class PresGPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # stop immediately if not a multiple
        assert config.n_embd % config.n_heads == 0
        
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight tying intuition: if inputs are encoded similar, outputs should be similar
        self.lm_head.weight = self.transformer.wte.weight
        
        self.apply(self.__init_weights)
        
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            
            if hasattr(module, 'STD_SCALE_INIT'):
                # 2 times since each Block has two residual sums
                std *= (2*self.config.n_layers)**-0.5
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, X, y=None):
        B,T = X.shape
        
        tok_emb = self.transformer.wte(X) # -> (B, T, config.n_embd)
        
        pos = torch.arange(0, T, 1, dtype=torch.long, device=X.device) # put it on the same device as X
        
        pos_emb = self.transformer.wpe(pos)
        
        x = pos_emb + tok_emb
        
        for b in self.transformer.h:
            x = b(x)
            
        x = self.transformer.ln_f(x)
        
        loss = None
        logits = self.lm_head(x) # B x T x vocab_size
        
        if y is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
        return logits, loss
