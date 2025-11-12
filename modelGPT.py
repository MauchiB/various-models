import torch
import torch.nn as nn
import math
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, BitsAndBytesConfig


class ModelConfig:
    def __init__(self,
                 vocab_size: int,
                 n_embd: int = 512,
                 n_head: int = 4,
                 dropout: float = 0.1,
                 n_positions: int = 100,
                 n_layer: int = 3,
                 k: int = 5):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.k = k
        # convenience alias used elsewhere in the code
        self.n_dim = n_embd


class BaseClase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        assert self.cfg.n_embd % self.cfg.n_head == 0, 'dim % head == 0'
        
        self.cfg.head_dim = config.n_embd // config.n_head


class Attention(BaseClase):
    def __init__(self, config):
        super().__init__(config)

        self.c_attn = nn.Linear(self.cfg.n_embd, self.cfg.n_embd * 3)
        self.c_proj = nn.Linear(self.cfg.n_embd, self.cfg.n_embd)

        self.register_buffer('bias', torch.tril(torch.ones(self.cfg.n_positions,
                             self.cfg.n_positions))
                             .view(1, 1, self.cfg.n_positions, self.cfg.n_positions))

    def forward(self, x):
        B, S, D = x.size()

        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.cfg.n_embd, dim=2)

        q = q.view(B, S,self.cfg.n_head,self.head_dim).transpose(1, 2)
        k = k.view(B, S,self.cfg.n_head,self.head_dim).transpose(1, 2)
        v = v.view(B, S,self.cfg.n_head,self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.cfg.head_dim))

        attn = attn.masked_fill(self.bias[:, :, :S, :S] == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.c_proj(out)

        return out
    


class MLP(BaseClase):
    def __init__(self, config):
        super().__init__(config)

        self.c_fc = nn.Linear(self.cfg.n_embd, self.cfg.n_embd * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.cfg.n_embd * 4, self.cfg.n_embd)
        

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
    


class GPTblock(BaseClase):
    def __init__(self, config):
        super().__init__(config)

        self.ln_1 = nn.LayerNorm(self.cfg.n_embd)
        self.ln_2 = nn.LayerNorm(self.cfg.n_embd)

        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(BaseClase):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd),
            'wpe': nn.Embedding(self.cfg.n_positions, self.cfg.n_embd),
            'h': nn.ModuleList([GPTblock(config) for _ in range(self.cfg.n_layer)]),
            'ln_f': nn.LayerNorm(self.cfg.n_dim)
        })
        self.lm_head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)
    
    def forward(self, x):
        B, S = x.size()

        assert S <= self.cfg.n_positions, 'seq len > n_positions'
        
        pos = torch.arange(start=0, end=S, device=x.device)
        wpe = self.transformer.wpe(pos)
        wte = self.transformer.wte(x)

        x = wte + wpe

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        return logits
    
    def generate(self, x, max_len, prompts=5):
        assert x.dim() < 3, 'dim > 2'

        if x.dim() == 1:
            x = x.unsqueeze(0).repeat(prompts, 1)

        self.eval()
        with torch.no_grad():
            while x.size(1) < max_len:
                logits = self(x)
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, self.cfg.k, dim=-1)

                ix = torch.multinomial(topk_probs, 1)

                pred = torch.gather(topk_indices, dim=-1, index=ix)
 
                x = torch.cat([x, pred], dim=-1)

        return x

                
    
    @classmethod
    def from_pretrained(cls, model_type):

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, 'model_type not gpt2'

        config_args = {
            'gpt2':dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['n_positions'] = 1024

        config = ModelConfig(**config_args)

        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()

        sd_keys = [key for key in sd_keys if not key.endswith('.attn.bias')]
        bit_config = BitsAndBytesConfig(load_in_8bit=True)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type, quantization_config=bit_config)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if k.endswith(transposed):
                assert sd[k].shape == sd_hf[k].shape[::-1], 'model shape != transpose hf model shape'
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())


            else:
                assert sd[k].shape == sd_hf[k].shape, 'model shape != hf model shape'
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])


        return model