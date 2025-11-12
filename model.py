import torch
import math
import torch.nn as nn
    
class Embedding(nn.Module):
    def __init__(self, vocab, dim, sqrt_dim=False):
        '''CUSTOM embedding layer
           
            Args:
                vocab(int): vocabulary size
                dim(int): model` dim
                sqrt_dim(bool): if True -> embedding(vocab, dim) * sqrt(dim)
                                DEFAULT: False
            
            Examples:
                >>> embedding = Embedding(10000, 512, sqrt_dim=True)
                >>> input = torch.randint(0, 10000, (32, 128)) #batch=32, seq_len=128
                >>> outputs = embedding(input) #shape: (32, 128, 512)
           
           '''
        super().__init__()

        self.vocab = vocab
        self.dim = dim
        self.sqrt_dim = sqrt_dim

        self.emb = nn.Embedding(vocab, dim)
    
    def forward(self, x):
        x = self.emb(x)

        if self.sqrt_dim:
            x = x * math.sqrt(self.dim)

        return x


class LearnedPosEmb(nn.Module):
    def __init__(self, seq_len, dim):
        '''Position learned
           
            Args:
                seq_len(int): sequence length
                dim(int): model` dim
            
            Examples:
                >>> embedding  = Embedding(10000, 512)
                >>> emb = embedding(input)
                >>> position = LearnedPosEmb(128, 512)
                >>> output = pos(emb) #emb + learned position with shape: (seq_len, dim)

            Returns:
                matrix: embedding + position
           
           '''
        super().__init__()
        self.emb = Embedding(seq_len, dim, sqrt_dim=False)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device)
        return x + self.emb(pos)
    

class SinCosPolEmb(nn.Module):
    def __init__(self, seq_len, dim):
        '''Sinus, Cosinus position embedding: Not learned
           
            Args:
                seq_len(int): sequence length
                dim(int): model` dim and dim % 2 == 0
            
            Examples:
                >>> embedding  = Embedding(10000, 512)
                >>> emb = embedding(input)
                >>> position = SinCosPolEmb(128, 512)
                >>> output = pos(emb) #emb + learned position with shape: (seq_len, dim)

            Returns:
                matrix: embedding + position
           
           '''
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len

        assert dim % 2 == 0, 'dim doesn`t divive to 2'

        pe = torch.zeros(seq_len, dim)

        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0)/dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        assert dim % heads == 0, 'dim doesn`t divide to heads'

        self.dim = dim
        self.heads = heads
        self.heads_dim = dim // heads

        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.k = nn.Linear(self.dim, self.dim, bias=False) 
        self.v = nn.Linear(self.dim, self.dim, bias=False) 

        self.proj = nn.Linear(self.heads_dim * heads, dim, bias=False)


    def scaled_dot_production_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.heads_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, V)
        return output
    
    def split_heads(self, x):
        batch, seq_len, dim = x.size()
        return x.view(batch, seq_len, self.heads, self.heads_dim).transpose(1, 2)

    def combine_heads(self, x):
        batch, heads, seq_len, dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
    
    def forward(self, q,k,v, mask=None):

        q = self.split_heads(self.q(q))
        k = self.split_heads(self.k(k))
        v = self.split_heads(self.v(v))


        attention = self.scaled_dot_production_attention(q, k, v, mask=mask)
        
        outputs = self.proj(self.combine_heads(attention))

        return outputs
    

    
class PositionForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.fc1(x))))
    
    
    

class EncoderBlock(nn.Module):
    def __init__(self, seq_len, dim, hidden_dim, heads, dropout, pos_learn=False):
        super().__init__()

        self.attn = MultiHeadAttention(dim=dim, heads=heads)
        self.ffn = PositionForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.drop = nn.Dropout(dropout)

    
    def forward(self, x, mask):
        

        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))

        return x


    
class DecoderBlock(nn.Module):
    def __init__(self, dim, hidden_dim, heads, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(dim, heads)
        self.cross_attention = MultiHeadAttention(dim, heads)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.drop = nn.Dropout(dropout)

        self.ffn = PositionForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)


    def forward(self, x, enc_out, src_mask, trg_mask):
        
        self_attn = self.self_attention(x, x, x, trg_mask)
        x = self.norm1(x + self.drop(self_attn))

        cross_attn = self.cross_attention(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop(cross_attn))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.drop(ffn_out))

        return x
    

class TransFormer(nn.Module):
    def __init__(self, dim=512, src_vocab=10000, trg_vocab=10000, 
                 heads=8, seq_len=100, hidden_dim=2048, dropout=0.1, 
                 layers=6, pos_learn=False, pad=0, eos=1, sqrt_dim=False,
                 init_weights=False):
        
        """Transformer class for sequence-to-sequence tasks.

            Args:
                dim: Model dimension (d_model)
                src_vocab: Vocabulary size for input language
                trg_vocab: Vocabulary size for target language
                heads: Number of attention heads (must divide dim evenly)
                seq_len: Maximum sequence length
                hidden_dim: Hidden dimension in FFN layers
                dropout: Dropout rate (0.0-1.0)
                layers: Number of encoder and decoder layers
                pos_learn: If True, uses learned positional embeddings
                pad: Padding token ID
                eos: End-of-sequence token ID
                sqrt_dim: If True, scales embeddings by sqrt(dim)
                init_weights: If True, applies custom weight initialization

            Example:
                >>> model = TransFormer(src_vocab=8500, trg_vocab=8000)
                >>> src = torch.randint(0, 8500, (32, 100))
                >>> trg = torch.randint(0, 8000, (32, 100))
                >>> output = model(src, trg)  # shape: (32, 100, 8000)
            """
        
        super().__init__()

        self.pad = pad
        self.eos = eos
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.encoder_emb = Embedding(src_vocab, dim, sqrt_dim=sqrt_dim)
        self.decoder_emb = Embedding(trg_vocab, dim, sqrt_dim=sqrt_dim)

        self.pos_emd = LearnedPosEmb(seq_len=seq_len, dim=dim) \
        if pos_learn else SinCosPolEmb(seq_len=seq_len, dim=dim)


        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, trg_vocab)

        self.encoders = nn.ModuleList([EncoderBlock(seq_len=seq_len, dim=dim, 
                                     hidden_dim=hidden_dim, heads=heads, 
                                     dropout=dropout, pos_learn=pos_learn) 

                                     for _ in range(layers)])

        self.decoders = nn.ModuleList([DecoderBlock(dim=dim, hidden_dim=hidden_dim, 
                                     heads=heads, dropout=dropout) 

                                     for _ in range(layers)])
        if init_weights:
            self.apply(self._init_weights)
    
    @torch.no_grad()
    def _init_weights(self, module):

        if isinstance(module, Embedding):
            nn.init.normal_(module.emb.weight, mean=0, std=0.02)

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def total_parameters(self):
        L = sum(p.numel() for p in self.parameters() if p.requires_grad)
        NL = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'Learn':L, 'Not learn':NL}
    

    def generate_mask(self, src, tgt):
        # Source mask (padding) - добавляем размерность для heads
        src_mask = (src != self.pad).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, src_len]
        
        # Target mask (padding + future)
        tgt_pad_mask = (tgt != self.pad).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, tgt_len]
        seq_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones(
            (1, 1, seq_len, seq_len), device=tgt.device)).bool()  # [1, 1, tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return src_mask, tgt_mask
    

    
    def forward(self, src, trg):
        src_mask, trg_mask = self.generate_mask(src, trg)

        encoder_emb = self.pos_emd(self.encoder_emb(src))
        decoder_emb = self.pos_emd(decoder_emb(trg))

        encoder_emb = self.drop(self.encoder_emb(src))
        decoder_emb = self.drop(self.decoder_emb(trg))

        encoder_out = encoder_emb

        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, src_mask)

        decoder_out = decoder_emb

        for decoder in self.decoders:
            decoder_out = decoder(decoder_out, encoder_out, src_mask, trg_mask)

        output = self.linear(decoder_out)
        return output
    