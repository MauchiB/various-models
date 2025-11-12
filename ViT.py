import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, token_len):
        super().__init__()


        self.linear = nn.Linear(patch_size * patch_size * 3, token_len)
        self.split = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.norm1 = nn.LayerNorm(token_len)

        self.patch_size = patch_size


    def forward(self, x):
        x = self.split(x).transpose(1, 2)
        return self.norm1(self.linear(x))
    


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_layer, out_layer, dropout=0.1):
        super().__init__()

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer, out_layer),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)
    


class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, dropout=0.1):
        super().__init__()

        self.inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        proj_out = not(num_heads == 1 and head_dim == dim)

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(-1)

        self.dropout = nn.Dropout()

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if proj_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, -1)

        b, n, _ = q.shape

        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        qk = torch.matmul(q, k.transpose(-1, -2))
        qk = qk * 1/math.sqrt(self.head_dim)

        logits = self.dropout(self.softmax(qk))

        out = torch.matmul(logits, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.inner_dim)
        return self.to_out(out)



class Block(nn.Module):
    def __init__(self, layers, dim, num_heads, head_dim, hidden_layer, out_layer, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(layers):
            self.layers.append(nn.ModuleList([Attention(dim, num_heads, head_dim, dropout),
                                              FeedForward(dim, hidden_layer, out_layer, dropout)]))

    def forward(self, x):
        for attn, ff, in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    

class ViT(nn.Module):
    def __init__(self, num_heads, head_dim, mlp_dim, out_mlp, layers, height, width, patch_size, token_len, channels, pool, num_classes, dropout=0.1):
        super().__init__()

        assert height % patch_size == 0 and width % patch_size == 0,'dont divide size on patch size'



        num_patches = (height // patch_size) * (width // patch_size)
        patch_dim = patch_size*patch_size*channels

        assert pool in {'cls', 'mean'}

        self.patch_embedding = PatchEmbedding(patch_size, token_len)


        self.pos_embedding = nn.Parameter(torch.randn([1, num_patches + 1, token_len]))
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, token_len]))
        self.dropout = nn.Dropout(dropout)


        self.transformer = Block(layers, token_len, num_heads, head_dim, mlp_dim, out_mlp, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(token_len, num_classes)

    def get_parameters(self):
        return sum([torch.numel(p) * p.element_size() for p in self.parameters() if p.requires_grad])


    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = self.cls_embedding.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    




