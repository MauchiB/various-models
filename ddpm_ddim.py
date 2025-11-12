import torch
import torch.nn as nn
import torch.nn.functional as F
import math





class DifussionForwardProcess:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alpha = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alpha, 0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)


    def add_noise(self, x, noise, t):

        sqrt_minus_alpha_bars = self.sqrt_minus_alpha_bars.to(x.device)[t][:, None, None, None]
        sqrt_alpha_bars = self.sqrt_alpha_bars.to(x.device)[t][:, None, None, None]

        return sqrt_alpha_bars * x + sqrt_minus_alpha_bars * noise
    




class DDPM:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):


        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alpha = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alpha, 0)


    def reverse_process(self, xt, noise_pred, t):

        x0 = xt - (1 - torch.sqrt(self.alpha_bars.to(xt.device)[t])) * noise_pred
        x0 = x0 * 1/torch.sqrt(self.alpha_bars.to(xt.device)[t])
        x0 = torch.clamp(x0, -1, 1)


        mean = xt - ((1 - self.alpha.to(xt.device)[t]) * noise_pred / (torch.sqrt(1 - self.alpha_bars.to(xt.device)[t])))
        mean = xt / torch.sqrt(self.alpha.to(xt.device)[t])

        if t == 0:
            return mean, x0
        else:
            varience = (1 - self.alpha_bars.to(xt.device)[t - 1]) / (1 - self.alpha_bars.to(xt.device)[t])
            varience = varience * self.betas[t]

            sigma = varience ** 0.5

            z = torch.randn(xt.shape).to(xt.device)

            return mean + sigma * z, x0
        


class DDIM:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):


        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alpha = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alpha, 0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)


    def reverse_process(self, xt, noise_pred, t_cur, t_prev):

        x0 = (xt - self.sqrt_minus_alpha_bars[t_cur] * noise_pred) / self.sqrt_alpha_bars[t_cur]
        x_pred = x0 * self.sqrt_alpha_bars[t_prev] + noise_pred * self.sqrt_minus_alpha_bars[t_prev]

        return x_pred

        




def get_time_embedding(
    time_steps: torch.Tensor,
    t_emb_dim: int
) -> torch.Tensor:

    """
    Transform a scalar time-step into a vector representation of size t_emb_dim.

    :param time_steps: 1D tensor of size -> (Batch,)
    :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)

    :return tensor of size -> (B, t_emb_dim)
    """

    assert t_emb_dim%2 == 0, "time embedding must be divisible by 2."

    factor = 2 * torch.arange(start = 0,
                              end = t_emb_dim//2,
                              dtype=torch.float32,
                              device=time_steps.device
                             ) / (t_emb_dim)

    factor = 10000**factor

    t_emb = time_steps[:,None] 
    t_emb = t_emb/factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1) 

    return t_emb




class ResBlock(nn.Module):
    def __init__(self, in_ch, ch_out, temb_ch, groups=32, droprate=0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.norm2 = nn.GroupNorm(groups, ch_out)

        self.conv1 = nn.Conv2d(in_ch, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop = nn.Dropout(droprate)

        self.linear1 = nn.Linear(temb_ch, ch_out, bias=True)

        if in_ch != ch_out:
            self.proj = nn.Conv2d(in_ch, ch_out, 1, 1, bias=True)
        else:
            self.proj = nn.Identity()


    def forward(self, x, temb):
        resuidal = x

        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        temb_proj = self.linear1(F.silu(temb))[:, :, None, None]
        x += temb_proj

        x = F.silu(self.norm2(x))
        x = self.drop(x)
        x = self.conv2(x)

        resuidal = self.proj(resuidal)
        return x + resuidal
    



class AttnBlock(nn.Module):
    def __init__(self, ch_in, ch_out, groups=32) -> None:
       super().__init__()
       self.conv_proj = nn.Conv2d(ch_in, ch_out * 3, kernel_size=1, stride=1, padding=0, bias=True)
       self.conv_out = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
       self.norm = nn.GroupNorm(groups, ch_in)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.conv_proj(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = torch.permute(q.view(B, C, H * W), dims=[0, 2, 1]) # B x H*W x C
        k = k.view(B, C, H * W)                                # B x C x H*W
        attn = torch.matmul(q, k) / math.sqrt(C)               # B x H*W x H*W
        attn = F.softmax(attn, dim=-1)

        v = torch.permute(v.view(B, C, H * W), dims=[0, 2, 1]) # B x H*W x C
        o = torch.matmul(attn, v)

        o = torch.permute(o, dims=[0, 2, 1]).view(B, C, H, W)
        o = self.conv_out(o)

        return o + x
    





class Encoder(nn.Module):
    def __init__(self, ch=128, droprate=0.1, groups=32) -> None:
        super().__init__()
        # Initial convolution для 28x28
        self.conv = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1, bias=True)            # 28x28 x ch

        # Уровень 1: 28x28
        self.resblock1_1 = ResBlock(ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock1_2 = ResBlock(ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.downsample1 = nn.Conv2d(ch, 2*ch, kernel_size=3, stride=2, padding=1, bias=True)     # 14x14 x ch

        # Уровень 2: 14x14
        self.resblock2_1 = ResBlock(2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups) # 14x14 x 2ch
        self.resblock2_2 = ResBlock(2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.downsample2 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=2, padding=1, bias=True) # 7x7 x 2ch

        # Уровень 3: 7x7 (нужно добавить padding для выравнивания размеров)
        self.resblock3_1 = ResBlock(2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock3_1 = AttnBlock(2*ch, 2*ch, groups=groups)
        self.resblock3_2 = ResBlock(2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        # Для 7x7 -> 4x4 используем padding=0 или адаптивный даунсемплинг
        self.downsample3 = nn.Sequential(
            nn.Conv2d(2*ch, 4*ch, kernel_size=3, stride=2, padding=1, bias=True) # 8x8 -> 4x4
        ) # 4x4 x 2ch

        # Уровень 4: 4x4
        self.resblock4_1 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups) # 4x4 x 4ch
        self.resblock4_2 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)

    def forward(self, x, temb):
        hs = [] # Список для skip-connections

        # Initial convolution
        x = self.conv(x)


        x = self.resblock1_1(x, temb)
        hs.append(x) # 32 32

        x = self.resblock1_2(x, temb)
        hs.append(x) # 32 32


        x = self.downsample1(x) #16 16


        x = self.resblock2_1(x, temb)
        hs.append(x) #16 16

        x = self.resblock2_2(x, temb)
        hs.append(x) # 16 16


        x = self.downsample2(x) # 8 8


        # Level 3: 8 8
        x = self.resblock3_1(x, temb)
        hs.append(x) # 8 8


        x = self.attnblock3_1(x)

        x = self.resblock3_2(x, temb)
        hs.append(x) # 8 8


        x = self.downsample3(x) # -> 4x4 x 2ch

        # Level 4: 4x4
        x = self.resblock4_1(x, temb)
        hs.append(x) # 4x4 x 4ch


        x = self.resblock4_2(x, temb)
        hs.append(x) # 4x4 x 4ch


        return x, hs

class Decoder(nn.Module):
    def __init__(self, ch=128, droprate=0.1, groups=32) -> None:
        super().__init__()

        self.resblock1_1 = ResBlock(4*ch + 4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock1_2 = ResBlock(4*ch + 4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(4*ch, 2*ch, kernel_size=3, stride=1, padding=1, bias=False)
        ) 

        self.resblock2_1 = ResBlock(2*ch + 2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock2_2 = ResBlock(2*ch + 2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock2_1 = AttnBlock(2*ch, 2*ch, groups=groups)
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.resblock3_1 = ResBlock(2*ch + 2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock3_2 = ResBlock(2*ch + 2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        ) 

  
        self.resblock4_1 = ResBlock(ch + ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock4_2 = ResBlock(ch + ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.to_rgb = nn.Sequential(
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x, temb, hs):

        x = self.resblock1_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock1_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.upsample1(x)


        x = self.resblock2_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.attnblock2_1(x)
        x = self.resblock2_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.upsample2(x)


        x = self.resblock3_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock3_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.upsample3(x)


        x = self.resblock4_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock4_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.to_rgb(x)

        return x
    



class UNet(nn.Module):
    def __init__(self, ch=128, droprate=0.1, groups=32, device="cuda") -> None:
        super().__init__()
        # Time Embedding
        self.ch = ch
        self.Proj = nn.Sequential(nn.Linear(ch, 4*ch),
                                  nn.SiLU(),
                                  nn.Linear(4*ch, 4*ch))
 
        self.encoder = Encoder(ch, droprate, groups)

        self.resblock1 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock = AttnBlock(4*ch, 4*ch, groups=groups)
        self.resblock2 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
 
        self.decoder = Decoder(ch, droprate, groups)

    def forward(self, x, t):
        hs = []

        temb = get_time_embedding(t, self.ch)
        temb = self.Proj(temb)
   
        x, hs = self.encoder(x, temb)
 
        x = self.resblock1(x, temb)
        x = self.attnblock(x)
        x = self.resblock2(x, temb)
      
        x = self.decoder(x, temb, hs)
        return x
