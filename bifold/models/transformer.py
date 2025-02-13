import torch
from einops import rearrange
from torch import nn

from .modules import FeedForward, PreNorm


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, attention_masks=None):

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attention_masks is not None:
            # Use a very large number as value but not float("-inf") due to stability
            dots.masked_fill_(attention_masks[:, None, :, None] == 0, -100000)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    ),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ])
            )

    def forward(self, x, attention_masks=None):
        for attn, ff in self.layers:  # type: ignore
            x = attn(x, attention_masks=attention_masks) + x
            x = ff(x) + x
        return x
