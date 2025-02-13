import torch
from timm.models.vision_transformer import Block
from torch import nn

from .utils import get_2d_sincos_pos_embed


class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.decoder_net = self._init_decoder()

    def _init_decoder(self):
        intermediate_channels1 = int(self.input_dim / 2)
        intermediate_channels2 = int(self.input_dim / 4)
        in_channels = [
            self.input_dim,
            intermediate_channels1,
            intermediate_channels1,
            intermediate_channels2,
            intermediate_channels2,
        ]
        out_channels = [
            intermediate_channels1,
            intermediate_channels1,
            intermediate_channels2,
            intermediate_channels2,
            self.output_dim,
        ]
        modules = []
        for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            modules.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=(0, 0),
                )
            )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

        return nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder_net(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim,
        decoder_embed_dim,
        patch_size,
        num_patches,
        decoder_num_heads,
        decoder_mlp_ratio,
        decoder_depth,
        out_channels,
        **kwargs,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                decoder_mlp_ratio,
                norm_layer=nn.LayerNorm,  # type: ignore
            )
            for i in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.out_dim = patch_size**2 * out_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.out_dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
