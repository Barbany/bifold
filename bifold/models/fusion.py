import torch
from torch import nn

from .transformer import Transformer
from .utils import init_weights


class ConcatTransformer(nn.Module):
    def __init__(
        self, dim, heads, depth, dropout, mlp_ratio, num_modalities=2, num_registers=0, **kwargs
    ):
        super().__init__()
        self.num_modalities = num_modalities
        # type embedding
        self.token_type_embeddings = nn.Embedding(self.num_modalities, dim)
        self.token_type_embeddings.apply(init_weights)

        self.dim_head = int(dim / heads)
        self.mlp_dim = dim * mlp_ratio
        # transformer encoder
        self.transformer_encoder = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=dropout,
        )

        if num_registers > 0:
            self.registers = nn.Parameter(torch.randn(num_registers, dim))
        else:
            self.registers = None

    def forward(self, *inputs, modalities=None, **kwargs):
        if modalities is None:
            # Assume every inputs comes from a different modality
            modalities = range(len(inputs))
        assert len(inputs) == len(modalities)

        if self.registers:
            concat = [self.registers]
        else:
            concat = []

        for mod, inp in zip(modalities, inputs):
            # type encoding
            type_embeddings = self.token_type_embeddings(
                torch.full((inp.shape[0], inp.shape[1]), mod, device=inp.device).long()
            )

            inp += type_embeddings
            concat.append(inp)

        # concatenate
        x = torch.cat(concat, dim=1)

        # transformer encoder
        x = self.transformer_encoder(x, **kwargs)

        # pick & place heatmap
        # take features corresponding to the last modality
        features = x[:, -inputs[-1].shape[1] :, :]
        return features, None


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, depth, dropout, num_modalities=2, **kwargs):
        super().__init__()
        self.num_modalities = num_modalities
        self.token_type_embeddings = nn.Embedding(self.num_modalities, dim)
        self.token_type_embeddings.apply(init_weights)

        self.num_heads = heads
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )

    def forward(self, *inputs, modalities=None, attention_masks=None, **kwargs):
        if modalities is None:
            # Assume every inputs comes from a different modality
            modalities = range(len(inputs))
        assert len(inputs) == len(modalities)

        concat = []

        for mod, inp in zip(modalities[:-1], inputs[:-1]):
            # type encoding
            type_embeddings = self.token_type_embeddings(
                torch.full((inp.shape[0], inp.shape[1]), mod, device=inp.device).long()
            )

            inp += type_embeddings
            concat.append(inp)

        type_embeddings = self.token_type_embeddings(
            torch.full(
                (inputs[-1].shape[0], inputs[-1].shape[1]),
                modalities[-1],
                device=inputs[-1].device,
            ).long()
        )
        input_tokens = inputs[-1] + type_embeddings

        condition_tokens = torch.cat(concat, dim=1)

        if attention_masks is not None:
            cross_attn_kwargs = {
                "key_padding_mask": None,
                "attn_mask": attention_masks[:, : condition_tokens.shape[1]]
                .unsqueeze(1)
                .repeat(self.num_heads, input_tokens.shape[1], 1),
            }
        else:
            cross_attn_kwargs = {}

        fused_output, attn_output_weights = self.cross_attention(
            query=input_tokens,
            key=condition_tokens,
            value=condition_tokens,
            need_weights=True,
            average_attn_weights=True,
            **cross_attn_kwargs,
        )
        return fused_output, attn_output_weights
