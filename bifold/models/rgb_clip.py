import torch
from einops import repeat
from torch import nn

from . import BaseModel, Components
from .clip import _MODELS, load


class RGBOnly(BaseModel):
    def __init__(
        self,
        patch_size,
        text_dropout,
        rgb_dropout,
        text_encoder,
        pick_place_model,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.rgb_size = self.image_size
        assert (
            self.rgb_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (self.rgb_size // self.patch_size) ** 2

        # text
        assert text_encoder in _MODELS
        self.clip_encoder, _ = load(text_encoder, device="cpu")
        self.frozen_submodule("clip_encoder")
        dim = self.clip_encoder.transformer.width

        self.project = nn.Linear(self.clip_encoder.visual.ln_post.normalized_shape[0], dim)

        self.text_token = nn.Parameter(torch.randn(1, 1, dim))
        # Length of text token + 1
        self.text_pos_embedding = nn.Parameter(torch.randn(1, 78, dim))
        self.text_dropout = nn.Dropout(text_dropout)

        self.rgb_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.rgb_dropout = nn.Dropout(rgb_dropout)

        self.pick_place = Components.get_by_name(
            pick_place_model,
            dim=dim,
            patch_size=patch_size,
            num_patches=num_patches,
            **kwargs,
        )

    def encode_rgb(self, img):
        x = self.clip_encoder.encode_image_with_embeddings(img)
        x = self.project(x)
        x += self.rgb_pos_embedding
        x = self.rgb_dropout(x)
        return x

    def encode_text(self, text):
        x = self.clip_encoder.encode_text_with_embeddings(text)
        b, n, _ = x.shape
        text_tokens = repeat(self.text_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((text_tokens, x), dim=1)
        x += self.text_pos_embedding[:, : (n + 1)]
        x = self.text_dropout(x)
        return x

    def forward(self, x):
        # encode text and rgb
        x_text = self.encode_text(x["instruction"])
        x_rgb = self.encode_rgb(x["rgb"])

        return self.pick_place(x_text, x_rgb)
