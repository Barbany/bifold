import torch
from einops import rearrange, repeat
from peft.tuners.lora import LoraConfig, LoraModel
from torch import nn
from transformers import AutoModel

from . import BaseModel, Components


class SigLip(BaseModel):
    def __init__(
        self,
        patch_size,
        dim,
        lora,
        r,
        lora_alpha,
        lora_dropout,
        automodel_name,
        target_modules,
        pick_place_model,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        siglip_model = AutoModel.from_pretrained(automodel_name).to(self.device)
        self.hidden_size = dim
        assert self.hidden_size == siglip_model.config.vision_config.hidden_size
        assert self.hidden_size == siglip_model.config.text_config.hidden_size

        self.num_patches = (self.image_size // self.patch_size) ** 2

        if lora:
            config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
            )
            self.siglip_model = LoraModel(siglip_model, config, "siglip_adapter")
        else:
            self.siglip_model = siglip_model
            self.frozen_submodule("siglip_model")

        self.text_token = nn.Parameter(torch.randn(1, 1, dim))
        self.image_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pick_place = Components.get_by_name(
            pick_place_model,
            dim=dim,
            patch_size=patch_size,
            num_patches=self.num_patches,
            **kwargs,
        )

    def forward(self, x):
        backend_outputs = self.siglip_model(input_ids=x["instruction"], pixel_values=x["rgb"])

        # (batch size, num patches, hidden_size)
        image_features = backend_outputs.vision_model_output.last_hidden_state
        b, _, _ = image_features.shape
        image_tokens = repeat(self.image_token, "1 1 d -> b 1 d", b=b)
        image_features = torch.cat((image_tokens, image_features), dim=1)

        # (batch size, num_tokens, hidden_size)
        text_features = backend_outputs.text_model_output.last_hidden_state
        text_tokens = repeat(self.text_token, "1 1 d -> b 1 d", b=b)
        text_features = torch.cat((text_tokens, text_features), dim=1)

        return self.pick_place(text_features, image_features)


class SiglipSequential(SigLip):
    def __init__(
        self,
        context_length,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.context_pos_embedding = nn.Parameter(
            torch.randn(1, context_length * (self.num_patches + 1), self.hidden_size)
        )

    def forward(self, x):
        b, *_ = x["rgb"].shape
        backend_outputs = self.siglip_model(input_ids=x["instruction"], pixel_values=x["rgb"])

        # (batch size, num patches, hidden_size)
        image_features = backend_outputs.vision_model_output.last_hidden_state
        image_tokens = repeat(self.image_token, "1 1 d -> b 1 d", b=b)
        image_features = torch.cat((image_tokens, image_features), dim=1)
        _, n, _ = image_features.shape

        # (batch size, num_tokens, hidden_size)
        text_features = backend_outputs.text_model_output.last_hidden_state
        text_tokens = repeat(self.text_token, "1 1 d -> b 1 d", b=b)
        text_features = torch.cat((text_tokens, text_features), dim=1)
        _, n_txt, _ = text_features.shape

        _, t, *_ = x["rgb_context"].shape
        backend_context_outputs = self.siglip_model.vision_model(
            pixel_values=rearrange(x["rgb_context"], "b t c h w -> (b t) c h w"),
        )
        image_context_features = rearrange(
            backend_context_outputs.last_hidden_state,
            "(b t) n d -> b t n d",
            b=b,
            t=t,
        )
        image_context_tokens = repeat(self.image_token, "1 1 d -> b t 1 d", b=b, t=t)
        image_context_features = torch.cat((image_context_tokens, image_context_features), dim=2)
        image_context_features = rearrange(
            image_context_features,
            "b t n d -> b (t n) d",
        )
        image_context_features = image_context_features + self.context_pos_embedding
        attention_masks = torch.cat(
            [
                torch.ones(b, n_txt).to(self.device),
                repeat(
                    x["context_attention_mask"],
                    "b t -> b (t n)",
                    b=b,
                    n=n,
                ),
                torch.ones(b, n).to(self.device),
            ],
            dim=-1,
        )
        return self.pick_place(
            text_features,
            image_context_features,
            image_features,
            attention_masks=attention_masks,
            modalities=[0, 1, 1],
        )
