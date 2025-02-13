import torch
from torch import nn
from transformers import T5EncoderModel

from . import BaseModel
from .clip import _MODELS, load


class FiLMConv(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super(FiLMConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gamma = nn.Linear(condition_dim, out_channels)
        self.beta = nn.Linear(condition_dim, out_channels)

    def forward(self, x, condition):
        gamma = self.gamma(condition).unsqueeze(2).unsqueeze(3)
        beta = self.beta(condition).unsqueeze(2).unsqueeze(3)
        return self.conv(x) * (1 + gamma) + beta


class FiLMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super(FiLMBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.film = FiLMConv(out_channels, out_channels, condition_dim)

    def forward(self, x1, x2, condition):
        x1 = self.convt(x1)
        x = torch.cat([x2, x1], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.film(x, condition)
        x = self.relu(x)
        return x


class TextConditionedUNet(BaseModel):
    def __init__(self, text_encoder, features, **kwargs):
        super().__init__(**kwargs)
        # text
        if text_encoder in _MODELS:
            self.clip_encoder, _ = load(text_encoder, device="cpu")
            self.frozen_submodule("clip_encoder")
            dim = self.clip_encoder.transformer.width
        else:
            self.clip_encoder = None
            self.text_encoder = T5EncoderModel.from_pretrained(text_encoder)
            dim = self.text_encoder.config.d_model  # type: ignore
            self.frozen_submodule("text_encoder")

        # Encoder - contracting path
        self.encoder = nn.ModuleList()
        for i in range(len(features)):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        1 if i == 0 else features[i - 1],
                        features[i],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(features[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features[i], features[i], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features[i]),
                    nn.ReLU(inplace=True),
                )
            )
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder - expanding path with skip connections
        self.decoder = nn.ModuleList()
        for i in range(len(features) - 2, -1, -1):
            self.decoder.append(FiLMBlock(features[i + 1], features[i], dim))

        if self.is_bimanual:
            # pick decoder
            self.left_pick_decoder = nn.Conv2d(features[0], 1, kernel_size=1)
            self.right_pick_decoder = nn.Conv2d(features[0], 1, kernel_size=1)

            # place decoder
            self.left_place_decoder = nn.Conv2d(features[0], 1, kernel_size=1)
            self.right_place_decoder = nn.Conv2d(features[0], 1, kernel_size=1)
        else:
            # pick decoder
            self.pick_decoder = nn.Conv2d(features[0], 1, kernel_size=1)

            # place decoder
            self.place_decoder = nn.Conv2d(features[0], 1, kernel_size=1)

    def encode_text(self, text):
        if self.clip_encoder:
            x = self.clip_encoder.encode_text_with_embeddings(text)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        else:
            x = self.text_encoder(text).last_hidden_state[:, 0, :]  # type: ignore
        return x

    def forward(self, x):
        return_dict = {}

        # Process text using your chosen model (e.g., Bert)
        with torch.no_grad():
            text_embedding = self.encode_text(x["instruction"])

        # Encode image features
        encoded_features = []
        x = x["depth"]
        for i, encoder_block in enumerate(self.encoder):
            if i != 0:
                x = self.pool(x)
            x = encoder_block(x)
            if i < len(self.encoder) - 1:
                encoded_features.append(x)

        # Decode and apply multiple heads
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x, encoded_features[-(i + 1)], text_embedding)

        if self.is_bimanual:
            return_dict["left_pick_heatmap"] = self.left_pick_decoder(x).squeeze(1).sigmoid()
            return_dict["right_pick_heatmap"] = self.right_pick_decoder(x).squeeze(1).sigmoid()
            return_dict["left_place_heatmap"] = self.left_place_decoder(x).squeeze(1).sigmoid()
            return_dict["right_place_heatmap"] = self.right_place_decoder(x).squeeze(1).sigmoid()

        else:
            return_dict["pick_heatmap"] = self.pick_decoder(x).squeeze(1).sigmoid()
            return_dict["place_heatmap"] = self.place_decoder(x).squeeze(1).sigmoid()
        return return_dict
