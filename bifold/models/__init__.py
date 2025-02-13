import numpy as np
import torch
from torch import nn

from bifold.env import Action

from .utils import sample_from_heatmap

DUMMY_PICK = -np.ones(2)


class Models:
    @staticmethod
    def get_by_name(cfg, *args, **kwargs):
        if cfg.name == "siglip":
            from .siglip import SigLip as Model
        elif cfg.name == "siglip_sequential":
            from .siglip import SiglipSequential as Model
        elif cfg.name == "rgb_clip":
            from .rgb_clip import RGBOnly as Model
        elif cfg.name == "text_unet":
            from .text_unet import TextConditionedUNet as Model
        else:
            raise ValueError(f"Model {cfg.name} not recognized")

        del cfg.name
        return Model(*args, **cfg, **kwargs)


class Components:
    @staticmethod
    def get_by_name(name, *args, **kwargs):
        if name == "pick_place_convdecoder":
            from .pickplace import PickPlaceConvDecoder as Model
        elif name == "pick_place_transdecoder":
            from .pickplace import PickPlaceTransDecoder as Model
        elif name == "concat_transformer":
            from .fusion import ConcatTransformer as Model
        elif name == "crossattention":
            from .fusion import CrossAttention as Model
        else:
            raise ValueError(f"Model {name} not recognized")
        return Model(*args, **kwargs)


class BaseModel(nn.Module):
    def __init__(
        self,
        image_size,
        is_bimanual,
        device,
        threshold=0.01,
        requires_graph=False,
        constrain_pick_mask=True,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.is_bimanual = is_bimanual
        self.device = device
        self.threshold = 0.01
        self.requires_graph = requires_graph
        self.constrain_pick_mask = constrain_pick_mask
        if not self.constrain_pick_mask:
            print("This run will not constrain the pick to the mask")

    def forward(self, x):
        raise NotImplementedError

    def frozen_submodule(self, submodule_name):
        for param in self.named_parameters():
            if submodule_name in param[0]:
                param[1].requires_grad = False

    def frozen_all(self):
        for param in self.named_parameters():
            param[1].requires_grad = False

    def get_action(self, sample, return_raw_output=False):
        with torch.no_grad():
            output = self.forward(sample)

        if self.is_bimanual:
            if self.requires_graph:
                arange = torch.arange(len(output["left_pick_heatmap"]))
                raw_left_pick = (
                    sample["pixel_sampled_pc"][
                        arange,
                        :,
                        output["left_pick_heatmap"].argmax(axis=1),
                    ]
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
                left_confidence = torch.max(output["left_pick_heatmap"], dim=1)[0].cpu().numpy()

                raw_right_pick = (
                    sample["pixel_sampled_pc"][
                        arange, :, output["right_pick_heatmap"].argmax(axis=1)
                    ]
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
                right_confidence = torch.max(output["right_pick_heatmap"], dim=1)[0].cpu().numpy()
            else:
                raw_left_pick, left_confidence = sample_from_heatmap(
                    output["left_pick_heatmap"],
                    sample["mask"],
                    return_confidence=True,
                )
                raw_right_pick, right_confidence = sample_from_heatmap(
                    output["right_pick_heatmap"],
                    sample["mask"],
                    return_confidence=True,
                )

            # Make sure at least one action is performed
            # If everything above confidence, perform bimanual action
            # If one arm below confidence for each action, perform single arm action with such arm
            # Otherwise, perform arm with the arm whose confidence is higher
            # (2, B, 2)
            pick = np.stack((raw_left_pick, raw_right_pick))
            # (2, B)
            confidences = np.stack((left_confidence, right_confidence))
            B = confidences.shape[-1]
            mask = np.logical_or(
                confidences >= self.threshold,
                confidences.argmax(axis=0) == np.tile(np.arange(2), (B, 1)).T,
            )
            pick[~mask] = DUMMY_PICK
            left_pick, right_pick = pick

            left_place = sample_from_heatmap(output["left_place_heatmap"])
            assert isinstance(left_place, np.ndarray)
            left_place[~mask[0]] = DUMMY_PICK

            right_place = sample_from_heatmap(output["right_place_heatmap"])
            assert isinstance(right_place, np.ndarray)
            right_place[~mask[1]] = DUMMY_PICK

            action = Action(
                left_pick=left_pick,
                right_pick=right_pick,
                left_place=left_place,
                right_place=right_place,
            )
        else:
            if self.requires_graph:
                pick = (
                    sample["pixel_sampled_pc"][
                        torch.arange(len(output["pick_heatmap"])),
                        :,
                        output["pick_heatmap"].argmax(axis=1),
                    ]
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
            else:
                pick = sample_from_heatmap(
                    output["pick_heatmap"],
                    sample.get("mask") if self.constrain_pick_mask else None,
                )

            action = Action(
                pick=pick,
                place=sample_from_heatmap(output["place_heatmap"]),
            )
        if return_raw_output:
            return action, output
        return action
