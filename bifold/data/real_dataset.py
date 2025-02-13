import os

import numpy as np
from PIL import Image

from . import BaseDataset
from .vr_folding_utils import folding_actions


class RealDataset(BaseDataset):
    fx = 605.70623779
    fy = 605.82971191

    def __init__(self, cfg, *args, **kwargs):
        self.max_context_length = cfg.max_context_length
        super().__init__(cfg, *args, **kwargs, max_context_length=self.max_context_length)
        assert self.partition == "test", "This dataset cannot be used for other than testing"

        self.depths = []
        self.rgbs = []
        self.masks = []
        self.instructions = []
        self.contexts = []
        self.ground_truth = []

        for category in os.listdir(self.dataset_path):
            if category != "empty":
                np_files = os.listdir(
                    os.path.join(self.dataset_path, category, "cropped_raw_depth")
                )
                groups = {}
                for np_file in np_files:
                    *prefix, _ = os.path.splitext(np_file)[0].split("_")
                    prefix_str = "_".join(prefix)
                    if prefix_str not in groups:
                        groups[prefix_str] = [np_file]
                    else:
                        groups[prefix_str].append(np_file)

                for prefix, np_files in groups.items():
                    cloth_id, *category, instruction_idx = prefix.split("_")
                    category = "_".join(category)
                    instruction_idx = int(instruction_idx)

                    try:
                        instructions = self.get_instructions(category, instruction_idx)
                        self.instructions.extend(instructions)
                        for i in range(len(instructions)):
                            self.depths.append(
                                np.median(
                                    [
                                        np.load(
                                            os.path.join(
                                                self.dataset_path,
                                                category,
                                                "cropped_raw_depth",
                                                np_file,
                                            )
                                        )
                                        for np_file in np_files
                                    ],
                                    axis=0,
                                )
                                / self.depth_scale
                            )
                            self.rgbs.append(
                                np.array(
                                    Image.open(
                                        os.path.join(
                                            self.dataset_path,
                                            category,
                                            "cropped_rgb",
                                            np_files[0].replace(".npy", ".png"),
                                        )
                                    )
                                )
                            )
                            self.masks.append(
                                np.array(
                                    Image.open(
                                        os.path.join(
                                            self.dataset_path,
                                            category,
                                            "cropped_mask",
                                            np_files[0].replace(".npy", ".png"),
                                        )
                                    )
                                )[:, :, 0]
                                / 255
                            )

                            *head, _ = np_files[0].split("_")
                            gt_file = os.path.join(
                                self.dataset_path,
                                category,
                                "cropped_annotations",
                                "_".join(head) + ".npy",
                            )
                            if os.path.isfile(gt_file):
                                gt = np.load(gt_file)
                                if len(gt.shape) == 1:
                                    gt = gt[None, :]
                                self.ground_truth.append(gt)
                            else:
                                self.ground_truth.append(None)

                            context_files = []
                            for ctx_idx in range(0, instruction_idx):
                                *head, _, tail = np_files[0].split("_")
                                context_files.append("_".join([*head, f"{ctx_idx}", tail]))
                            self.contexts.append([
                                {
                                    "depth": np.load(
                                        os.path.join(
                                            self.dataset_path,
                                            category,
                                            "cropped_raw_depth",
                                            context_file,
                                        )
                                    )
                                    / self.depth_scale,
                                    "rgb": np.array(
                                        Image.open(
                                            os.path.join(
                                                self.dataset_path,
                                                category,
                                                "cropped_rgb",
                                                context_file.replace(".npy", ".png"),
                                            )
                                        )
                                    ),
                                    "mask": np.array(
                                        Image.open(
                                            os.path.join(
                                                self.dataset_path,
                                                category,
                                                "cropped_mask",
                                                context_file.replace(".npy", ".png"),
                                            )
                                        )
                                    )[:, :, 0]
                                    / 255,
                                }
                                for context_file in context_files
                            ])
                    except ValueError:
                        # There is no action to be done
                        pass

        self.K = np.eye(4)
        self.K[0, 0] = self.fx
        self.K[1, 1] = self.fy
        self.K[0, 2] = self.depths[0].shape[0] / 2
        self.K[1, 2] = self.depths[0].shape[1] / 2

        self.matrix_world_to_camera = np.eye(4)

        assert len(self.depths) == len(self.rgbs) == len(self.instructions)

    @staticmethod
    def get_instructions(category, instruction_idx):
        if category == "long_shirt":
            if instruction_idx == 0:
                return [
                    instruction.format(which="left") for instruction in folding_actions["sleeves"]
                ]
            elif instruction_idx == 1:
                return [
                    instruction.format(which="right") for instruction in folding_actions["sleeves"]
                ]
            elif instruction_idx == 2:
                return [
                    instruction.format(garment="tshirt", which1="top", which2="bottom")
                    for instruction in folding_actions["fold"]
                ]
            else:
                raise ValueError(f"Instruction {instruction_idx} for {category} not supported")
        else:
            if category == "short_shirt":
                garments = ["tshirt"]
            elif category == "dress":
                garments = ["dress", "skirt", "top"]
            elif category == "pants":
                garments = ["trousers"]
            elif category == "towel":
                garments = [
                    "towel",
                    "cloth",
                    "tshirt",
                    "trousers",
                    "pants",
                    "top",
                    "skirt",
                ]
            else:
                raise ValueError(f"Category {category} not supported")

            instructions = []
            for garment in garments:
                if instruction_idx == 0:
                    instructions.extend([
                        instruction.format(garment=garment, which1="left", which2="right")
                        for instruction in folding_actions["fold"]
                    ])
                elif instruction_idx == 1:
                    instructions.extend([
                        instruction.format(garment=garment, which1="top", which2="bottom")
                        for instruction in folding_actions["fold"]
                    ])
                else:
                    raise ValueError(f"Instruction {instruction_idx} for {category} not supported")
            return instructions

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, index):
        kwargs = {}
        if self.ground_truth[index] is not None:
            left_pick = self.ground_truth[index][:, [0, 1]]
            left_place = self.ground_truth[index][:, [2, 3]]
            right_pick = self.ground_truth[index][:, [4, 5]]
            right_place = self.ground_truth[index][:, [6, 7]]

            # Do not penalize symmetry
            kwargs["left_pick"] = np.r_[left_pick, right_pick, left_place, right_place]
            kwargs["left_place"] = np.r_[left_place, right_place, left_pick, right_pick]
            kwargs["right_pick"] = np.r_[right_pick, left_pick, right_place, left_place]
            kwargs["right_place"] = np.r_[right_place, left_place, right_pick, left_pick]

        return_dict = self.processor(
            rgb=self.rgbs[index],
            depth=self.depths[index],
            mask=self.masks[index],
            instruction=self.instructions[index],
            context=self.contexts[index],
            K=self.K,
            matrix_world_to_camera=self.matrix_world_to_camera,
            **kwargs,
        )
        return return_dict
