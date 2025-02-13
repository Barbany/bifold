import os

import numpy as np
import pandas as pd
import zarr
from PIL import Image

from . import BaseDataset
from .bimanual_dataset import get_mask_from_depth
from .utils import parse_list_string


class BimanualDatasetSequential(BaseDataset):
    def __init__(self, cfg, *args, **kwargs):
        self.max_context_length = cfg.max_context_length

        super().__init__(cfg, *args, **kwargs, max_context_length=self.max_context_length)

        zarr_path = os.path.join(self.dataset_path, "vr_folding_dataset.zarr")
        categories = os.listdir(zarr_path)
        self.zarr_datasets = {
            category: zarr.open(os.path.join(zarr_path, category), mode="r")
            for category in categories
        }

        # Read the CSV file
        columns_with_lists = [
            "left_grip_from",
            "left_grip_to",
            "right_grip_from",
            "right_grip_to",
        ]  # Specify the columns containing lists
        converters = {col: parse_list_string for col in columns_with_lists}
        self.actions_df = pd.read_csv(
            os.path.join(
                self.dataset_path,
                "sequential_actions",
                self.partition + ".csv",
            ),
            converters=converters,
            index_col=0,
        )

        self.renders_path = os.path.join(self.dataset_path, "renders")

        self.image_size = cfg.image_size

    @staticmethod
    def get_info_from_action(action):
        frame = None
        left_idx, right_idx = None, None

        if isinstance(action["left_start_idx"], str):
            left_idx = int(action["left_start_idx"].split("_")[-1])
        else:
            frame = action["right_start_idx"]
        if isinstance(action["right_start_idx"], str):
            right_idx = int(action["right_start_idx"].split("_")[-1])
        else:
            frame = action["left_start_idx"]

        if frame is None:
            assert left_idx is not None and right_idx is not None
            if left_idx <= right_idx:
                frame = action["left_start_idx"]
            else:
                frame = action["right_start_idx"]

        category = frame.split("_")[1]
        camera_file = "_".join(frame.split("_")[:-1]) + ".npy"
        return frame, category, camera_file

    @staticmethod
    def get_last_frame_from_action(action):
        frame = None
        left_idx, right_idx = None, None

        if isinstance(action["left_end_idx"], str):
            left_idx = int(action["left_end_idx"].split("_")[-1])
        else:
            frame = action["right_end_idx"]
        if isinstance(action["right_end_idx"], str):
            right_idx = int(action["right_end_idx"].split("_")[-1])
        else:
            frame = action["left_end_idx"]

        if frame is None:
            assert left_idx is not None and right_idx is not None
            if left_idx <= right_idx:
                frame = action["right_end_idx"]
            else:
                frame = action["left_end_idx"]
        return frame

    def __len__(self):
        return len(self.actions_df)

    def project(self, category, frame, vertices, camera_matrix):
        if vertices is None:
            return None
        else:
            sample = self.zarr_datasets[category]["samples"][frame]
            assert isinstance(sample, zarr.Group)
            mesh = sample.get("mesh")
            assert isinstance(mesh, zarr.Group)
            world_coords = np.array(mesh["cloth_verts"][vertices])
            homogeneous_coords = np.column_stack((world_coords, np.ones(world_coords.shape[0])))
            unnorm_coords = (camera_matrix @ homogeneous_coords.T).T
            screen_coords = unnorm_coords[:, :2] / unnorm_coords[:, -2:-1]
            screen_coords[:, 0] = self.image_size - screen_coords[:, 0]
            return screen_coords

    def __getitem__(self, index):
        action = self.actions_df.iloc[index]

        # If there is more then one view, choose at random
        frame, category, camera_file = self.get_info_from_action(action)

        depth = (
            np.array(Image.open(os.path.join(self.renders_path, category, "depth", frame + ".png")))
            / self.cfg.depth_scale
        )
        assert (
            self.image_size == depth.shape[0]
        ), f"Image size {self.image_size} != depth size {depth.shape[0]}"
        mask = get_mask_from_depth(depth)

        K = np.load(os.path.join(self.renders_path, category, "intrinsics.npy"))
        camera_matrix = np.load(
            os.path.join(self.renders_path, category, "camera_matrix", camera_file)
        )

        intr = np.eye(4)
        intr[:3, :3] = K

        # camera_matrix = intr @ matrix_world_to_camera
        matrix_world_to_camera = np.linalg.inv(intr) @ camera_matrix

        rgb = np.array(
            Image.open(os.path.join(self.renders_path, category, "colors", frame + ".png"))
        )
        context = [
            {
                "rgb": np.array(
                    Image.open(
                        os.path.join(self.renders_path, category, "colors", frame_ctx + ".png")
                    )
                ),
                "depth": (
                    np.array(
                        Image.open(
                            os.path.join(self.renders_path, category, "depth", frame_ctx + ".png")
                        )
                    )
                    / self.cfg.depth_scale
                ),
            }
            for frame_ctx in eval(action["context"])
        ]

        for ctx in context:
            ctx["mask"] = get_mask_from_depth(ctx["depth"])

        left_pick_pixels = self.project(
            category=category,
            frame=frame,
            vertices=action["left_grip_from"],
            camera_matrix=camera_matrix,
        )
        right_pick_pixels = self.project(
            category=category,
            frame=frame,
            vertices=action["right_grip_from"],
            camera_matrix=camera_matrix,
        )

        left_place_pixels = self.project(
            category=category,
            frame=action["left_end_idx"],
            vertices=action["left_grip_to"],
            camera_matrix=camera_matrix,
        )
        right_place_pixels = self.project(
            category=category,
            frame=action["right_end_idx"],
            vertices=action["right_grip_to"],
            camera_matrix=camera_matrix,
        )

        assert (
            left_pick_pixels is None
            or np.logical_and(0 < left_pick_pixels, left_pick_pixels < depth.shape).all()
        ), f"Failed on {frame}, {action}"
        assert (
            right_pick_pixels is None
            or np.logical_and(0 < right_pick_pixels, right_pick_pixels < depth.shape).all()
        ), f"Failed on {frame}, {action}"
        assert (
            left_place_pixels is None
            or np.logical_and(0 < left_place_pixels, left_place_pixels < depth.shape).all()
        ), f"Failed on {frame}, {action}"
        assert (
            right_place_pixels is None
            or np.logical_and(0 < right_place_pixels, right_place_pixels < depth.shape).all()
        ), f"Failed on {frame}, {action}"

        return_dict = self.processor(
            rgb=rgb,
            depth=depth,
            mask=mask,
            context=context,
            instruction=action["text"],
            matrix_world_to_camera=matrix_world_to_camera,
            K=K,
            left_pick=left_pick_pixels,
            left_place=left_place_pixels,
            right_pick=right_pick_pixels,
            right_place=right_place_pixels,
        )

        return_dict["frame_start"] = frame
        return_dict["frame_end"] = self.get_last_frame_from_action(action)
        return_dict["context"] = eval(action["context"])
        if len(return_dict["context"]) < self.max_context_length:
            return_dict["context"] = [""] * (
                self.max_context_length - len(return_dict["context"])
            ) + return_dict["context"]
        elif len(return_dict["context"]) > self.max_context_length:
            return_dict["context"] = return_dict["context"][-self.max_context_length :]

        return_dict["context"] = "+".join(return_dict["context"])

        return return_dict
