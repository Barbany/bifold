import os

import numpy as np
import torch
from PIL import Image
from scipy.stats import multivariate_normal
from torchvision.transforms import v2

from bifold.models.clip import _MODELS, tokenize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL.Image import Resampling

    BICUBIC = Resampling.BICUBIC

from . import depth_augmentations, mask_augmentations
from .utils import compute_edge_attr, fps, voxelize_pointcloud

DUMMY = -torch.ones((8, 2))


class Processor:
    def __init__(
        self,
        cfg,
        partition,
        max_context_length=None,
        num_nodes=None,
        neighbor_radius=None,
        voxel_size=None,
        autoprocessor_name=None,
    ):
        self.requires_graph = cfg.requires_graph
        self.image_size = cfg.model_image_size
        self.cfg = cfg
        self.partition = partition
        if self.requires_graph:
            from torch_geometric.data import Data

            self.torch_geometric_data = Data
            self.num_nodes = num_nodes
            self.neighbor_radius = neighbor_radius
            self.voxel_size = voxel_size

        self.resize = v2.Resize(self.image_size, interpolation=BICUBIC)

        common_image_transforms = [
            v2.ToImage(),
            v2.Resize(self.image_size, interpolation=BICUBIC),
            v2.CenterCrop(self.image_size),
        ]

        self.mask_transforms = v2.Compose([*common_image_transforms, mask_augmentations.Round()])

        depth_transforms = []
        if self.partition == "train":
            if cfg.depth_augmentations.random_depth_shift:
                depth_transforms.append(
                    depth_augmentations.DepthScale(
                        min_shift=cfg.depth_augmentations.min_shift,
                        max_shift=cfg.depth_augmentations.max_shift,
                    )
                )
            if cfg.depth_augmentations.add_depth_noise:
                depth_transforms.append(depth_augmentations.DepthNoise())

        depth_transforms.append(depth_augmentations.MaskDepth())

        depth_transforms.extend(common_image_transforms)
        depth_transforms.append(v2.ToDtype(torch.float32))
        if self.cfg.standardize_depth:
            depth_transforms.append(depth_augmentations.TruncatedDepthStandardization())

        self.depth_transforms = v2.Compose(depth_transforms)

        if autoprocessor_name:
            from transformers import AutoProcessor

            self.autoprocessor = AutoProcessor.from_pretrained(autoprocessor_name)
        else:
            self.autoprocessor = None

            self.rgb_transform = v2.Compose([
                *common_image_transforms,
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    self.cfg.image_mean,
                    self.cfg.image_std,
                ),
            ])

            if self.cfg.text_encoder not in _MODELS:
                from transformers import AutoTokenizer

                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg.text_encoder, model_max_length=64
                )
                print(f"{self.cfg.text_encoder} tokenizer")
            else:
                self.tokenizer = None

        if max_context_length is not None:
            self.max_context_length = max_context_length
            self.process_context = True
            self.dummy_depth = torch.ones(
                (max_context_length, 1, self.image_size, self.image_size),
                dtype=torch.float32,
            )
            self.dummy_rgb = torch.ones(
                (max_context_length, 3, self.image_size, self.image_size),
                dtype=torch.float32,
            )
        else:
            self.process_context = False

    def _process_depth(self, depth, mask=None):
        mask = mask if self.cfg.mask_depth else None
        return self.depth_transforms({"depth": depth, "mask": mask})

    def _process_instruction(self, instruction):
        if self.autoprocessor is None:
            if instruction is not None:
                if self.tokenizer is not None:
                    return self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        max_length=77,
                        padding="max_length",
                    ).input_ids.squeeze(0)
                else:
                    return tokenize(instruction).squeeze(0)
        else:
            inputs = self.autoprocessor(text=instruction, padding="max_length", return_tensors="pt")
            if "input_ids" in inputs:
                return inputs["input_ids"].squeeze(0)

    def _process_rgb(self, rgb):
        if self.autoprocessor is None:
            return self.rgb_transform(rgb)
        else:
            inputs = self.autoprocessor(images=rgb, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

    def _process_context(self, context, compute_rgb=False):
        return_dict = {}

        processed_depth = []
        processed_rgb = []

        for item in context[-self.max_context_length :]:
            processed_depth.append(self._process_depth(item["depth"], mask=item.get("mask")))
            if "rgb" in item:
                if "mask" in item:
                    masked_rgb = (
                        item["rgb"] * item["mask"][:, :, None] + (1 - item["mask"][:, :, None]) * 77
                    ).astype(np.uint8)
                    processed_rgb.append(self._process_rgb(masked_rgb))
                else:
                    processed_rgb.append(self._process_rgb(item["rgb"]))

        pad_length = len(self.dummy_depth) - len(processed_depth)
        # Pad context with dummies and indicate them with attention mask
        return_dict["context_attention_mask"] = torch.LongTensor(
            [1] * len(processed_depth) + [0] * (pad_length)
        )
        if processed_depth:
            return_dict["depth_context"] = torch.cat(
                (torch.stack(processed_depth), self.dummy_depth[:pad_length])
            )
            if compute_rgb:
                return_dict["rgb_context"] = torch.cat(
                    (torch.stack(processed_rgb), self.dummy_rgb[:pad_length])
                )
        else:
            return_dict["depth_context"] = self.dummy_depth
            if compute_rgb:
                return_dict["rgb_context"] = self.dummy_rgb

        return return_dict

    def __call__(
        self,
        rgb=None,
        depth=None,
        mask=None,
        instruction=None,
        matrix_world_to_camera=None,
        K=None,
        context=None,
        **kwargs,
    ):
        return_dict = {}

        if depth is not None:
            return_dict["depth"] = self._process_depth(depth, mask)
            depth_ori = self.resize(torch.from_numpy(depth).unsqueeze(0))[0].numpy()

            scale_h = depth.shape[0] / depth_ori.shape[0]
            scale_w = depth.shape[1] / depth_ori.shape[1]

            assert depth.shape[0] == depth.shape[1], (
                "Input image was not square. Need to account for the center crop in intrinsics "
                "and ground truth pixel adjusment"
            )
        else:
            scale_h, scale_w, depth_ori = None, None, None

        if mask is not None:
            return_dict["mask"] = self.mask_transforms(mask)
            mask_ori = return_dict["mask"][0].numpy()
        else:
            mask_ori = None

        if self.requires_graph:

            assert K is not None, "Intrinsics are required for creating the graph"
            scaled_K = K.copy()

            scaled_K[0, :] /= scale_h
            scaled_K[1, :] /= scale_w

            return_dict["graph"], sampled_pc = self.create_graph(
                depth_ori=depth_ori,
                mask=mask_ori,
                matrix_world_to_camera=matrix_world_to_camera,
                K=scaled_K,
            )
            for k, val in kwargs.items():
                if "pick" in k:
                    # Not really a heatmap but point probability
                    # Named heatmap so that it can be used seamlessly with
                    # the heatmap loss framework
                    return_dict[f"{k}_heatmap"] = self.get_pick_graph_heatmap(
                        pick_pixel=val / scale_h,
                        sampled_pc=sampled_pc,
                        depth_ori=depth_ori,
                        matrix_world_to_camera=matrix_world_to_camera,
                        K=scaled_K,
                    )
            if self.partition == "test":
                return_dict["pixel_sampled_pc"] = torch.from_numpy(
                    self.get_pixel_from_world_coords(
                        sampled_pc,
                        matrix_world_to_camera=matrix_world_to_camera,
                        K=scaled_K,
                    )
                )

        if rgb is not None:
            if mask is not None:
                masked_rgb = (rgb * mask[:, :, None] + (1 - mask[:, :, None]) * 77).astype(np.uint8)
                return_dict["rgb"] = self._process_rgb(masked_rgb)
            else:
                return_dict["rgb"] = self._process_rgb(rgb)
            if True or self.partition == "test":
                return_dict["raw_rgb"] = torch.from_numpy(
                    np.array(self.resize(Image.fromarray(rgb)))
                )

        if instruction is not None:
            return_dict["raw_instruction"] = instruction
            return_dict["instruction"] = self._process_instruction(instruction)

        if context is not None and self.process_context:
            return_dict.update(self._process_context(context, compute_rgb=rgb is not None))

        for k, val in kwargs.items():
            if "pick" in k or "place" in k:
                if val is not None:
                    assert scale_h == scale_w, "Acount for different scales"
                    if len(val.shape) == 1:
                        return_dict[k] = torch.FloatTensor(val / scale_w).unsqueeze(0)
                    else:
                        return_dict[k] = torch.FloatTensor(val / scale_w)
                elif self.partition == "train":
                    return_dict[k] = None
                else:
                    return_dict[k] = DUMMY

        if False or self.partition == "train":
            if self.cfg.spatial_augment:
                return_dict = self._spatial_augmentation(return_dict)

            for k, val in kwargs.items():
                if "pick" in k or "place" in k:
                    if f"{k}_heatmap" not in return_dict:
                        if val is not None:
                            return_dict[f"{k}_heatmap"] = self._make_gaussmap(return_dict[k])
                        else:
                            return_dict[f"{k}_heatmap"] = self._make_gaussmap()
                            return_dict[k] = DUMMY
                    elif self.requires_graph and return_dict[k] is None:
                        return_dict[k] = DUMMY
        return return_dict

    def _spatial_augmentation(self, return_dict):
        done = False

        tmp_dict = {}
        angle, dx, dy = None, None, None
        for _ in range(self.cfg.spatial_augmentations.max_augmentation_trials):
            angle = np.random.uniform(*self.cfg.spatial_augmentations.rotate_augmentation)
            dx = np.random.uniform(*self.cfg.spatial_augmentations.translate_augmentation)
            dy = np.random.uniform(*self.cfg.spatial_augmentations.translate_augmentation)
            try:
                for k, val in return_dict.items():
                    if ("pick" in k or "place" in k) and "heatmap" not in k:
                        if val is not None:
                            tmp_dict[k] = self._aug_pixel(
                                val.clone(),
                                -angle,
                                dx,
                                dy,
                                size=self.image_size - 1,
                            )
                done = True
                break
            except AssertionError:
                # Out of frame. Try again
                pass

        if done:
            for k, val in tmp_dict.items():
                return_dict[k] = val

            for k, val in return_dict.items():
                if "rgb" in k or "depth" in k:
                    assert angle is not None and dx is not None and dy is not None
                    return_dict[k] = v2.functional.affine(
                        val,
                        angle=angle,
                        translate=[dx, dy],
                        scale=1.0,
                        shear=[0.0],
                    )
        return return_dict

    def _aug_pixel(self, pixel, angle, dx, dy, size):
        rad = np.deg2rad(angle)
        R = torch.from_numpy(
            np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        ).float()
        pixel -= size / 2
        pixel = pixel @ R.T
        pixel += size / 2
        pixel[:, 0] += dx
        pixel[:, 1] += dy
        assert torch.all(pixel >= 0) and torch.all(pixel < size)
        return pixel

    def create_graph(self, depth_ori, mask, matrix_world_to_camera, K):
        world_coordinates = self.get_world_coords_from_pixels(depth_ori, matrix_world_to_camera, K)
        world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
        pointcloud = world_coords[mask.flatten() > 0].astype(np.float32)
        vox_pc = voxelize_pointcloud(pointcloud, self.voxel_size)
        sampled_pc = fps(vox_pc, self.num_nodes).astype(np.float32)

        normalized_vox_pc = sampled_pc - np.mean(sampled_pc, axis=0)
        node_attr = torch.from_numpy(normalized_vox_pc)
        edges, edge_attr = compute_edge_attr(normalized_vox_pc, self.neighbor_radius)
        graph_data = self.torch_geometric_data.from_dict(
            {"x": node_attr, "edge_index": edges, "edge_attr": edge_attr}
        )
        return graph_data, sampled_pc

    def get_pick_graph_heatmap(self, pick_pixel, sampled_pc, depth_ori, matrix_world_to_camera, K):
        if pick_pixel is not None:
            if len(pick_pixel.shape) > 1:
                # Take first pick pixel if there is more than one
                pick_pixel = pick_pixel[0]
            pick_pos = self.get_world_coord_from_pixel(
                pick_pixel, depth_ori, matrix_world_to_camera, K
            )
            distances = ((pick_pos - sampled_pc) ** 2).sum(axis=1)
            pick_point = torch.tensor(distances == np.min(distances)).float()
        else:
            pick_point = torch.zeros(len(sampled_pc))
        return pick_point

    @staticmethod
    def get_world_coord_from_pixel(pixel, depth, matrix_world_to_camera, K):
        matrix_camera_to_world = np.linalg.inv(matrix_world_to_camera)
        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        u, v = pixel[0], pixel[1]
        z = depth[int(np.rint(u)), int(np.rint(v))]
        x = (u - u0) * z / fx
        y = (v - v0) * z / fy

        cam_coord = np.ones(4)
        cam_coord[:3] = (x, y, z)
        world_coord = matrix_camera_to_world @ cam_coord

        return world_coord[:3]

    def _make_gaussmap(self, points=None):
        xy_grid = np.arange(self.image_size)
        x, y = np.meshgrid(xy_grid, xy_grid)
        if points is None:
            gauss_map = np.zeros_like(x, dtype=np.float64)
        else:
            points = points.numpy()
            strategy = self.cfg.strategy if len(points) > 1 else "first"
            if strategy == "first":
                center_x = round(points[0, 0])
                center_y = round(points[0, 1])
                dist = (x - center_x) ** 2 + (y - center_y) ** 2
                gauss_map = np.exp(-dist / (2 * self.cfg.sigma * self.cfg.sigma))
            elif strategy == "gmm":
                gauss_map = np.zeros_like(x, dtype=np.float64)
                for center_x, center_y in np.round(points):
                    dist = (x - center_x) ** 2 + (y - center_y) ** 2
                    gauss_map += np.exp(-dist / (2 * self.cfg.sigma**2)) / (
                        (2 * np.pi) * self.cfg.sigma**2
                    )
                assert gauss_map.max() != 0, f"Gauss map for points {points} is not valid"
                gauss_map /= gauss_map.max()
            elif strategy == "fit":
                # Covariance in general is too small as points are concentrated
                # Maybe scale it up
                mean = np.mean(points, axis=0)
                covariance_matrix = np.cov(points, rowvar=False)
                bivariate_normal_dist = multivariate_normal(mean=mean, cov=covariance_matrix)
                pos = np.dstack((x, y))  # stack the grids to create (x, y) coordinates

                # Evaluate the bivariate normal distribution at each point on the grid
                gauss_map = bivariate_normal_dist.pdf(pos)
            else:
                raise ValueError(f"Strategy {strategy} not recognized")
        return torch.FloatTensor(gauss_map)

    @staticmethod
    def get_world_coords_from_pixels(depth, matrix_world_to_camera, K):
        height, width = depth.shape
        # Apply back-projection: K_inv @ pixels * depth
        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        x = np.linspace(0, width - 1, width).astype(float)
        y = np.linspace(0, height - 1, height).astype(float)
        u, v = np.meshgrid(x, y)
        one = np.ones((height, width, 1))
        x = (u - u0) * depth / fx
        y = (v - v0) * depth / fy
        z = depth
        cam_coords = np.dstack([x, y, z, one])

        # convert the camera coordinate back to the world coordinate using the rotation
        # and translation matrix
        cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
        world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
        world_coords = world_coords.transpose().reshape((height, width, 4))

        return world_coords

    @staticmethod
    def get_pixel_from_world_coords(coord, matrix_world_to_camera, K):
        world_coordinate = np.concatenate([coord, np.ones((len(coord), 1))], axis=1)
        camera_coordinate = matrix_world_to_camera @ world_coordinate.T
        camera_coordinate = camera_coordinate.T

        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        x, y, depth = (
            camera_coordinate[:, 0],
            camera_coordinate[:, 1],
            camera_coordinate[:, 2],
        )
        u = x * fx / depth + u0
        v = y * fy / depth + v0

        pixel = np.array([u, v])

        return pixel
