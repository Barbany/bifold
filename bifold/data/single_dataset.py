import pickle

from bifold.env.softgym_utils import get_matrix_world_to_camera, intrinsic_from_fov

from . import BaseDataset
from .utils import DENG_CAMERA_PARAMS, get_mask_from_depth


class SingleDataset(BaseDataset):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Use same dataset for train and test
        # load dataset
        with open(self.dataset_path, "rb") as f:
            data = pickle.load(f)

        self.rgbs = data["rgbs"]
        self.depths = data["depth"]
        self.pick_pixels = data["pick"]
        self.place_pixels = data["place"]
        self.instructions = data["instruction"]

        self.img_size = self.depths[0].shape[0]

        assert (
            len(self.rgbs)
            == len(self.depths)
            == len(self.pick_pixels)
            == len(self.place_pixels)
            == len(self.instructions)
        )

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        depth = self.depths[index] / self.depth_scale
        mask = get_mask_from_depth(depth)

        pick_pixel = self.pick_pixels[index]
        place_pixel = self.place_pixels[index]

        return self.processor(
            rgb=self.rgbs[index],
            depth=depth,
            mask=mask,
            instruction=self.instructions[index],
            matrix_world_to_camera=get_matrix_world_to_camera(DENG_CAMERA_PARAMS),
            pick=pick_pixel,
            place=place_pixel,
            K=intrinsic_from_fov(
                height=DENG_CAMERA_PARAMS["default_camera"]["height"],
                width=DENG_CAMERA_PARAMS["default_camera"]["width"],
                fov=45,
            ),
        )
