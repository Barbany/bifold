import pickle

from bifold.env.softgym_utils import get_matrix_world_to_camera

from . import BaseDataset
from .utils import DENG_CAMERA_PARAMS, get_mask_from_depth


class SingleDatasetSequential(BaseDataset):
    def __init__(self, cfg, *args, **kwargs):
        self.max_context_length = cfg.max_context_length

        super().__init__(cfg, *args, **kwargs, max_context_length=self.max_context_length)

        # Use same dataset for train and test
        # load dataset
        with open(self.dataset_path, "rb") as f:
            data = pickle.load(f)

        self.episodes = data["episodes"]

        self.img_size = None

        self.num_events = 0
        self.event_data = []
        for num_episode, episode in enumerate(self.episodes):
            for num_event, depth in enumerate(episode["depth"]):
                if self.img_size is None:
                    self.img_size = depth.shape[0]

                self.event_data.append({
                    "episode": num_episode,
                    "index": num_event,
                    "context": list(range(num_event)),
                })
                assert num_event - 1 <= self.max_context_length, (
                    f"The context of the dataset exceeds {self.max_context_length} "
                    f"for episode {num_episode}"
                )
                self.num_events += 1

    def __len__(self):
        return self.num_events

    def __getitem__(self, event_index):
        event_data = self.event_data[event_index]
        episode = self.episodes[event_data["episode"]]

        depth = episode["depth"][event_data["index"]] / self.depth_scale
        mask = get_mask_from_depth(depth)

        pick_pixel = episode["pick"][event_data["index"]]
        place_pixel = episode["place"][event_data["index"]]

        context = [
            {
                "rgb": episode["rgbs"][idx],
                "depth": episode["depth"][idx] / self.depth_scale,
                "mask": get_mask_from_depth(episode["depth"][idx] / self.depth_scale),
            }
            for idx in event_data["context"]
        ]

        return self.processor(
            rgb=episode["rgbs"][event_data["index"]],
            depth=depth,
            mask=mask,
            instruction=episode["instruction"][event_data["index"]],
            matrix_world_to_camera=get_matrix_world_to_camera(DENG_CAMERA_PARAMS),
            pick=pick_pixel,
            place=place_pixel,
            context=context,
        )
