import numpy as np
import open3d as o3d
import torch


class TruncatedDepthStandardization:
    def __init__(self, thresh=0.1):
        self.thresh = thresh

    def __call__(self, depth):
        trunc_depth = torch.sort(depth.reshape(-1), dim=0)[0]
        trunc_depth = trunc_depth[
            int(self.thresh * trunc_depth.shape[0]) : int(
                (1 - self.thresh) * trunc_depth.shape[0]
            )
        ]
        return (depth - trunc_depth.mean()) / torch.sqrt(trunc_depth.var() + 1e-6)


class DepthNoise:
    def __init__(self):
        data = o3d.data.RedwoodIndoorLivingRoom1()
        noise_model_path = data.noise_model_path
        self.simulator = o3d.t.io.DepthNoiseSimulator(noise_model_path)

    def __call__(self, sample):
        noisy_depth = self.simulator.simulate(
            o3d.t.geometry.Image(sample["depth"].astype(np.float32)), depth_scale=1
        )
        sample["depth"] = np.asarray(noisy_depth).squeeze()
        return sample


class DepthScale:
    def __init__(self, min_shift, max_shift):
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, sample):
        shift = np.random.uniform(self.min_shift, self.max_shift)
        sample["depth"] = sample["depth"] + shift
        return sample


class MaskDepth:
    def __call__(self, sample):
        if sample["mask"] is not None:
            depth = sample["depth"] * sample["mask"]
        else:
            depth = sample["depth"]
        return depth[:, :, None]
