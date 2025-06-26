import os
import random
from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ..data import Datasets
from ..metrics import Metrics
from ..models import Models
from .visualization import (
    save_predictions,
    visualize_action,
    visualize_attention,
    visualize_features,
)


@hydra.main(version_base=None, config_path="../conf", config_name="visualization")
def main(cfg: DictConfig):
    print(os.getcwd())

    visualizer = Visualizer(cfg)
    visualizer.visualize_pixel()


class Visualizer:
    def __init__(self, cfg):
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not cfg.use_cpu
            else torch.device("cpu")
        )

        self.cfg = cfg

        self.seed_randomness()

        self.model = Models.get_by_name(
            self.cfg.model, device=self.device, return_attentions=True
        ).to(self.device)

        if self.cfg.load_pretrained:
            self.load_model(load_best=self.cfg.load_best)

        self.model.eval()

        self.train_dataloader, self.test_dataloader, self.input_processor = (
            Datasets.get_dataloaders(self.cfg, shuffle_test=False)
        )
        self.metrics = Metrics(self.cfg.metrics)

    def seed_randomness(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    @torch.inference_mode()
    def visualize_pixel(self):
        self.metrics.reset()
        num_samples = 0
        for sample in tqdm(self.test_dataloader, desc="Evaluate"):
            if num_samples > 1 or num_samples + self.cfg.batch_size < 1:
                num_samples += self.cfg.batch_size
                continue

            sample = self.move_sample_to_device(sample)

            # Pick and place prediction
            ret_get_action = self.model.get_action(
                sample,
                return_raw_output=True,
                output_attentions=self.cfg.visualize_attention,
                output_features=self.cfg.visualize_features,
            )
            assert isinstance(ret_get_action, Tuple)
            action, raw_output = ret_get_action

            if any("pick" in k for k in sample.keys()):
                # If samples are available
                self.metrics(action=action, sample=sample, raw_output=raw_output)

            if not self.cfg.load_pretrained:
                del raw_output["attn_weights"]
                del raw_output["out_features"]

            out_folder = os.path.join(
                "icra_workshop",
                self.cfg.test_dataset.name + "_pt" if self.cfg.load_pretrained else "scratch_",
                "pixel_metrics",
            )

            visualizations = visualize_action(sample, action)
            for j in range(len(visualizations)):
                if self.cfg.visualize_actions and self.cfg.load_pretrained:
                    kwargs = {
                        "rgb": sample["raw_rgb"][j].cpu().numpy(),
                        "depth": sample["depth"][j].squeeze().cpu().numpy(),
                        "viz": visualizations[j],
                    }
                    if raw_output is not None:
                        for k, val in raw_output.items():
                            if "heatmap" in k:
                                kwargs[k] = val[j]
                    save_predictions(
                        out_folder=out_folder,
                        out_file_name=f"{num_samples}_{sample['raw_instruction'][j]}.png",
                        **kwargs,
                    )

                if self.cfg.visualize_attention:
                    visualize_attention(
                        sample_num=j,
                        inputs=sample,
                        raw_output=raw_output,
                        out_folder=out_folder,
                        num_samples=num_samples,
                        patch_size=self.cfg.model.patch_size,
                    )

                if self.cfg.visualize_features:
                    visualize_features(
                        sample_num=j,
                        inputs=sample,
                        raw_output=raw_output,
                        out_folder=out_folder,
                        num_samples=num_samples,
                        patch_size=self.cfg.model.patch_size,
                    )

                num_samples += 1

        if num_samples >= self.cfg.n_visualized_samples:
            return

    def visualize_softgym(self):
        self.model.eval()
        if self.cfg.test_dataset.is_bimanual:
            self.visualize_softgym_bimanual()
        else:
            self.visualize_softgym_single()

    def visualize_softgym_single(self):
        from ..env.softgym_evaluator import SoftgymSingleEvaluator

        evaluator = SoftgymSingleEvaluator(
            cfg=self.cfg,
            model=self.model,
            processor=self.input_processor,
        )
        for task in [
            "CornerFold",
            "TriangleFold",
            "StraightFold",
            "TshirtFold",
            "TrousersFold",
        ]:
            evaluator.evaluate(num_evals=self.cfg.num_evals, task=task)
        evaluator.close()
        raise NotImplementedError

    def visualize_softgym_bimanual(self):
        from ..env.softgym_evaluator import SoftgymBimanualEvaluator

        evaluator = SoftgymBimanualEvaluator(
            self.cfg, model=self.model, processor=self.input_processor
        )
        for i, sample in enumerate(tqdm(self.test_dataloader, desc="Evaluating with SoftGym")):
            evaluator.evaluate(sample)
        evaluator.close()
        raise NotImplementedError

    def load_model(self, load_best=False):
        if load_best and os.path.isfile(os.path.join("checkpoints", "best.pth")):
            checkpoint_file = os.path.join("checkpoints", "best.pth")
        elif os.path.isfile(os.path.join("checkpoints", "last.pth")):
            checkpoint_file = os.path.join("checkpoints", "last.pth")
        else:
            raise FileNotFoundError(f"No checkpoint found in {os.path.abspath('checkpoints')}")

        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.start_epoch = checkpoint["epoch"]
        random.setstate(checkpoint["random_states"][0])
        np.random.set_state(checkpoint["random_states"][1])
        torch.set_rng_state(checkpoint["random_states"][2].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["random_states"][3].cpu())

        self.model.load_state_dict(checkpoint["model"])

        print(f"Loaded model from checkpoint {os.path.abspath(checkpoint_file)}")

    def move_sample_to_device(self, sample):
        # Move all inputs to device
        for k, val in sample.items():
            if isinstance(val, torch.Tensor) or k == "graph":
                sample[k] = val.to(self.device)
        return sample


if __name__ == "__main__":
    main()
