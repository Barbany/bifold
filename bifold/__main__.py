import os
import random
from typing import Dict, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2
from tqdm import tqdm

from .data import Datasets
from .losses import Losses
from .metrics import Metrics
from .models import Models
from .optim import Optimizers, Schedulers
from .utils.visualization import save_predictions, visualize_action


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(os.getcwd())
    with open("config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = Trainer(cfg)

    if not cfg.eval_only:
        trainer.prepare_train()
        trainer.train()
    trainer.eval()


class Trainer:
    def __init__(self, cfg):
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not cfg.use_cpu
            else torch.device("cpu")
        )

        if cfg.use_wandb and not cfg.eval_only and not cfg.debug:
            self.writer = wandb.init(
                project="bifold",
                group=cfg.train_dataset.name,
                name="+".join(HydraConfig.get().overrides.task),
                resume="allow",
            )
            wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        else:
            self.writer = None

        self.cfg = cfg

        self.seed_randomness()

        self.model = Models.get_by_name(self.cfg.model, device=self.device).to(self.device)

        self.train_dataloader, self.test_dataloader, self.input_processor = (
            Datasets.get_dataloaders(self.cfg)
        )
        self.metrics = Metrics(self.cfg.metrics)

    def train(self):
        if self.start_epoch < self.cfg.epochs:
            for epoch in tqdm(list(range(self.start_epoch, self.cfg.epochs)), desc="Train"):
                self.train_epoch(epoch)
                if self.cfg.eval_epochs and (epoch + 1) % self.cfg.eval_epochs == 0:
                    has_improved, _ = self.eval_epoch(epoch)
                    if has_improved:
                        self.save_model(epoch, is_best=True)
                if self.cfg.save_epochs and (epoch + 1) % self.cfg.save_epochs == 0:
                    self.save_model(epoch)

            epoch = self.cfg.epochs - 1
            self.save_model(epoch)

    def eval(self):
        self.load_model(load_best=self.cfg.load_best)
        eval_file = f"eval_{self.cfg.test_dataset.name}.yaml"
        _, metric_dict = self.eval_epoch()
        for k, val in metric_dict.items():
            if isinstance(val, dict):
                for sub_k, sub_val in val.items():
                    print(f"{k} {sub_k}:\t{sub_val:.2f}")
            else:
                print(f"{k}:\t{val:.2f}")

        if os.path.isfile(eval_file):
            print("Found YAML file")
            with open(eval_file, "r") as f:
                old_results = yaml.load(f, Loader=yaml.Loader)
            for k, val in old_results.items():
                if k not in metric_dict:
                    metric_dict[k] = val
                else:
                    if val != metric_dict[k]:
                        print(f"Old value for {k} = {val} ; New value {metric_dict[k]}")
        with open(eval_file, "w") as f:
            yaml.dump(metric_dict, f)

    def seed_randomness(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def prepare_train(self):
        params_non_frozen = filter(lambda p: p.requires_grad, self.model.parameters())
        self.loss_fn = Losses.get_by_name(cfg=self.cfg.loss)
        self.optimizer = Optimizers.get_by_name(cfg=self.cfg.optim, params=params_non_frozen)
        assert self.train_dataloader is not None
        self.scheduler = Schedulers.get_by_name(
            cfg=self.cfg.scheduler,
            optimizer=self.optimizer,
            max_iters=len(self.train_dataloader) * self.cfg.epochs,
        )
        self.load_model()

    def train_epoch(self, epoch=None):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_dataloader)
        for i, sample in enumerate(pbar):
            sample = self.move_sample_to_device(sample)

            # Visualize inputs to model
            if self.cfg.debug and self.cfg.visualize_model_inputs:
                self.visualize_model_inputs(sample)

            output = self.model(sample)

            loss, intermediate_losses = self.loss_fn(output, sample)

            self.optimizer.zero_grad()
            assert loss is not None
            loss.backward()

            if self.cfg.debug:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is None:
                        raise ValueError(f"Parameter {name} might not have gradient attached!")

            if self.cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.writer:
                self.writer.log({"loss": loss, "epoch": epoch})
                for k, val in intermediate_losses.items():
                    self.writer.log({k: val})
                for j, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.log({f"lr_{j}": param_group["lr"]})

            assert loss is not None
            total_loss += loss.item()
            pbar.set_description("Training loss:{}".format(total_loss / (i + 1)))

    def eval_epoch(self, epoch=None) -> Tuple[Optional[bool], Dict[str, torch.Tensor]]:
        self.model.eval()
        has_improved = None
        if epoch is not None or self.cfg.simulator is None:
            # During training or in testing if simulator is not specified
            has_improved, metric_dict = self.eval_epoch_pixel(epoch)
        elif self.cfg.simulator == "softgym":
            if self.cfg.test_dataset.is_bimanual:
                metric_dict = self.eval_epoch_softgym_bimanual(epoch)
            else:
                metric_dict = self.eval_epoch_softgym_single(epoch)
        else:
            raise ValueError(f"Simulator {self.cfg.simulator} not supported")

        if self.writer:
            for k, val in metric_dict.items():
                self.writer.log({k: val})

        if epoch is not None:
            assert (
                has_improved is not None
            ), "has_improved boolean must be specified during training"
        return has_improved, metric_dict

    def eval_epoch_pixel(self, epoch):
        self.metrics.reset()
        num_samples = 0
        for sample in tqdm(self.test_dataloader, desc="Evaluate"):
            sample = self.move_sample_to_device(sample)

            # Pick and place prediction
            ret_get_action = self.model.get_action(sample, return_raw_output=True)
            assert isinstance(ret_get_action, Tuple)
            action, raw_output = ret_get_action

            if any("pick" in k for k in sample.keys()):
                # If samples are available
                self.metrics(action=action, sample=sample, raw_output=raw_output)

            if self.cfg.visualize_predictions:
                out_folder = (
                    os.path.join(
                        "eval_background",
                        self.cfg.test_dataset.name,
                        "pixel_metrics",
                        "best" if self.cfg.load_best else "last",
                    )
                    if epoch is None
                    else os.path.join("eval", "pixel_metrics", f"epoch_{epoch}")
                )
                visualizations = visualize_action(sample, action)
                for j in range(len(visualizations)):
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
                    num_samples += 1

        has_improved, metric_dict = self.metrics.summary()
        if self.writer:
            for k, val in metric_dict.items():
                self.writer.log({k: val})

        return has_improved, metric_dict

    def eval_epoch_softgym_single(self, epoch):
        from .env.softgym_evaluator import SoftgymSingleEvaluator

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

        return evaluator.summary()

    def eval_epoch_softgym_bimanual(self, epoch):
        from .env.softgym_evaluator import SoftgymBimanualEvaluator

        evaluator = SoftgymBimanualEvaluator(
            self.cfg, model=self.model, processor=self.input_processor
        )
        for i, sample in enumerate(tqdm(self.test_dataloader, desc="Evaluating with SoftGym")):
            evaluator.evaluate(sample)
        evaluator.close()

        return evaluator.summary()

    def load_model(self, load_best=False):
        if load_best and os.path.isfile(os.path.join("checkpoints", "best.pth")):
            checkpoint_file = os.path.join("checkpoints", "best.pth")
        elif os.path.isfile(os.path.join("checkpoints", "last.pth")):
            checkpoint_file = os.path.join("checkpoints", "last.pth")
        else:
            if self.cfg.eval_only and not self.cfg.debug:
                raise FileNotFoundError("Cannot evaluate with untrained model")
            self.start_epoch = 0
            return

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.start_epoch = checkpoint["epoch"]
        random.setstate(checkpoint["random_states"][0])
        np.random.set_state(checkpoint["random_states"][1])
        torch.set_rng_state(checkpoint["random_states"][2].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["random_states"][3].cpu())

        self.model.load_state_dict(checkpoint["model"])

        if not self.cfg.eval_only:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            if self.scheduler is not None and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            if "best_eval" in checkpoint:
                self.metrics.best_eval = checkpoint["best_eval"]

        print(f"Loaded model from checkpoint {checkpoint_file}")

    def save_model(self, epoch, is_best=False):
        os.makedirs("checkpoints", exist_ok=True)
        if is_best:
            model_path = os.path.join("checkpoints", "best.pth")
        else:
            model_path = os.path.join("checkpoints", "last.pth")
        state_dict = {
            "epoch": epoch + 1,
            "random_states": (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
            ),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self.metrics.best_eval is not None:
            state_dict["best_eval"] = self.metrics.best_eval
        torch.save(state_dict, model_path)

    def move_sample_to_device(self, sample):
        # Move all inputs to device
        for k, val in sample.items():
            if isinstance(val, torch.Tensor) or k == "graph":
                sample[k] = val.to(self.device)
        return sample

    def visualize_model_inputs(self, sample):
        input_visualizations = {}
        if self.writer is None:
            fig, axes = plt.subplots(ncols=4, nrows=3)
            flat_axes = axes.flat
            i = 0
        else:
            fig, flat_axes, i = None, None, None

        for k, val in sample.items():
            if any(subk in k for subk in ["rgb", "depth", "heatmap"]):
                if "rgb" in k:
                    transform = v2.Compose([
                        v2.Normalize(
                            (0.0, 0.0, 0.0),
                            (
                                1 / 0.26862954,
                                1 / 0.26130258,
                                1 / 0.27577711,
                            ),
                        ),
                        v2.Normalize(
                            (-0.48145466, -0.4578275, -0.40821073),
                            (1.0, 1.0, 1.0),
                        ),
                        v2.ToPILImage(),
                    ])
                else:
                    transform = v2.ToPILImage()

                if len(val.shape) > 2:
                    if len(val.shape) < 5:
                        imgs = [transform(val[0])]
                        names = [k]
                    else:
                        imgs = [transform(v) for v in val[0]]
                        names = [f"{k}_{i}" for i in range(len(imgs))]

                    for img, name in zip(imgs, names):
                        if self.writer:
                            input_visualizations[name] = wandb.Image(
                                img,
                                caption=sample["raw_instruction"][0],
                            )
                        else:
                            assert flat_axes is not None
                            assert i is not None
                            flat_axes[i].imshow(img)
                            if k == "depth":
                                for k_in, val_in in sample.items():
                                    if "heatmap" in k_in and len(val_in[0].shape) > 1:
                                        flat_axes[i].imshow(transform(val_in[0]), alpha=0.5)
                            flat_axes[i].set_title(name)
                            i += 1

        if self.writer:
            self.writer.log(input_visualizations)
        else:
            assert fig is not None
            fig.suptitle(sample["raw_instruction"][0])
            plt.show()


if __name__ == "__main__":
    main()
