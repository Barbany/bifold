from typing import Dict, Optional

import numpy as np
from scipy.stats import ecdf
from torchmetrics.classification import BinaryJaccardIndex

from bifold.env import Action


class Metrics:
    def __init__(self, cfg):
        self.best_eval = None
        self.tracked_metric = cfg.tracked_metric
        self.metrics = {
            metric_name: self.get_by_name(metric_name) for metric_name in cfg.computed_metrics
        }

    @staticmethod
    def get_by_name(metric_name, *args, **kwargs):
        if metric_name == "kp_mse":
            return KeypointMSE()
        elif metric_name.startswith("ap_"):
            threshold = int(metric_name.split("ap_")[-1])
            return AveragePrecision(threshold)
        elif metric_name == "iou":
            return IoU()
        elif metric_name == "quantile_prob":
            return QuantileProb()
        else:
            raise ValueError(f"Metric {metric_name} not recognized")

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def summary(self):
        has_improved = False
        metric_dict = {}
        for metric_name, metric in self.metrics.items():
            metric_value = metric.summary()
            metric_dict[metric_name] = metric_value
            if metric_name == self.tracked_metric:
                if metric.is_better(old_value=self.best_eval, new_value=metric_value):
                    self.best_eval = metric_value
                    has_improved = True
        return has_improved, metric_dict

    def __call__(self, *args, **kwargs):
        for metric in self.metrics.values():
            metric(*args, **kwargs)


class BaseMetric:
    def __init__(self, *args, **kwargs):
        self.values = []

    def __call__(self, action: Action, sample, **kwargs):
        raise NotImplementedError

    @staticmethod
    def is_better(old_value, new_value):
        # By default, best means lower
        if old_value is None:
            return True
        elif new_value < old_value:
            return True

    def reset(self):
        self.values = []

    def summary(self):
        # By default return the average
        return float(np.array(self.values).mean())


class IoU(BaseMetric):
    def __init__(self):
        super().__init__()
        self.metric = BinaryJaccardIndex()

    def __call__(self, sample, raw_output, **kwargs):
        if "mask_heatmap" in raw_output:
            if self.metric.device != raw_output["mask_heatmap"].device:
                self.metric.to(raw_output["mask_heatmap"])
            self.values.append(
                100 * self.metric(raw_output["mask_heatmap"], sample["mask"].squeeze(1)).item()
            )
        else:
            pass

    def summary(self):
        if self.values:
            return super().summary()
        else:
            return np.nan

    @staticmethod
    def is_better(old_value, new_value):
        # By default, best means lower
        if old_value is None:
            return True
        elif new_value > old_value:
            return True


class KeypointMSE(BaseMetric):
    def __call__(self, action: Action, sample, **kwargs):
        total_loss = 0
        n = 0
        for k, pred in action.__dict__.items():
            target = sample[k].cpu().numpy()
            if len(target.shape) == 3:
                valid = np.min(target, axis=(1, 2)) > 0
                # Multiple keypoints
                batch_loss = np.linalg.norm(
                    target[valid].round() - pred[valid, None, :], axis=-1
                ).min(axis=1)
            else:
                valid = np.min(target, axis=1) > 0
                batch_loss = np.linalg.norm(target[valid].round() - pred[valid], axis=-1)

            total_loss += batch_loss.mean()
            n += valid.sum()
        loss = total_loss / n if n != 0 else 0
        self.values.append(loss)


class QuantileProb(BaseMetric):
    def __call__(self, action: Action, sample, raw_output: Optional[Dict] = None, **kwargs):
        total_prob = 0
        n = 0
        assert raw_output is not None
        for k in action.__dict__.keys():
            if len(raw_output[k + "_heatmap"].shape) > 2:
                target = sample[k].cpu().numpy()
                if len(target.shape) == 3:
                    # Multiple keypoints
                    valid = np.min(target, axis=(1, 2)) > 0
                else:
                    valid = np.min(target, axis=1) > 0
                for i, v in enumerate(valid):
                    heatmap = raw_output[k + "_heatmap"][i].detach().cpu().numpy()
                    center_x = np.round(target[i][:, 0].astype(int))
                    center_y = np.round(target[i][:, 1]).astype(int)
                    cdf = ecdf(heatmap.flatten()).cdf
                    quantiles_idx = np.where(
                        heatmap[center_y, center_x][None, :] == cdf.quantiles[:, None]
                    )[0]
                    if v:
                        total_prob += cdf.probabilities[quantiles_idx].mean()
                    else:
                        total_prob += 1 - cdf.probabilities[quantiles_idx].mean()
                    n += 1
            else:
                target = sample[k + "_heatmap"].cpu().numpy()
                valid = np.max(target, axis=1) > 0
                for i, v in enumerate(valid):
                    heatmap = raw_output[k + "_heatmap"][i].detach().cpu().numpy()
                    target_idx = target[i].argmax()
                    cdf = ecdf(heatmap).cdf
                    quantiles_idx = np.where(heatmap[target_idx] == cdf.quantiles)[0]
                    if v:
                        total_prob += cdf.probabilities[quantiles_idx].mean()
                    else:
                        total_prob += 1 - cdf.probabilities[quantiles_idx].mean()
                    n += 1
        prob = total_prob / n if n != 0 else 0
        self.values.append(prob * 100)

    @staticmethod
    def is_better(old_value, new_value):
        # By default, best means lower
        if old_value is None:
            return True
        elif new_value > old_value:
            return True


class AveragePrecision(BaseMetric):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def __call__(self, action: Action, sample, **kwargs):
        total_precision = 0
        n = 0
        for k, pred in action.__dict__.items():
            target = sample[k].cpu().numpy()
            if len(target.shape) == 3:
                valid = np.min(target, axis=(1, 2)) > 0
                # Multiple keypoints
                distances = np.linalg.norm(
                    target[valid].round() - pred[valid, None, :], axis=-1
                ).min(axis=1)
            else:
                valid = np.min(target, axis=1) > 0
                distances = np.linalg.norm(target[valid].round() - pred[valid], axis=-1)
            # Precision = 1 if valid and within distance or if invalid and predicted invalid
            total_precision += (distances < self.threshold).sum() + (
                pred[~valid].min(axis=1) < 0
            ).sum()
            n += len(pred)
            assert total_precision <= n
        precision = (total_precision / n) * 100
        self.values.append(precision)

    @staticmethod
    def is_better(old_value, new_value):
        # The higher the better
        if old_value is None:
            return True
        elif new_value > old_value:
            return True
