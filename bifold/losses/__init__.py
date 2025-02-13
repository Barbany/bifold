from torch import nn
from torch.nn import functional as F


class Losses:
    @staticmethod
    def get_by_name(cfg=None, name=None, *args, **kwargs):
        if cfg is not None:
            name = cfg.name
        if name == "bce_gaussmap":
            Loss = BCEGaussMap
        elif name == "bce_mask":
            Loss = BCEMask
        elif name == "composed":
            Loss = ComposedLoss
        elif name == "dice":
            Loss = DiceLoss
        elif name == "focal":
            Loss = FocalLoss
        else:
            assert cfg is not None
            raise ValueError(f"Loss {cfg.name} not recognized")
        if cfg is not None:
            del cfg.name
            return Loss(*args, **cfg, **kwargs)
        else:
            return Loss(*args, **kwargs)


class ComposedLoss:
    def __init__(self, loss_names, weights, **kwargs):
        assert len(loss_names) == len(weights)
        self.loss_dict = {
            loss_name: Losses.get_by_name(name=loss_name, **kwargs) for loss_name in loss_names
        }
        self.weights = {loss_name: weight for loss_name, weight in zip(loss_names, weights)}

    def __call__(self, output, sample):
        loss = None
        intermediate_losses = {}
        for loss_name, loss_fn in self.loss_dict.items():
            curr_loss, curr_intermediate_losses = loss_fn(output, sample)
            if loss is None:
                loss = curr_loss * self.weights[loss_name]
            else:
                loss += curr_loss * self.weights[loss_name]

            intermediate_losses[loss_name] = curr_loss
            for k, val in curr_intermediate_losses.items():
                intermediate_losses[loss_name + " " + k] = val
        return loss, intermediate_losses


class BCEGaussMap:
    def __init__(
        self,
        is_bimanual,
        mask_pick_heatmap,
        **kwargs,
    ):
        self.loss_fn = nn.BCELoss()
        self.is_bimanual = is_bimanual
        self.mask_pick_heatmap = mask_pick_heatmap

    def __call__(self, output, sample):
        if self.is_bimanual:
            return self._bimanual_loss(output, sample)
        else:
            return self._single_loss(output, sample)

    def _bimanual_loss(self, output, sample):
        intermediate_losses = {}
        loss = None
        for arm in ["left", "right"]:
            for action in ["pick", "place"]:
                if action == "pick" and self.mask_pick_heatmap:
                    target = sample[f"{arm}_{action}_heatmap"] * sample["mask"].squeeze(1)
                else:
                    target = sample[f"{arm}_{action}_heatmap"]
                curr_loss = self.loss_fn(
                    input=output[f"{arm}_{action}_heatmap"],
                    target=target,
                )
                intermediate_losses[f"{arm}_{action}"] = curr_loss
                if loss is None:
                    loss = curr_loss
                else:
                    loss = loss + curr_loss
        return loss, intermediate_losses

    def _single_loss(self, output, sample):
        intermediate_losses = {}
        loss = None
        for action in ["pick", "place"]:
            if action == "pick" and self.mask_pick_heatmap:
                target = sample[f"{action}_heatmap"] * sample["mask"].squeeze(1)
            else:
                target = sample[f"{action}_heatmap"]

            curr_loss = self.loss_fn(
                input=output[f"{action}_heatmap"],
                target=target,
            )
            intermediate_losses[f"{action}"] = curr_loss
            if loss is None:
                loss = curr_loss
            else:
                loss = loss + curr_loss
        return loss, intermediate_losses


class BCEMask:
    def __init__(self, **kwargs):
        self.loss_fn = nn.BCELoss()

    def __call__(self, output, sample):
        return self.loss_fn(output["mask_heatmap"], sample["mask"].squeeze(1)), {}


"""
Focal loss and dice loss used in "End-to-End Object Detection with Transformers"
Same loss combination is used in "Segment Anything" with a 20:1 ratio
"""


class DiceLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, output, sample):
        inputs = output["mask_heatmap"].flatten(1)
        targets = sample["mask"].flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum(), {}


class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2, **kwargs):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, output, sample):
        prob = output["mask_heatmap"]
        targets = sample["mask"].squeeze(1)
        ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum(), {}
