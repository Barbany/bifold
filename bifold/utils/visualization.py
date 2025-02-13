import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def save_predictions(out_folder, out_file_name, rgb=None, colormap="viridis", **kwargs):
    cm = plt.get_cmap(colormap)

    if rgb is not None:
        folder = os.path.join(out_folder, "rgb")
        os.makedirs(folder, exist_ok=True)
        rgb_img = Image.fromarray(rgb)
        rgb_img.save(os.path.join(folder, out_file_name))
    else:
        rgb_img = None

    for k, val in kwargs.items():
        if val is not None:
            folder = os.path.join(out_folder, k)
            os.makedirs(folder, exist_ok=True)
            if "heatmap" in k or k == "depth":
                if "heatmap" in k:
                    val = val.squeeze().cpu().numpy()

                if len(val.shape) > 1:
                    heatmap = Image.fromarray((cm(val)[:, :, :3] * 255).astype(np.uint8))
                    if rgb_img is not None and "heatmap" in k:
                        blended = Image.blend(rgb_img, heatmap, alpha=0.3)
                        blended.save(os.path.join(folder, out_file_name))
                    else:
                        heatmap.save(os.path.join(folder, out_file_name))
            elif k == "particle_pos":
                np.save(
                    file=os.path.join(folder, out_file_name.replace(".png", ".npy")),
                    arr=val,
                )
            elif k == "viz":
                Image.fromarray(val).save(os.path.join(folder, out_file_name))
            elif k == "rgb_gt":
                Image.fromarray(val).save(os.path.join(folder, out_file_name))
            else:
                raise ValueError(f"Unrecognized argument {k}")


def visualize_action(sample, action):
    gt_colors = [(255, 0, 0), (0, 255, 0)]
    pred_colors = [(0, 0, 255), (0, 255, 255)]

    images = []

    if len(sample["raw_rgb"].shape) == 4:
        raw_rgb = sample["raw_rgb"]
        batched = True
    else:
        assert len(sample["raw_rgb"].shape) == 3
        raw_rgb = [sample["raw_rgb"]]
        batched = False

    for i, img in enumerate(raw_rgb):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if len(action.__dict__) == 2:
            if "pick" in sample and "place" in sample:
                img = _pick_place_viz(
                    img,
                    sample["pick"][i].cpu().numpy() if batched else sample["pick"],
                    sample["place"][i].cpu().numpy() if batched else sample["place"],
                    color=gt_colors[0],
                )
            img = _pick_place_viz(
                img,
                [action.__dict__["pick"][i]],
                [action.__dict__["place"][i]],
                color=pred_colors[0],
            )
        elif len(action.__dict__) == 4:
            for arm, gt_color, pred_color in zip(["left", "right"], gt_colors, pred_colors):
                if f"{arm}_pick" in sample:
                    pick = (
                        sample[f"{arm}_pick"][i].cpu().numpy() if batched else sample[f"{arm}_pick"]
                    )
                    place = (
                        sample[f"{arm}_place"][i].cpu().numpy()
                        if batched
                        else sample[f"{arm}_place"]
                    )
                    img = _pick_place_viz(
                        img,
                        pick,
                        place,
                        color=gt_color,
                    )
                img = _pick_place_viz(
                    img,
                    action.__dict__[f"{arm}_pick"][i],
                    action.__dict__[f"{arm}_place"][i],
                    color=pred_color,
                )
        else:
            raise ValueError(f"Invalid action {action} with {len(action.__dict__)} items")
        images.append(img)
    return images


def _pick_place_viz(img, picks, places, color):
    if not isinstance(picks, list) and len(picks.shape) == 1:
        picks = [picks]
        places = [places]
    for pick, place in zip(picks, places):
        if pick[0] >= 0:
            cv2.circle(
                img=img,
                center=(round(pick[0]), round(pick[1])),
                radius=3,
                color=color,
                thickness=2,
            )
        if place[0] >= 0:
            cv2.arrowedLine(
                img=img,
                pt1=(round(pick[0]), round(pick[1])),
                pt2=(round(place[0]), round(place[1])),
                color=color,
                thickness=2,
            )
    return img
