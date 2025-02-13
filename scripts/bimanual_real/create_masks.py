import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def get_input_point(image_file: str) -> np.ndarray:
    """Get the input point for the given image file.
    Can specify the input point for each image file based on the cloth_id and instruction.

    Args:
        image_file (str): The image file name.

    Returns:
        np.ndarray: The input points for the given image, with shape (N, 2), where N is the number
            of points use to prompt SAM. The points have been manually selected.
    """
    cloth_id, *_, instruction, _ = image_file.split("_")
    cloth_id = int(cloth_id)
    instruction = int(instruction)

    # TODO: Update the input points for each new image files or use GroundedSAM or similar
    # text-based methods to automatically generate the input points.
    if cloth_id == 4 and instruction == 0:
        point = [[650, 375], [550, 300], [350, 400]]
    if cloth_id == 4 and instruction == 1:
        point = [[650, 375], [550, 300], [350, 400], [600, 500]]
    elif cloth_id == 7 and instruction == 2:
        point = [[400, 500]]
    elif cloth_id == 2 and instruction == 0:
        point = [[500, 450], [500, 250]]
    elif cloth_id == 2 and instruction == 2:
        point = [[700, 500]]
    elif cloth_id == 3 and instruction == 0:
        point = [[700, 500], [700, 250]]
    elif cloth_id == 3 and instruction == 1:
        point = [[600, 250]]
    elif cloth_id == 0 and instruction == 1:
        point = [[800, 500], [800, 300]]
    elif cloth_id in [0, 1] and instruction in [1, 2]:
        point = [[800, 500]]
    elif cloth_id == 3 and instruction == 2:
        point = [[600, 250], [550, 180], [550, 300]]
    elif cloth_id in [5, 6] and instruction == 2:
        point = [[500, 600]]
    else:
        point = [[500, 375]]
    return np.array(point)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--path_to_data", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="vit_h")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    for category in tqdm(os.listdir(args.path_to_data), desc="Creating masks"):
        os.makedirs(os.path.join(args.path_to_data, category, "mask"), exist_ok=True)
        os.makedirs(os.path.join(args.path_to_data, category, "mask_overlay"), exist_ok=True)
        for image_file in os.listdir(os.path.join(args.path_to_data, category, "rgb")):
            image = cv2.imread(os.path.join(args.path_to_data, category, "rgb", image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_point = get_input_point(image_file)
            if input_point is not None:
                predictor.set_image(image)

                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=np.ones(len(input_point)),
                    multimask_output=False,
                )
                cv2.imwrite(
                    os.path.join(args.path_to_data, category, "mask", image_file),
                    (255 * masks[0]).astype(np.uint8),
                )
                plt.imshow(image)
                plt.imshow(masks[0], alpha=0.5)
                marker_size = 375
                plt.scatter(
                    input_point[:, 0],
                    input_point[:, 1],
                    color="green",
                    marker="*",
                    s=marker_size,
                    edgecolor="white",
                    linewidth=1.25,
                )
                plt.savefig(os.path.join(args.path_to_data, category, "mask_overlay", image_file))
                plt.clf()
