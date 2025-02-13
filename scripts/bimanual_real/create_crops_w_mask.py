import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, required=True)
    parser.add_argument("--margin", type=int, default=10)
    args = parser.parse_args()

    for category in tqdm(os.listdir(args.path_to_data), desc="Creating crops"):
        if category != "empty":
            min_x = {}
            max_x = {}
            min_y = {}
            max_y = {}
            for image_file in os.listdir(os.path.join(args.path_to_data, category, "rgb")):
                cloth_id, *_, instruction, _ = image_file.split("_")

                mask = cv2.imread(os.path.join(args.path_to_data, category, "mask", image_file))
                if cloth_id not in min_x:
                    min_x[cloth_id] = mask.shape[0]
                    max_x[cloth_id] = 0
                    min_y[cloth_id] = mask.shape[1]
                    max_y[cloth_id] = 0

                x, y, _ = np.where(mask != 0)
                min_x[cloth_id] = min(min_x[cloth_id], x.min())
                max_x[cloth_id] = max(max_x[cloth_id], x.max())
                min_y[cloth_id] = min(min_y[cloth_id], y.min())
                max_y[cloth_id] = max(max_y[cloth_id], y.max())

            for image_file in os.listdir(os.path.join(args.path_to_data, category, "rgb")):
                cloth_id, *_, instruction, _ = image_file.split("_")

                side_half = args.margin + np.ceil(
                    max(max_x[cloth_id] - min_x[cloth_id], max_y[cloth_id] - min_y[cloth_id]) / 2
                ).astype(int)

                mid_x = np.ceil((min_x[cloth_id] + max_x[cloth_id]) / 2).astype(int)
                mid_y = np.ceil((min_y[cloth_id] + max_y[cloth_id]) / 2).astype(int)

                for modality in ["mask", "rgb", "depth", "raw_rgb", "raw_depth"]:
                    if "raw" not in modality:
                        path_to_file = os.path.join(
                            args.path_to_data, category, modality, image_file
                        )
                    else:
                        path_to_file = os.path.join(
                            args.path_to_data,
                            category,
                            modality,
                            image_file.replace(".png", ".npy"),
                        )

                    if os.path.isfile(path_to_file):
                        os.makedirs(
                            os.path.join(args.path_to_data, category, f"cropped_{modality}"),
                            exist_ok=True,
                        )
                        if "raw" not in modality:
                            img = cv2.imread(path_to_file)
                        else:
                            img = np.load(path_to_file)

                        pad_min_x = -min(0, mid_x - side_half)
                        pad_max_x = max(img.shape[0], mid_x + side_half + pad_min_x) - img.shape[0]

                        pad_min_y = -min(0, mid_y - side_half)
                        pad_max_y = max(img.shape[1], mid_y + side_half + pad_min_y) - img.shape[1]

                        if len(img.shape) == 2:
                            img = np.pad(img, ((pad_min_x, pad_max_x), (pad_min_y, pad_max_y)))
                        elif len(img.shape) == 3:
                            img = np.pad(
                                img, ((pad_min_x, pad_max_x), (pad_min_y, pad_max_y), (0, 0))
                            )
                        else:
                            raise ValueError

                        img = img[
                            mid_x - side_half + pad_min_x : mid_x + side_half + pad_min_x,
                            mid_y - side_half + pad_min_y : mid_y + side_half + pad_min_y,
                        ]

                        assert img.shape[0] == img.shape[1], f"{image_file} - {modality}"

                        if "raw" not in modality:
                            cv2.imwrite(
                                os.path.join(
                                    args.path_to_data, category, "cropped_" + modality, image_file
                                ),
                                img,
                            )
                        else:
                            np.save(
                                os.path.join(
                                    args.path_to_data,
                                    category,
                                    "cropped_" + modality,
                                    image_file.replace(".png", ".npy"),
                                ),
                                img,
                            )
