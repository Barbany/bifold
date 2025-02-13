import argparse
import os

import pandas as pd
from render import render_sample
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_root_path", type=str, required=True)
    parser.add_argument(
        "--obj_root_path", type=str, required=True, desc="Path to textured .obj meshes"
    )
    parser.add_argument("--renders_root_path", type=str, required=True)
    parser.add_argument("--cloth3d_root_path", type=str, required=True)
    parser.add_argument("--context_frames", type=int, default=5)
    parser.add_argument("--render_context", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    for action_file in tqdm(os.listdir(args.action_root_path)):
        df = pd.read_csv(os.path.join(args.action_root_path, action_file))
        for idx, action in tqdm(df.iterrows(), total=len(df), leave=False, desc=action_file):
            frame = None

            if isinstance(action["left_start_idx"], str):
                left_idx = int(action["left_start_idx"].split("_")[-1])
            else:
                frame = action["right_start_idx"]
            if isinstance(action["right_start_idx"], str):
                right_idx = int(action["right_start_idx"].split("_")[-1])
            else:
                frame = action["left_start_idx"]

            if frame is None:
                if left_idx <= right_idx:
                    frame = action["left_start_idx"]
                else:
                    frame = action["right_start_idx"]

            category = frame.split("_")[1]

            frames = [frame]
            if args.render_context:
                *prefix, str_idx = frame.split("_")
                # Temporal indices in garment tracking dataset
                # are integers with a step size of 5, i.e., 0, 5, 10, 15, ...
                pick_idx = int(str_idx) // 5
                for idx in range(max(pick_idx - 3, 0), pick_idx + args.context_frames):
                    frames.append("_".join([*prefix, f"{idx * 5:06d}"]))

            for frame in tqdm(frames, desc="Neighboring frames", leave=False):
                if not os.path.isfile(
                    os.path.join(args.renders_root_path, category, "colors", frame + ".png")
                ):
                    render_sample(
                        group_name=frame,
                        obj_root_path=args.obj_root_path,
                        renders_root_path=args.renders_root_path,
                        cloth3d_root_path=args.cloth3d_root_path,
                        verbose=False,
                    )
