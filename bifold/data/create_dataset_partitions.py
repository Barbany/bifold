import argparse
import os
import random

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from bifold.data.vr_folding_utils import create_groups_df


def get_frame(action):
    frame = None
    left_idx, right_idx = None, None

    if isinstance(action["left_start_idx"], str):
        left_idx = int(action["left_start_idx"].split("_")[-1])
    else:
        frame = action["right_start_idx"]
    if isinstance(action["right_start_idx"], str):
        right_idx = int(action["right_start_idx"].split("_")[-1])
    else:
        frame = action["left_start_idx"]

    if frame is None:
        assert left_idx is not None and right_idx is not None
        if left_idx <= right_idx:
            frame = action["left_start_idx"]
        else:
            frame = action["right_start_idx"]
    return frame


def create_context(frames):
    grouped = {}
    for frame in frames:
        *prefix, idx = frame.split("_")
        prefix = "_".join(prefix)
        if prefix not in grouped:
            grouped[prefix] = [idx]
        else:
            grouped[prefix].append(idx)

    for k in grouped.keys():
        grouped[k] = sorted(grouped[k])

    context = []
    for frame in frames:
        *prefix, idx = frame.split("_")
        prefix = "_".join(prefix)
        seq_idx = grouped[prefix].index(idx)
        context.append([prefix + "_" + ctx_idx for ctx_idx in grouped[prefix][:seq_idx]])
    return context


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    categories = os.listdir(os.path.join(args.actions_path, os.pardir, "vr_folding_dataset.zarr"))

    dfs = {}

    for category in tqdm(categories, desc="Categories"):
        df_file = os.path.join(args.actions_path, category + "_actions.csv")
        if os.path.isfile(df_file):
            dfs[category] = pd.read_csv(df_file)
        else:
            root = zarr.open(
                os.path.join(args.actions_path, os.pardir, "vr_folding_dataset.zarr", category),
                mode="r",
            )
            dfs[category] = create_groups_df(root["samples"])
            dfs[category].to_csv(df_file)

    train_indices = {}
    test_indices = {}

    removed_actions = 0
    total_actions = 0
    for category in categories:
        n = len(dfs[category])
        total_actions += n
        if args.remove_bad_sequences:
            indices = (~dfs[category]["bad_sequence"]).nonzero()[0]
            removed_actions += n - len(indices)
        else:
            indices = list(range(n))
        random.shuffle(indices)
        split_idx = int(len(indices) * args.train_portion)
        train_indices[category] = indices[:split_idx]
        test_indices[category] = indices[split_idx:]

        frames = [get_frame(action) for _, action in dfs[category].iterrows()]
        assert len(frames) == n
        dfs[category].insert(len(dfs[category].columns), "context", create_context(frames))

    train_df = pd.concat(
        [dfs[category].iloc[train_indices[category]] for category in categories],
        ignore_index=True,
    )
    train_df.to_csv(os.path.join(args.actions_path, "train.csv"))

    test_df = pd.concat(
        [dfs[category].iloc[test_indices[category]] for category in categories],
        ignore_index=True,
    )
    test_df.to_csv(os.path.join(args.actions_path, "test.csv"))

    print(
        f"Filtered out {removed_actions} actions out of {total_actions}, "
        f"i.e., {removed_actions / total_actions * 100:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actions_path",
        type=str,
        required=True,
    )
    parser.add_argument("--train_portion", type=float, default=0.9)
    parser.add_argument(
        "--remove_bad_sequences", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.actions_path, exist_ok=True)
    main(args)
