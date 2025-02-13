import argparse
import os
import pickle
import random

import imageio
import numpy as np

Done = np.array([0, 0])


def create_dataset(root, tasks, save_path, use_rgb, n_demos):
    episodes = []
    total_num = 0
    seen_num = 0

    if "All" in tasks:
        tasks = os.listdir(root)
        print("Load All Tasks: ", tasks)

    trajs = [
        os.path.join(root, task, traj)
        for task in tasks
        for traj in os.listdir(os.path.join(root, task))
    ]

    random.shuffle(trajs)

    each_task_num = {task: 0 for task in tasks}
    for traj in trajs:
        task = traj.split(os.path.sep)[-2]
        depths = []
        picks = []
        places = []
        instructions = []
        success = []
        primitives = []
        rgbs = []
        if each_task_num[task] < n_demos:
            # actions & instructions
            with open(os.path.join(traj, "info.pkl"), "rb") as f:
                data = pickle.load(f)
                pick_pixels = data["pick"]
                place_pixels = data["place"]
                langs = data["instruction"]
                prims = data["primitive"]
                unseens = data["unseen_flags"]
            num_actions = len(pick_pixels)
            total_num += num_actions
            each_task_num[task] += 1
            depth_path = os.path.join(traj, "depth")
            rgb_path = os.path.join(traj, "rgb")

            i = 0
            while i < num_actions:
                unseen = unseens[i]
                if not unseen:
                    seen_num += 1
                    # insert actions & instructions
                    picks.append(pick_pixels[i])
                    places.append(place_pixels[i])
                    instructions.append(langs[i])
                    primitives.append(prims[i])
                    success.append(0)

                    # observations
                    depths.append(imageio.imread(os.path.join(depth_path, str(i) + ".png")))
                    if use_rgb:
                        rgbs.append(imageio.imread(os.path.join(rgb_path, str(i) + ".png")))

                i = i + 1  # next step

        if depths:
            assert len(depths) == len(picks) == len(places) == len(instructions) == len(success)
            episodes.append({
                "depth": depths,
                "pick": picks,
                "place": places,
                "instruction": instructions,
                "success": success,
                "primitive": primitives,
            })
            if use_rgb:
                episodes[-1]["rgbs"] = rgbs

    print("build {} seen tasks from {} tasks".format(seen_num, total_num))

    print(each_task_num)
    # save

    with open(save_path, "wb+") as f:
        pickle.dump({"episodes": episodes}, f)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--tasks", type=str, help="choose single task / all task(all)")
    parser.add_argument("--use_rgb", action="store_true", help="choose with inst feature or not")
    parser.add_argument("--root", type=str, default="raw_data")
    parser.add_argument("--save_path_root", type=str, default="data_sequential")
    parser.add_argument("--n_demos", type=int, default=100, help="num of demos")
    args = parser.parse_args()

    os.makedirs(args.save_path_root, exist_ok=True)
    if args.tasks == "All":
        save_path = os.path.join(args.save_path_root, f"All_{args.n_demos}.pkl")
    else:
        save_path = os.path.join(args.save_path_root, args.tasks + ".pkl")

    tasks = [args.tasks]
    create_dataset(args.root, tasks, save_path, args.use_rgb, args.n_demos)
