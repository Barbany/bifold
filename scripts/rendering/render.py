import os
import subprocess


def render_sample(group_name, obj_root_path, renders_root_path, cloth3d_root_path, verbose=True):
    category = group_name.split("_")[1]
    return_code = subprocess.call(
        [
            "blenderproc",
            "run",
            "render_view_blenderproc.py",
            "--obj_file",
            os.path.join(obj_root_path, category, group_name + ".obj"),
            "--renders_root_path",
            renders_root_path,
            "--cloth3d_root_path",
            cloth3d_root_path,
        ],
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.STDOUT,
    )
    if return_code != 0:
        raise RuntimeError(f"Failed with {group_name}")
