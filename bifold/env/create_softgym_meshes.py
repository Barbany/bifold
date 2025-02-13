import os
import pickle as pkl
from copy import deepcopy
from shutil import rmtree

import numpy as np
import pandas as pd
import pyflex
import zarr
from bifold.data.utils import parse_list_string
from bifold.data.vr_folding_utils import create_textured_obj
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def center_object():
    """
    Center the object to be at the origin
    NOTE: call a pyflex.set_positions and then pyflex.step
    """
    pos = pyflex.get_positions().reshape(-1, 4)
    pos[:, [0, 2]] -= np.mean(pos[:, [0, 2]], axis=0, keepdims=True)
    pyflex.set_positions(pos.flatten())
    pyflex.step()


def vectorized_range(start, end):
    """Return an array of NxD, iterating from the start to the end"""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)[:, None] / N + start[:, None]).astype("int")
    return idxes


def vectorized_meshgrid(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y


def get_current_covered_area(cloth_particle_radius=0.00625, pos=None):
    if pos is None:
        pos = pyflex.get_positions()
    pos = np.reshape(pos, [-1, 4])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.0
    pos2d = pos[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(
        np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0
    )
    slotted_x_high = np.minimum(
        np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100
    )
    slotted_y_low = np.maximum(
        np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0
    )
    slotted_y_high = np.minimum(
        np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100
    )
    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    return np.sum(grid) * span[0] * span[1]


def set_cloth3d_scene(config, state=None):
    render_mode = 2
    camera_params = config["camera_params"][config["camera_name"]]
    env_idx = 6
    scene_params = np.concatenate([
        config["pos"][:],
        [config["scale"], config["rot"]],
        config["vel"][:],
        [config["stiff"], config["mass"], config["radius"]],
        camera_params["pos"][:],
        camera_params["angle"][:],
        [camera_params["width"], camera_params["height"]],
        [render_mode],
        [config["cloth_type"]],
        [config["cloth_index"]],
    ])

    pyflex.set_scene(env_idx, scene_params, 0)
    initial_pos = pyflex.get_positions().reshape(-1, 4)[:, :3].copy()
    rotate_particles([180, 0, 90])
    move_to_pos([0, 0.05, 0])
    for _ in range(50):
        pyflex.step()

    if state is not None:
        set_state(state)

    return initial_pos


def set_state(state_dict):
    pyflex.set_positions(state_dict["particle_pos"])
    pyflex.set_velocities(state_dict["particle_vel"])
    pyflex.set_shape_states(state_dict["shape_pos"])
    pyflex.set_phases(state_dict["phase"])
    camera_params = deepcopy(state_dict["camera_params"])
    update_camera(camera_params, "default_camera")


def update_camera(camera_params, camera_name="default_camera"):
    camera_param = camera_params[camera_name]
    pyflex.set_camera_params(
        np.array([
            *camera_param["pos"],
            *camera_param["angle"],
            camera_param["width"],
            camera_param["height"],
        ])
    )


def rotate_particles(angle):
    r = R.from_euler("zyx", angle, degrees=True)
    pos = pyflex.get_positions().reshape(-1, 4)
    center = np.mean(pos, axis=0)
    pos -= center
    new_pos = pos.copy()[:, :3]
    new_pos = r.apply(new_pos)
    new_pos = np.column_stack([new_pos, pos[:, 3]])
    new_pos += center
    pyflex.set_positions(new_pos)


def move_to_pos(new_pos):
    pos = pyflex.get_positions().reshape(-1, 4)
    center = np.mean(pos, axis=0)
    pos[:, :3] -= center[:3]
    pos[:, :3] += np.asarray(new_pos)
    pyflex.set_positions(pos)


def get_cloth3d_default_config():
    cam_pos, cam_angle = np.array([0, 1.0, 0]), np.array([0 * np.pi, -90 / 180.0 * np.pi, 0])
    config = {
        "pos": [0, 0, 0],
        "scale": -1,
        "rot": 0,
        "vel": [0.0, 0.0, 0.0],
        "stiff": 1.0,
        "mass": 0.5 / (40 * 40),
        "radius": 0.00625,
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "pos": cam_pos,
                "angle": cam_angle,
                "width": 720,
                "height": 720,
            }
        },
        "cloth_type": None,
        "cloth_index": None,
    }
    return config


dataset_root = "/home/obarbany/bifold/datasets/folding"
cloth3d_root = "/data/cloth3d"
softgym_cache = "/home/obarbany/bifold/configs"

# Read the CSV file
columns_with_lists = [
    "left_grip_from",
    "left_grip_to",
    "right_grip_from",
    "right_grip_to",
]  # Specify the columns containing lists
converters = {col: parse_list_string for col in columns_with_lists}
test_file = pd.read_csv(
    os.path.join(dataset_root, "sequential_actions", "test.csv"),
    converters=converters,
    index_col=0,
)
textured_meshes = os.path.join(dataset_root, "textured_meshes")

out_path = os.path.join(os.environ["CLOTH3D_PATH"], "Bimanual")
if os.path.isdir(out_path):
    rmtree(out_path)
os.makedirs(out_path)

# Modify
# https://github.com/dengyh16code/language_deformable/blob/main/PyFlex/bindings/softgym_scenes/softgym_cloth3d.h#L346
# and recompile pyflex to include the cloth type
pyflex.init(True, True, 720, 720)
cloth_type = 2
max_meshes = 9999
idx = 0

# Scale set empirically by comparing meshes in CLOTH3D_PATH and textured_meshes
first_scale = 0.3

# Softgym parameters
max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this

backup = {"configs": {}, "state": {}, "keypoints": {}}

softgym_configs = {}
softgym_states = {}
softgym_keypoints = {}
zarr_dataset = {}


processed_frames = set()
for _, row in tqdm(test_file.iterrows(), total=len(test_file)):
    frame = None
    left_idx, right_idx = None, None

    if isinstance(row["left_start_idx"], str):
        left_idx = int(row["left_start_idx"].split("_")[-1])
        cloth3d_id, category, *_ = row["left_start_idx"].split("_")
    else:
        frame = row["right_start_idx"]
        cloth3d_id, category, *_ = row["right_start_idx"].split("_")
    if isinstance(row["right_start_idx"], str):
        right_idx = int(row["right_start_idx"].split("_")[-1])
    else:
        frame = row["left_start_idx"]

    if frame is None:
        assert left_idx is not None and right_idx is not None
        if left_idx <= right_idx:
            frame = row["left_start_idx"]
        else:
            frame = row["right_start_idx"]

    if category not in zarr_dataset:
        zarr_dataset[category] = zarr.open(
            os.path.join(dataset_root, "vr_folding_dataset.zarr", category), "r"
        )

    for idx_fr, fr in enumerate([frame] + eval(row["context"])):
        processed_frames.add(fr)
        mesh_file = os.path.join(textured_meshes, category, fr + ".obj")
        if not os.path.isfile(mesh_file):
            os.makedirs(os.path.join(textured_meshes, category), exist_ok=True)
            create_textured_obj(
                cloth3d_textured_file=os.path.join(
                    cloth3d_root,
                    "train",
                    cloth3d_id,
                    "textured_" + category + ".obj",
                ),
                mesh_file=mesh_file,
                sample_group=zarr_dataset[category]["samples"][fr],
            )

        pre_vertex_data = []
        vertex_data = []
        post_vertex_data = []
        # Do not use open3d/trimesh/... to avoid getting rid of duplicates
        pre_vertex = True
        with open(mesh_file, "r") as f:
            for line in f.readlines():
                if line.startswith("v "):
                    pre_vertex = False
                    vertex_data.append([float(v) for v in line.split("v ")[-1].rstrip().split()])
                elif pre_vertex:
                    pre_vertex_data.append(line)
                else:
                    post_vertex_data.append(line)

        raw_vertices = np.array(vertex_data) / first_scale

        x_displ = raw_vertices[:, 0].mean()
        table_height = raw_vertices[:, 1].min()
        y_displ = raw_vertices[:, 2].mean()

        vertices = raw_vertices[:, [0, 2, 1]]

        vertices[:, 0] -= x_displ
        vertices[:, 1] -= y_displ
        vertices[:, -1] -= table_height

        config = get_cloth3d_default_config()
        config["cloth_type"] = cloth_type
        config["cloth_index"] = idx

        with open(os.path.join(out_path, f"{idx:04d}.obj"), "w") as f:
            f.writelines(pre_vertex_data)
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            f.writelines(post_vertex_data)

        softgym_configs[fr] = deepcopy(config)

        print(f"Mesh with {len(vertices)} vertices")
        # Save state
        # Update camera parameters
        update_camera(config["camera_params"], config["camera_name"])

        initial_pos = set_cloth3d_scene(config)

        if initial_pos.shape[0] != vertices.shape[0]:
            # Find correspondences as Pyflex changes the number of vertices
            # Get faces
            faces = []
            for line in post_vertex_data:
                if line.startswith("f "):
                    faces.append([
                        int(idx.split("/")[0]) - 1 for idx in line.split("f ")[-1].rstrip().split()
                    ])

            # Pyflex transformations
            avg_edge_len = 0
            for idx0, idx1, idx2 in faces:
                v0 = vertices[idx0]
                v1 = vertices[idx1]
                v2 = vertices[idx2]
                avg_edge_len += (
                    np.linalg.norm(v0 - v1) + np.linalg.norm(v1 - v2) + np.linalg.norm(v2 - v0)
                )
            avg_edge_len /= 3 * len(faces)
            scale = config["radius"] / avg_edge_len

            mesh_lower = vertices.min(axis=0)

            pyflex_mesh = vertices.copy()
            pyflex_mesh -= mesh_lower
            pyflex_mesh *= scale

            distances = cdist(pyflex_mesh, initial_pos)
            softgym_indices = distances.argmin(axis=-1)
        else:
            softgym_indices = np.arange(len(vertices))

        if idx_fr == 0:
            softgym_keypoints[fr] = {
                "left_pick_idx": (
                    softgym_indices[row["left_grip_from"][0]]
                    if row["left_grip_from"] is not None
                    else None
                ),
                "left_place_idx": (
                    softgym_indices[
                        np.linalg.norm(
                            zarr_dataset[category]["samples"][row["left_start_idx"]]["mesh"][
                                "cloth_verts"
                            ][:]
                            - zarr_dataset[category]["samples"][row["left_end_idx"]]["mesh"][
                                "cloth_verts"
                            ][row["left_grip_to"][0]],
                            axis=-1,
                        ).argmin(axis=0)
                    ]
                    if row["left_grip_to"] is not None
                    else None
                ),
                "right_pick_idx": (
                    softgym_indices[row["right_grip_from"][0]]
                    if row["right_grip_from"] is not None
                    else None
                ),
                "right_place_idx": (
                    softgym_indices[
                        np.linalg.norm(
                            zarr_dataset[category]["samples"][row["right_start_idx"]]["mesh"][
                                "cloth_verts"
                            ]
                            - zarr_dataset[category]["samples"][row["right_end_idx"]]["mesh"][
                                "cloth_verts"
                            ][row["right_grip_to"][0]],
                            axis=-1,
                        ).argmin(axis=0)
                    ]
                    if row["right_grip_to"] is not None
                    else None
                ),
            }

        if fr not in backup["configs"]:

            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.all(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()

            max_area = get_current_covered_area()
            pos = pyflex.get_positions()

            print(f"#vertices in pyflex positions is {len(pos) / 4}\n\n")
            vel = pyflex.get_velocities()
            shape_pos = pyflex.get_shape_states()
            phase = pyflex.get_phases()
            camera_params = deepcopy(config["camera_params"])
            state = {
                "particle_pos": pos,
                "particle_vel": vel,
                "shape_pos": shape_pos,
                "phase": phase,
                "camera_params": camera_params,
            }
            state["max_area"] = max_area

            softgym_states[fr] = deepcopy(state)

            # Store keypoints

            assert idx <= max_meshes
            idx += 1
        else:
            softgym_configs[fr] = deepcopy(backup["configs"][fr])
            softgym_states[fr] = deepcopy(backup["states"][fr])
            softgym_keypoints[fr] = deepcopy(backup["keypoints"][fr])


with open(os.path.join(softgym_cache, "bimanual_extended.pkl"), "wb") as f:
    pkl.dump(
        {
            "configs": softgym_configs,
            "states": softgym_states,
            "keypoints": softgym_keypoints,
        },
        f,
        protocol=pkl.HIGHEST_PROTOCOL,
    )
