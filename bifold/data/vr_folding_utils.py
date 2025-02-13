from random import choice
from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from tqdm import tqdm

# Prompts taken from
# https://github.com/dengyh16code/language_deformable/blob/main/Policy/demonstrator.py
# for folding sleeves and folds. Others obtained with ChatGPT
folding_actions = {
    "sleeves": [
        "Fold the {which} sleeve towards the inside.",
        "Inwardly fold the {which} sleeve.",
        "Fold the {which} sleeve towards the body.",
        "Bend the {which} sleeve towards the inside.",
        "Fold the {which} sleeve to the center.",
        "Fold the {which} sleeve towards the middle.",
        "Bring the {which} sleeve to the center.",
        "Fold the {which} sleeve inward to the halfway point.",
        "Tuck the {which} sleeve towards the center.",
        "Meet the {which} sleeve at the center.",
        "Fold the {which} sleeve to the midpoint.",
        "Center the {which} sleeve.",
        "Align the {which} sleeve to the center.",
        "Fold the {which} sleeve to the axis.",
        "Bring the {which} sleeve to the median.",
        "Fold the {which} sleeve to the central point.",
        "Fold the {which} sleeve towards the midpoint of the shirt.",
        "Bring the {which} sleeve to the center seam.",
        "Fold the {which} sleeve to the centerline of the shirt.",
        "Fold the {which} sleeve to the centerline of the shirt.",
    ],
    "refine": [
        "Fold the {which} part of the {garment} neatly.",
        "Align the {which} part of the {garment} properly.",
        "Arrange the {which} part of the {garment} neatly.",
        "Straighten out the {which} part of the {garment}.",
        "Place the {which} part of the {garment} in the correct position.",
        "Ensure the {which} part of the {garment} is well-positioned.",
    ],
    "fold": [
        "Fold the {garment} in half, {which1} to {which2}.",
        "Fold the {garment} from the {which1} side towards the {which2} side.",
        "Fold the {garment} in half, starting from the {which1} and ending at the {which2}.",
        "Fold the {garment}, {which1} side over {which2} side.",
        "Bend the {garment} in half, from {which1} to {which2}.",
        "Fold the {garment}, making sure the {which1} side touches the {which2} side.",
        "Fold the {garment}, bringing the {which1} side to meet the {which2} side.",
        "Crease the {garment} down the middle, from {which1} to {which2}.",
        "Fold the {garment} in half horizontally, {which1} to {which2}.",
        "Make a fold in the {garment}, starting from the {which1} and ending at the {which2}.",
        "Fold the {garment} in half, aligning the {which1} and {which2} sides.",
        "Fold the {garment}, ensuring the {which1} side meets the {which2} side.",
        "Fold the {garment}, orientating from the {which1} towards the {which2}.",
        "Fold the {garment} cleanly, from the {which1} side to the {which2} side.",
        "Fold the {garment} in half, with the {which1} side overlapping the {which2}.",
        "Create a fold in the {garment}, going from {which1} to {which2}.",
        "Bring the {which1} side of the {garment} towards the {which2} side and fold them in half.",
        "Fold the waistband of the {garment} in half, from {which1} to {which2}.",
        "Fold the {garment} neatly, from the {which1} side to the {which2} side.",
        "Fold the {garment}, making a crease from the {which1} to the {which2}.",
    ],
}

opposite_locations = {
    "bottom": "top",
    "top": "bottom",
    "right": "left",
    "left": "right",
}


def create_groups_df(samples_group):
    df = _get_groups_df(samples_group)
    actions = {
        "left_start_idx": [],
        "left_grip_from": [],
        "left_grip_to": [],
        "left_end_idx": [],
        "right_start_idx": [],
        "right_grip_from": [],
        "right_grip_to": [],
        "right_end_idx": [],
        "text": [],
        "bad_sequence": [],
        "info": [],
    }
    for instance_id, df_instance in tqdm(df.groupby("instance_id"), "Creating dataframe"):
        nocs_vertices = None
        faces = None
        pp_actions_l = []
        pp_actions_r = []

        curr_action_l = PPAction()
        curr_action_r = PPAction()

        prev_l = -1
        prev_r = -1
        prev_index = None

        categories = df_instance["garment_name"].unique()
        assert len(categories) == 1, "Non-unique garment name in the same instance ID"
        category = categories[0].lower()

        is_bad = {}
        for count, index in enumerate(df_instance.sort_values("sample_id").index):
            is_bad[index] = filter_bad_meshes(
                vertices=samples_group[index]["mesh"]["cloth_verts"][:],
                nocs_vertices=samples_group[index]["mesh"]["cloth_nocs_verts"][:],
                faces=samples_group[index]["mesh"]["cloth_faces_tri"],
            )
            left_grip = samples_group[index]["grip_vertex_id"]["left_grip_vertex_id"][:]
            right_grip = samples_group[index]["grip_vertex_id"]["right_grip_vertex_id"][:]
            if nocs_vertices is None:
                nocs_vertices = samples_group[index]["mesh"]["cloth_nocs_verts"][:]
                faces = samples_group[index]["mesh"]["cloth_faces_tri"][:]
            else:
                assert np.array_equal(
                    nocs_vertices,
                    samples_group[index]["mesh"]["cloth_nocs_verts"][:],
                )
                assert faces is not None
                assert np.array_equal(
                    faces,
                    samples_group[index]["mesh"]["cloth_faces_tri"][:],
                )
            if left_grip[0] != -1 and prev_l == -1:
                # Starting left action
                curr_action_l.start_idx = index
                curr_action_l.start_mesh = samples_group[index]["mesh"]["cloth_verts"][:]
                curr_action_l.vertex_trajectory.append(left_grip)
                curr_action_l.world_trajectory.append(
                    samples_group[index]["mesh"]["cloth_verts"][left_grip]
                )
                curr_action_l.counts.append(count)
            elif left_grip[0] == -1 and prev_l != -1:
                curr_action_l.end_idx = prev_index
                curr_action_l.end_mesh = samples_group[prev_index]["mesh"]["cloth_verts"][:]
                # Finishing left action
                pp_actions_l.append(curr_action_l)
                curr_action_l = PPAction()
            elif left_grip[0] != -1 and prev_l != -1:
                curr_action_l.vertex_trajectory.append(left_grip)
                curr_action_l.world_trajectory.append(
                    samples_group[index]["mesh"]["cloth_verts"][left_grip]
                )
                curr_action_l.counts.append(count)

            if right_grip[0] != -1 and prev_r == -1:
                # Starting right action
                curr_action_r.start_idx = index
                curr_action_r.start_mesh = samples_group[index]["mesh"]["cloth_verts"][:]
                curr_action_r.vertex_trajectory.append(right_grip)
                curr_action_r.world_trajectory.append(
                    samples_group[index]["mesh"]["cloth_verts"][right_grip]
                )
                curr_action_r.counts.append(count)
            elif right_grip[0] == -1 and prev_r != -1:
                # Finishing right action
                curr_action_r.end_idx = prev_index
                curr_action_r.end_mesh = samples_group[prev_index]["mesh"]["cloth_verts"][:]
                pp_actions_r.append(curr_action_r)
                curr_action_r = PPAction()
            elif right_grip[0] != -1 and prev_r != -1:
                curr_action_r.vertex_trajectory.append(right_grip)
                curr_action_r.world_trajectory.append(
                    samples_group[index]["mesh"]["cloth_verts"][right_grip]
                )
                curr_action_r.counts.append(count)

            prev_l = left_grip[0]
            prev_r = right_grip[0]
            prev_index = index

        try:
            add_actions_to_dataset(
                pp_actions_l=pp_actions_l,
                pp_actions_r=pp_actions_r,
                category=category,
                actions=actions,
                nocs_vertices=nocs_vertices,
                faces=faces,
                is_bad=is_bad,
            )
        except ValueError as e:
            # 03252_Tshirt_000403_* does a correct action followed by a
            # nonsense action and the sequence finishes without a successful fold
            print(f"Ignoring {instance_id} due to {e}.")

    groups_df = pd.DataFrame(data=actions)
    # Removed drop duplicates because it uses hashing and lists are unhashable
    # groups_df.drop_duplicates(inplace=True)

    return groups_df


class PPAction:
    def __init__(self):
        self.start_idx = None
        self.end_idx = None

        self.start_mesh = None
        self.end_mesh = None

        self.world_trajectory = []
        self.vertex_trajectory = []
        self.counts = []

    def __repr__(self):
        return f"Pick {self.start_idx} and place {self.end_idx}"


def _get_groups_df(samples_group):
    rows = {}
    for key, group in tqdm(samples_group.items(), desc="Get groups df"):
        rows[key] = group.attrs.asdict()
    groups_df = pd.DataFrame(data=list(rows.values()), index=list(rows.keys()))
    groups_df.drop_duplicates(inplace=True)
    groups_df["group_key"] = groups_df.index
    return groups_df


def visualize_action_nocs(nocs_vertices, faces, vertices):
    data: List[Any] = [
        go.Mesh3d(
            x=nocs_vertices[:, 0],
            y=nocs_vertices[:, 1],
            z=nocs_vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
        )
    ]
    for name, vertex in vertices.items():
        data.append(
            go.Scatter3d(
                x=[nocs_vertices[vertex, 0]],
                y=[nocs_vertices[vertex, 1]],
                z=[nocs_vertices[vertex, 2]],
                mode="markers",
                name=name,
                showlegend=True,
            )
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[0, 1],
            ),
            yaxis=dict(
                nticks=4,
                range=[0, 1],
            ),
            zaxis=dict(
                nticks=4,
                range=[0, 1],
            ),
        ),
    )
    fig.write_html("mesh_plot.html")


def filter_bad_meshes(
    vertices,
    nocs_vertices,
    faces,
    method="zscore",
    max_edge_length=0.07,
    max_edge_length_ratio=80,
    zscore_ratio=3.5,
):
    """
    Return True if bad

    The bad mesh 00156_Top_000013_000000 had a ratio of 185 and the
    good mesh 00001_Top_000000_000005 of 17

    Meshes that are wrong:
    00156_Top_000013_*
    00241_Top_000029_*
    00269_Top_000034_*
    and possibly many more
    """
    mesh = trimesh.Trimesh(vertices, faces)
    edge_lengths = np.linalg.norm(
        mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=-1
    )
    if method == "length":
        return not np.all(edge_lengths < max_edge_length)
    elif method == "ratio":
        return np.max(edge_lengths) / np.min(edge_lengths) >= max_edge_length_ratio
    elif method == "zscore":
        nocs_mesh = trimesh.Trimesh(nocs_vertices, faces)
        nocs_edge_lengths = np.linalg.norm(
            nocs_mesh.vertices[nocs_mesh.edges[:, 0]] - nocs_mesh.vertices[nocs_mesh.edges[:, 1]],
            axis=-1,
        )
        zscore = (np.max(edge_lengths) - np.mean(edge_lengths)) / np.std(edge_lengths)
        zscore_nocs = (np.max(nocs_edge_lengths) - np.mean(nocs_edge_lengths)) / np.std(
            nocs_edge_lengths
        )
        return zscore / zscore_nocs >= zscore_ratio
    else:
        raise NotImplementedError(f"Method {method} not recognized")


def aggregate_text(text1, text2):
    if text1 is None:
        return text2
    if text2 is None:
        return text1
    elif text1 == text2:
        return text1
    else:
        return None


def nocs_to_text(action, action_type, nocs_vertices, x_thresh=0.5, z_thresh=0.5):
    # z_thresh due to 01353_Tshirt_000176_000330
    if action is None:
        return None, None, None
    if action_type == "pick":
        # Pick location can be simply inferred using the NOCS coordinates of the picked vertices
        vertex_index = action.vertex_trajectory[0]
    else:
        # Place location has to be inferred using the end position in the world coordinates, finding
        # close vertices of the mesh at the action start time and using their NOCS coordinates
        vertex_index = np.linalg.norm(
            action.start_mesh[:, None, :] - action.world_trajectory[-1], axis=-1
        ).argmin(axis=0)
    # Majority counting
    return (
        (
            "left"
            if (nocs_vertices[vertex_index, 0] >= x_thresh).sum() > len(vertex_index) // 2
            else "right"
        ),
        (
            "top"
            if (nocs_vertices[vertex_index, -1] >= z_thresh).sum() > len(vertex_index) // 2
            else "bottom"
        ),
        vertex_index,
    )


def get_text_location(action_l, action_r, nocs_vertices, category, faces):
    # Can do that using feature distillation and cosine similarity with text features
    is_sleeve = False
    info = ""
    action_text = {}

    vertices = {}

    for action_type in ["pick", "place"]:
        l_lr, l_tb, l_vertex = nocs_to_text(action_l, action_type, nocs_vertices)
        r_lr, r_tb, r_vertex = nocs_to_text(action_r, action_type, nocs_vertices)

        vertices[action_type + "_left"] = l_vertex
        vertices[action_type + "_right"] = r_vertex
        lr = aggregate_text(l_lr, r_lr)
        tb = aggregate_text(l_tb, r_tb)
        if lr:
            if tb:
                # Agree on both left-right location and top-bottom location

                # First try to avoid same pick and place location
                if action_type == "place":
                    if action_text["pick"] == lr:
                        action_text[action_type] = tb
                    elif action_text["pick"] == tb:
                        action_text[action_type] = lr
                    # Then, select opposite if possible
                    elif action_text["pick"] == opposite_locations[lr]:
                        action_text[action_type] = lr
                    elif action_text["pick"] == opposite_locations[tb]:
                        action_text[action_type] = tb
                    else:
                        action_text[action_type] = tb + " " + lr
                else:
                    # If it is a T-shirt, consider it a sleeve
                    if category == "tshirt" and tb == "top":
                        action_text[action_type] = lr
                        is_sleeve = True
                        # Place action is not used
                        action_text["place"] = None
                        break
                    else:
                        action_text[action_type] = tb + " " + lr
            else:
                action_text[action_type] = lr
        else:
            if tb:
                action_text[action_type] = tb
            else:
                # Use heuristics: opposite of pick action
                info += "Using heuristics."
                # Pick is very clear, however place vertex indeces may be spotted at a
                # wrong location
                # e.g., when a sleeve is over the bottom part
                if action_type == "place":
                    if action_text["pick"] in opposite_locations:
                        action_text[action_type] = opposite_locations[action_text["pick"]]
                    else:
                        # Maybe it's an action with a composed location
                        # e.g. bottom left, top right, ...
                        action_text[action_type] = " ".join(
                            [opposite_locations[text] for text in action_text["pick"].split()]
                        )
                else:
                    visualize_action_nocs(nocs_vertices, faces, vertices)
                    raise ValueError(
                        f"Combination of NOCS pick and place coordinates not supported "
                        f"for {action_l} {action_r}"
                    )
    pick, place = action_text["pick"], action_text["place"]
    if pick == place:
        # Maybe it's all about improving the shape of a given side as in
        # 00087_Tshirt_000014_000295 -> 00087_Tshirt_000014_000320
        visualize_action_nocs(nocs_vertices, faces, vertices)
        info += "Same pick and place location."
    return pick, place, is_sleeve, info


def add_actions_to_dataset(
    pp_actions_l, pp_actions_r, category, actions, nocs_vertices, faces, is_bad
):
    pp_actions_l, pp_actions_r = clean_actions(pp_actions_l, pp_actions_r)
    for action_l, action_r in zip(pp_actions_l, pp_actions_r):
        bad_sequence = False
        pick, place, is_sleeve, info = get_text_location(
            action_l, action_r, nocs_vertices, category, faces
        )
        if action_l is None:
            one_arm = " only using the right arm."
        elif action_r is None:
            one_arm = " only using the left arm."
        else:
            one_arm = None
        if category == "tshirt" and is_sleeve:
            text = choice(folding_actions["sleeves"]).format(which=pick)
        elif pick == place:
            text = choice(folding_actions["refine"]).format(garment=category, which=pick)
        else:
            text = choice(folding_actions["fold"]).format(
                garment=category, which1=pick, which2=place
            )

        if one_arm is not None:
            # Append at the end considering the final stop
            text = text.replace(".", one_arm)

        if action_l is not None:
            actions["left_start_idx"].append(action_l.start_idx)
            actions["left_grip_from"].append(action_l.vertex_trajectory[0].tolist())
            actions["left_grip_to"].append(action_l.vertex_trajectory[-1].tolist())
            actions["left_end_idx"].append(action_l.end_idx)

            bad_sequence |= is_bad[action_l.start_idx]
            bad_sequence |= is_bad[action_l.end_idx]
        else:
            actions["left_start_idx"].append(None)
            actions["left_grip_from"].append(None)
            actions["left_grip_to"].append(None)
            actions["left_end_idx"].append(None)

        if action_r is not None:
            actions["right_start_idx"].append(action_r.start_idx)
            actions["right_grip_from"].append(action_r.vertex_trajectory[0].tolist())
            actions["right_grip_to"].append(action_r.vertex_trajectory[-1].tolist())
            actions["right_end_idx"].append(action_r.end_idx)

            bad_sequence |= is_bad[action_r.start_idx]
            bad_sequence |= is_bad[action_r.end_idx]
        else:
            actions["right_start_idx"].append(None)
            actions["right_grip_from"].append(None)
            actions["right_grip_to"].append(None)
            actions["right_end_idx"].append(None)

        actions["text"].append(text)
        actions["bad_sequence"].append(bad_sequence)
        actions["info"].append(info)


def clean_actions(
    pp_actions_l,
    pp_actions_r,
    fast_action_threshold=5,
    small_action_threshold=0.1,
):
    clean_actions_l = []
    clean_actions_r = []
    for action in pp_actions_l:
        if (
            len(action.counts) > fast_action_threshold
            and np.linalg.norm(action.world_trajectory[-1] - action.world_trajectory[0])
            > small_action_threshold
        ):
            clean_actions_l.append(action)

    for action in pp_actions_r:
        if (
            len(action.counts) > fast_action_threshold
            and np.linalg.norm(action.world_trajectory[-1] - action.world_trajectory[0])
            > small_action_threshold
        ):
            clean_actions_r.append(action)

    aligned_actions_l = []
    index_l = 0

    aligned_actions_r = []
    index_r = 0
    while index_l < len(clean_actions_l) and index_r < len(clean_actions_r):
        if index_l >= len(clean_actions_l):
            # No more clean actions for left gripper
            aligned_actions_l.append(None)
            aligned_actions_r.append(clean_actions_r[index_r])
            index_r += 1
        elif index_r >= len(clean_actions_r):
            # No more clean actions for right gripper
            aligned_actions_l.append(clean_actions_l[index_l])
            aligned_actions_r.append(None)
            index_l += 1
        elif len(set(clean_actions_l[index_l].counts) & set(clean_actions_r[index_r].counts)) > 0:
            # Actions overlap in time. Consider them the same action
            aligned_actions_l.append(clean_actions_l[index_l])
            aligned_actions_r.append(clean_actions_r[index_r])
            index_l += 1
            index_r += 1
        else:
            if clean_actions_l[index_l].counts[0] < clean_actions_r[index_r].counts[0]:
                # Left action is first
                aligned_actions_l.append(clean_actions_l[index_l])
                aligned_actions_r.append(None)
                index_l += 1
            else:
                # Right action is first
                aligned_actions_l.append(None)
                aligned_actions_r.append(clean_actions_r[index_r])
                index_r += 1
    assert len(aligned_actions_l) == len(aligned_actions_r)
    return aligned_actions_l, aligned_actions_r
