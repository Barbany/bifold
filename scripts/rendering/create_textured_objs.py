import argparse
import os

import numpy as np
import zarr
from scipy.spatial import KDTree
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_root_path", type=str, required=True)
    parser.add_argument(
        "--obj_root_path", type=str, required=True, desc="Path to textured .obj meshes"
    )
    parser.add_argument("--cloth3d_root_path", type=str, required=True)
    parser.add_argument("--min_distance", type=float, default=1e-6)
    args = parser.parse_args()

    for category in tqdm(os.listdir(args.zarr_root_path)):
        obj_root_path = args.obj_root_path + category

        os.makedirs(obj_root_path, exist_ok=True)

        root = zarr.open(os.path.join(args.zarr_root_path, category), mode="r")

        cloth3d_to_textures_coords = {}
        cloth3d_to_textures_faces = {}
        cloth3d_to_metadata = {}

        for group_name, sample_group in tqdm(root["samples"].groups(), leave=False):
            if not os.path.isfile(os.path.join(obj_root_path, group_name + ".obj")):
                cloth3d_id = group_name.split("_")[0]
                if cloth3d_id not in cloth3d_to_textures_faces:
                    # Infer texture coordinates. This has to be done once as
                    # NOCS vertices are always the same

                    # Read texture coordinates from Cloth3D
                    vertices = []
                    texture_coords = []
                    faces_verts = []
                    faces_texts = []
                    metadata = []

                    # Textured obj is created by using NOCS vertices and assigning material
                    with open(
                        os.path.join(
                            args.cloth3d_root_path, cloth3d_id, "textured_" + category + ".obj"
                        ),
                        "r",
                    ) as f:
                        for line in f.readlines():
                            if line.startswith("v "):
                                vertices.append(
                                    [float(n) for n in line.replace("v ", "").split(" ")]
                                )
                            elif line.startswith("vt "):
                                texture_coords.append(line)
                            elif line.startswith("f "):
                                face_verts = []
                                face_texts = []
                                for element in line.replace("f ", "").rsplit():
                                    vertex_id, texture_id = element.split("/")
                                    face_verts.append(vertex_id)
                                    face_texts.append(texture_id)
                                faces_verts.append(face_verts)
                                faces_texts.append(face_texts)
                            else:
                                if line.strip():
                                    metadata.append(line)

                    vertices = np.array(vertices)

                    cloth3d_to_textures_coords[cloth3d_id] = texture_coords
                    cloth3d_to_metadata[cloth3d_id] = metadata

                    # Read target vertices and faces
                    target_vertices = sample_group["mesh"]["cloth_nocs_verts"][:]
                    target_faces = sample_group["mesh"]["cloth_faces_tri"][:]

                    # Map vertices to target vertices
                    distances, trg_to_src = KDTree(vertices).query(target_vertices)
                    assert np.all(
                        distances < args.min_distance
                    ), f"Maximum distance is {distances.max()}"

                    target_texts = []

                    # Find which face we are in (3 out of the 4 vertices match)
                    for target_face in tqdm(target_faces, leave=False):
                        found = False
                        trg_verts = [str(v + 1) for v in trg_to_src[target_face]]
                        for src_verts, src_texts in zip(faces_verts, faces_texts):
                            common_elements = len(set(src_verts) & set(trg_verts))
                            if common_elements == 3:
                                # The faces correspond
                                found = True
                                target_texts.append(
                                    [src_texts[src_verts.index(v)] for v in trg_verts]
                                )
                                break
                        assert found

                    cloth3d_to_textures_faces[cloth3d_id] = target_texts

                # Write mesh in .obj format with inferred textures coordinates
                with open(os.path.join(obj_root_path, group_name + ".obj"), "w") as f:
                    f.writelines(cloth3d_to_metadata[cloth3d_id])

                    for v in sample_group["mesh"]["cloth_verts"][:]:
                        f.write(f"v {v[0]} {v[1]} {v[2]}\n")

                    f.writelines(cloth3d_to_textures_coords[cloth3d_id])
                    for face_vert, face_text in zip(
                        sample_group["mesh"]["cloth_faces_tri"][:],
                        cloth3d_to_textures_faces[cloth3d_id],
                    ):
                        f.write(
                            f"f {face_vert[0] + 1}/{face_text[0]} "
                            f"{face_vert[1] + 1}/{face_text[1]} "
                            f"{face_vert[2] + 1}/{face_text[2]}\n"
                        )
