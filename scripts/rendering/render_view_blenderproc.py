import blenderproc as bproc

""""""
import argparse
import os
import random
import shutil

import bpy
import cv2
import numpy as np
from mathutils import Vector
from scipy.spatial.transform import Rotation as R

DEBUG = False


def is_vertex_occluded_for_scene_camera(co, helper_cube_scale: float = 0.0001) -> bool:
    """Checks if a vertex is occluded by objects in the scene w.r.t. the camera.

    Args:
        co (Vector): the world space x, y and z coordinates of the vertex.

    Returns:
        boolean: visibility
    """
    co = Vector(co)

    bpy.context.view_layer.update()  # ensures camera matrix is up to date
    scene = bpy.context.scene
    camera_obj = scene.camera  # bpy.types.Object

    # add small cube around coord to make sure the ray will intersect
    # as the ray_cast is not always accurate
    # cf https://blender.stackexchange.com/a/87755
    bpy.ops.mesh.primitive_cube_add(
        location=co, scale=(helper_cube_scale, helper_cube_scale, helper_cube_scale)
    )
    cube = bpy.context.object
    direction = co - camera_obj.location
    hit, location, _, _, _, _ = scene.ray_cast(
        bpy.context.view_layer.depsgraph,
        origin=camera_obj.location + direction * 0.0001,  # avoid self intersection
        direction=direction,
    )

    if DEBUG:
        print(f"hit location: {location}")
        bpy.ops.mesh.primitive_ico_sphere_add(
            location=location,
            scale=(helper_cube_scale, helper_cube_scale, helper_cube_scale),
        )

    # remove the auxiliary cube
    if not DEBUG:
        bpy.data.objects.remove(cube, do_unlink=True)

    if not hit:
        raise ValueError(
            "No hit found, this should not happen as the ray should always hit the vertex itself."
        )
    # if the hit is the vertex itself, it is not occluded
    if (location - co).length < helper_cube_scale * 2:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blenderproc rendering pipeline")
    parser.add_argument("--obj_file", required=True, type=str)
    parser.add_argument("--renders_root_path", required=True, type=str)
    parser.add_argument("--cloth3d_root_path", required=True, type=str)
    parser.add_argument("--compute_visibility", action="store_true")
    args = parser.parse_args()

    os.makedirs("assets", exist_ok=True)

    resolution = 384
    radius_min = 1.8
    radius_max = 2
    elevation_min = 45
    elevation_max = 90

    depth_scale = 1000
    depth_truncate = 1000

    seed = 42

    random.seed(seed)
    np.random.seed(seed)

    bproc.init()

    # activate normal and depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.camera.set_resolution(resolution, resolution)

    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    cameras = {}

    obj_file_full_path = args.obj_file
    obj_file = os.path.split(obj_file_full_path)[1]

    cloth3d_id, category, *_ = obj_file.split("_")
    garment_tracking_seq = "_".join(obj_file.split("_")[:-1])
    shutil.copyfile(
        src=obj_file_full_path,
        dst="assets/mesh.obj",
    )
    shutil.copyfile(
        src=os.path.join(args.cloth3d_root_path, cloth3d_id, "texture.mtl"),
        dst="assets/texture.mtl",
    )
    cloth = bproc.loader.load_obj(filepath="assets/mesh.obj")
    assert len(cloth) == 1
    cloth[0].set_shading_mode("smooth")
    if garment_tracking_seq not in cameras:
        os.makedirs(
            os.path.join(
                args.renders_root_path,
                category,
                "cameras",
            ),
            exist_ok=True,
        )
        cam2world_file = os.path.join(
            args.renders_root_path,
            category,
            "cameras",
            garment_tracking_seq + ".npy",
        )
        if not os.path.isfile(cam2world_file):
            poi = bproc.object.compute_poi(cloth)
            # location = bproc.sampler.sphere(center=poi, radius=radius, mode="SURFACE")
            location = bproc.sampler.shell(
                center=poi,
                radius_min=radius_min,
                radius_max=radius_max,
                elevation_min=elevation_min,
                elevation_max=elevation_max,
            )
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

            cameras[garment_tracking_seq] = cam2world_matrix

            np.save(
                cam2world_file,
                cam2world_matrix,
            )

        else:
            cam2world_matrix = np.load(cam2world_file)
            cameras[garment_tracking_seq] = cam2world_matrix

        camera_matrix_file = os.path.join(
            args.renders_root_path,
            category,
            "camera_matrix",
            garment_tracking_seq + ".npy",
        )
        if not os.path.isfile(camera_matrix_file):
            os.makedirs(
                os.path.join(
                    args.renders_root_path,
                    category,
                    "camera_matrix",
                ),
                exist_ok=True,
            )
            rot = np.eye(4)
            rot[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix()

            intr = np.eye(4)
            intr[:3, :3] = bproc.python.camera.CameraUtility.get_intrinsics_as_K_matrix()
            camera_matrix = intr @ np.linalg.inv(cam2world_matrix) @ rot

            np.save(
                camera_matrix_file,
                camera_matrix,
            )

    bproc.camera.add_camera_pose(cameras[garment_tracking_seq], 0)

    # render the whole pipeline
    data = bproc.renderer.render()

    if args.compute_visibility:
        obj = bpy.context.selected_objects[0]
        visibility = []
        for vertex in obj.data.vertices:
            coords = obj.matrix_world @ vertex.co
            visibility.append(is_vertex_occluded_for_scene_camera(coords))
        data["visibility"] = [np.array(visibility)]

    for k, vals in data.items():
        out_folder = os.path.join(args.renders_root_path, category, k)
        os.makedirs(out_folder, exist_ok=True)
        assert len(vals) == 1, f"Vals have length {len(vals)}"
        if k == "colors":
            cv2.imwrite(
                os.path.join(out_folder, obj_file.replace(".obj", ".png")),
                cv2.cvtColor(vals[0], cv2.COLOR_RGB2BGR),
            )
        elif k == "depth":
            cv2.imwrite(
                os.path.join(out_folder, obj_file.replace(".obj", ".png")),
                (np.clip(vals[0], a_min=0, a_max=depth_truncate) * depth_scale).astype(np.uint16),
            )
        else:
            np.save(
                file=os.path.join(out_folder, obj_file.replace(".obj", ".npy")),
                arr=vals[0],
            )
