import numpy as np


def get_matrix_world_to_camera(camera_params):
    cam_x, cam_y, cam_z = (
        camera_params["default_camera"]["pos"][0],
        camera_params["default_camera"]["pos"][1],
        camera_params["default_camera"]["pos"][2],
    )
    cam_x_angle, cam_y_angle = (
        camera_params["default_camera"]["angle"][0],
        camera_params["default_camera"]["angle"][1],
    )

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(-cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(-cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = -cam_x
    translation_matrix[1][3] = -cam_y
    translation_matrix[2][3] = -cam_z

    return rotation_matrix @ translation_matrix


def get_rotation_matrix(angle, axis):
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    # m[0][1] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    # m[0][2] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    # m[1][0] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    # m[1][2] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    # m[2][0] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    # m[2][1] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360.0 * 2.0 * np.pi
    fx = width / (2.0 * np.tan(hfov / 2.0))

    vfov = 2.0 * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2.0 * np.tan(vfov / 2.0))

    return np.array([[fx, 0, px, 0.0], [0, fy, py, 0.0], [0, 0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
