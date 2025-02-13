import abc
from copy import deepcopy
from typing import List, Optional

import cv2
import imageio
import numpy as np
import pyflex
import scipy.spatial
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from .softgym_utils import get_matrix_world_to_camera


class SoftgymClothEnv:
    def __init__(
        self,
        gui=False,
        dump_visualizations=False,
        render_dim=224,
        particle_radius=0.00625,
    ):

        # environment state variables
        self.grasp_states = [False, False]
        self.particle_radius = particle_radius
        self.image_dim = render_dim

        # visualizations
        self.gui = gui
        self.dump_visualizations = dump_visualizations
        self.gui_render_freq = 2
        self.gui_step = 0

        # setup env
        self.setup_env()

        # primitives parameters
        self.grasp_height = self.action_tool.picker_radius
        self.default_speed = 1e-2
        self.reset_pos = [[0.5, 0.2, 0.5], [-0.5, 0.2, 0.5]]
        self.default_pos = [-0.5, 0.2, 0.5]
        self.fling_speed = 5e-2

    def close(self):
        pyflex.clean()

    def setup_env(self):
        pyflex.init(not self.gui, True, 720, 720)
        self.action_tool = PickerPickPlace(
            num_picker=2,
            particle_radius=self.particle_radius,
            picker_threshold=0.005,
            picker_low=(-10.0, 0.0, -10.0),
            picker_high=(10.0, 10.0, 10.0),
        )
        if self.dump_visualizations:
            self.frames = []

    def get_world_coord_from_pixel(self, pixel, depth):
        assert np.all(pixel >= 0)
        assert np.all(pixel < depth.shape)
        matrix_camera_to_world = np.linalg.inv(self.camera_matrix)
        height, width = depth.shape
        K = self.intrinsic_from_fov(height, width, 45)

        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        u, v = pixel[0], pixel[1]
        z = depth[int(np.rint(u)), int(np.rint(v))]
        x = (u - u0) * z / fx
        y = (v - v0) * z / fy

        cam_coord = np.ones(4)
        cam_coord[:3] = (x, y, z)
        world_coord = matrix_camera_to_world @ cam_coord

        return world_coord[:3]

    @staticmethod
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

        return np.array(
            [[fx, 0, px, 0.0], [0, fy, py, 0.0], [0, 0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

    def reset(
        self,
        config,
        state,
        cloth3d,
        pick_speed=5e-3,
        move_speed=5e-3,
        place_speed=5e-3,
        lift_height=0.1,
    ):
        self.current_config = deepcopy(config)
        if cloth3d:
            set_cloth3d_scene(config=config, state=state)
        else:
            set_square_scene(config=config, state=state)
        self.camera_params = deepcopy(state["camera_params"])
        self.camera_matrix = get_matrix_world_to_camera(self.camera_params)

        self.action_tool.reset(self.reset_pos[0])
        self.step_simulation()
        self.set_grasp(False)
        if self.dump_visualizations:
            self.frames = []

        self.pick_speed = pick_speed
        self.move_speed = move_speed
        self.place_speed = place_speed
        self.lift_height = lift_height

        self.max_area = state["max_area"]

    def step_simulation(self):
        pyflex.step()
        if self.gui and self.gui_step % self.gui_render_freq == 0:
            pyflex.render()
        self.gui_step += 1

    def set_grasp(self, grasp):
        self.grasp_states = [grasp] * len(self.grasp_states)

    def render_image(self):
        rgb, depth = pyflex.render()
        rgb = rgb.reshape((720, 720, 4))[::-1, :, :3]
        depth = depth.reshape((720, 720))[::-1]
        rgb = cv2.resize(rgb, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        return rgb, depth

    def render_gif(self, path):
        with imageio.get_writer(path, mode="I", fps=30) as writer:
            for frame in tqdm(self.frames):
                writer.append_data(frame)  # type: ignore

    # Picker
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.1
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr) for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(
                target_pos, curr_pos, deltas, dists, self.grasp_states
            ):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta / dist
                    action.extend([*(curr + delta * speed), float(gs)])

            action = np.array(action)
            self.action_tool.step(action, step_sim_fn=self.step_simulation)
            if self.dump_visualizations:
                self.frames.append(self.render_image()[0])

    # single arm primitive, default use picker1 for manipulation
    def pick_and_place_single(self, pick_pos, place_pos):
        pick_pos[1] = self.grasp_height
        place_pos[1] = self.grasp_height

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = self.lift_height

        preplace_pos = place_pos.copy()
        preplace_pos[1] = self.lift_height

        # execute action
        self.movep([prepick_pos, self.default_pos], speed=0.5)
        self.movep([pick_pos, self.default_pos], speed=0.005)
        self.set_grasp(True)
        self.movep([prepick_pos, self.default_pos], speed=self.pick_speed)
        self.movep([preplace_pos, self.default_pos], speed=self.move_speed)
        self.movep([place_pos, self.default_pos], speed=self.place_speed)
        self.set_grasp(False)
        self.movep([preplace_pos, self.default_pos], speed=0.5)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    # pick and drop
    def pick_and_drop(self, pick_pos):
        pick_pos[1] = self.grasp_height
        prepick_pos = pick_pos.copy()
        prepick_pos[1] = self.lift_height

        # execute action
        self.movep([prepick_pos, self.default_pos], speed=0.5)
        self.movep([pick_pos, self.default_pos], speed=0.005)
        self.set_grasp(True)
        self.movep([prepick_pos, self.default_pos], speed=self.pick_speed)
        self.set_grasp(False)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    # dual arm primitive
    def pick_and_place_dual(self, pick_pos_left, place_pos_left, pick_pos_right, place_pos_right):
        pick_pos_left[1] = self.grasp_height
        place_pos_left[1] = self.grasp_height
        pick_pos_right[1] = self.grasp_height
        place_pos_right[1] = self.grasp_height

        prepick_pos_left = pick_pos_left.copy()
        prepick_pos_left[1] = self.lift_height
        prepick_pos_right = pick_pos_right.copy()
        prepick_pos_right[1] = self.lift_height

        preplace_pos_left = place_pos_left.copy()
        preplace_pos_left[1] = self.lift_height
        preplace_pos_right = place_pos_right.copy()
        preplace_pos_right[1] = self.lift_height

        # execute action
        self.movep([prepick_pos_left, prepick_pos_right], speed=0.5)
        self.movep([pick_pos_left, pick_pos_right], speed=0.005)
        self.set_grasp(True)
        self.movep([prepick_pos_left, prepick_pos_right], speed=self.pick_speed)
        self.movep([preplace_pos_left, preplace_pos_right], speed=self.move_speed)
        self.movep([place_pos_left, place_pos_right], speed=self.place_speed)
        self.set_grasp(False)
        self.movep([preplace_pos_left, preplace_pos_right], speed=0.5)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    def pick_and_fling(self, pick_pos_left, pick_pos_right):
        pick_pos_left[1] = self.grasp_height
        pick_pos_right[1] = self.grasp_height

        prepick_pos_left = pick_pos_left.copy()
        prepick_pos_left[1] = self.lift_height

        prepick_pos_right = pick_pos_right.copy()
        prepick_pos_right[1] = self.lift_height

        # grasp distance
        dist = np.linalg.norm(np.array(prepick_pos_left) - np.array(prepick_pos_right))

        # pick cloth
        self.movep([prepick_pos_left, prepick_pos_right])
        self.movep([pick_pos_left, pick_pos_right])
        self.set_grasp(True)

        # prelift & stretch
        self.movep([[-dist / 2, 0.3, -0.3], [dist / 2, 0.3, -0.3]], speed=5e-3)
        if not self.is_cloth_grasped():
            return False
        dist = self.stretch_cloth(grasp_dist=dist, max_grasp_dist=0.4, fling_height=0.5)

        # lift
        fling_height = self.lift_cloth(grasp_dist=dist, fling_height=0.5)

        # fling
        self.fling(dist=dist, fling_height=fling_height, fling_speed=self.fling_speed)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    def fling(self, dist, fling_height, fling_speed):
        # fling
        self.movep(
            [[-dist / 2, fling_height, -0.2], [dist / 2, fling_height, -0.2]],
            speed=fling_speed,
        )
        self.movep(
            [[-dist / 2, fling_height, 0.2], [dist / 2, fling_height, 0.2]],
            speed=fling_speed,
        )
        self.movep(
            [[-dist / 2, fling_height, 0.2], [dist / 2, fling_height, 0.2]],
            speed=1e-2,
            min_steps=4,
        )

        # lower & flatten
        self.movep(
            [
                [-dist / 2, self.grasp_height * 2, 0.2],
                [dist / 2, self.grasp_height * 2, 0.2],
            ],
            speed=fling_speed,
        )

        self.movep(
            [[-dist / 2, self.grasp_height, 0], [dist / 2, self.grasp_height, 0]],
            speed=fling_speed,
        )

        self.movep(
            [[-dist / 2, self.grasp_height, -0.2], [dist / 2, self.grasp_height, -0.2]],
            speed=5e-3,
        )

        # release
        self.set_grasp(False)

        if self.dump_visualizations:
            self.movep(
                [
                    [-dist / 2, self.grasp_height * 2, -0.2],
                    [dist / 2, self.grasp_height * 2, -0.2],
                ],
                min_steps=10,
            )

    def stretch_cloth(self, grasp_dist, fling_height=0.7, max_grasp_dist=0.7, increment_step=0.02):
        # lift cloth in the air
        left, right = self.action_tool._get_pos()[0]
        left[1] = fling_height
        right[1] = fling_height
        midpoint = (left + right) / 2
        direction = left - right
        direction = direction / np.linalg.norm(direction)
        self.movep([left, right], speed=5e-4, min_steps=20)
        stable_steps = 0
        cloth_midpoint = 1e2
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            # get midpoints
            high_positions = positions[positions[:, 1] > fling_height - 0.1, ...]
            if (high_positions[:, 0] < 0).all() or (high_positions[:, 0] > 0).all():
                # single grasp
                return grasp_dist
            positions = [p for p in positions]
            positions.sort(key=lambda pos: np.linalg.norm(pos[[0, 2]] - midpoint[[0, 2]]))
            new_cloth_midpoint = positions[0]
            stable = np.linalg.norm(new_cloth_midpoint - cloth_midpoint) < 1.5e-2
            if stable:
                stable_steps += 1
            else:
                stable_steps = 0
            stretched = stable_steps > 2
            if stretched:
                return grasp_dist
            cloth_midpoint = new_cloth_midpoint
            grasp_dist += increment_step
            left = midpoint + direction * grasp_dist / 2
            right = midpoint - direction * grasp_dist / 2
            self.movep([left, right], speed=5e-4)
            if grasp_dist > max_grasp_dist:
                return max_grasp_dist

    def lift_cloth(
        self,
        grasp_dist,
        fling_height: float = 0.7,
        increment_step: float = 0.05,
        max_height=0.7,
    ):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            heights = positions[:, 1]
            if heights.min() > 0.02:
                return fling_height
            fling_height += increment_step
            self.movep(
                [
                    [-grasp_dist / 2, fling_height, -0.3],
                    [grasp_dist / 2, fling_height, -0.3],
                ],
                speed=1e-3,
            )
            if fling_height >= max_height:
                return fling_height

    # Ground truth
    # square cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    # Cloth Keypoints are defined:
    #  0  1  2
    #  3  4  5
    #  6  7  8
    def get_square_keypoints_idx(self):
        """The keypoints are defined as the four corner points of the cloth"""
        dimx, dimy = self.current_config["ClothSize"]
        idx0 = 0
        idx1 = int((dimx - 1) / 2)
        idx2 = dimx - 1
        idx3 = int((dimy - 1) / 2) * dimx
        idx4 = int((dimy - 1) / 2) * dimx + int((dimx - 1) / 2)
        idx5 = int((dimy - 1) / 2) * dimx + dimx - 1
        idx6 = dimx * (dimy - 1)
        idx7 = dimx * (dimy - 1) + int((dimx - 1) / 2)
        idx8 = dimx * dimy - 1
        return [idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8]

    def get_keypoints(self, keypoints_index=None):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        if keypoints_index is None:
            return particle_pos
        else:
            keypoint_pos = particle_pos[keypoints_index, :3]
            return keypoint_pos

    def is_cloth_grasped(self):
        positions = pyflex.get_positions().reshape((-1, 4))
        positions = positions[:, :3]
        heights = positions[:, 1]
        return heights.max() > 0.2


class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """Reset"""

    @abc.abstractmethod
    def step(self, action) -> Optional[int]:
        """Step funciton to change the action space states. Does not call pyflex.step()"""


class Picker(ActionToolBase):
    def __init__(
        self,
        num_picker=1,
        picker_radius=0.05,
        init_pos=(0.0, -0.1, 0.0),
        picker_threshold=0.005,
        particle_radius=0.05,
        picker_low=(-0.4, 0.0, -0.4),
        picker_high=(0.4, 0.5, 0.4),
        init_particle_pos=None,
        spring_coef=1.2,
        **kwargs,
    ):
        """

        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles: List[Optional[int]] = [None] * self.num_picker
        self.picker_low, self.picker_high = np.array(list(picker_low)), np.array(list(picker_high))
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        self.spring_coef = spring_coef  # Prevent picker to drag two particles too far away

    def update_picker_boundary(self, picker_low, picker_high):
        self.picker_low, self.picker_high = (
            np.array(picker_low).copy(),
            np.array(picker_high).copy(),
        )

    def visualize_picker_boundary(self):
        halfEdge = np.array(self.picker_high - self.picker_low) / 2.0
        center = np.array(self.picker_high + self.picker_low) / 2.0
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        pyflex.add_box(halfEdge, center, quat)

    def _apply_picker_boundary(self, picker_pos):
        clipped_picker_pos = picker_pos.copy()
        for i in range(3):
            clipped_picker_pos[i] = np.clip(
                picker_pos[i],
                self.picker_low[i] + self.picker_radius,
                self.picker_high[i] - self.picker_radius,
            )
        return clipped_picker_pos

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.0
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, state):
        for i in (0, 2):
            offset = state[i] - (self.picker_high[i] + self.picker_low[i]) / 2.0
            self.picker_low[i] += offset
            self.picker_high[i] += offset
        init_picker_poses = self._get_centered_picker_pos(state)

        for picker_pos in init_picker_poses:
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(state)
        for i, centered_picker_pos in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack(
                [centered_picker_pos, centered_picker_pos, [1, 0, 0, 0], [1, 0, 0, 0]]
            )
        pyflex.set_shape_states(shape_state)
        # pyflex.step() # Remove this as having an additional step here
        # may affect the cloth drop env
        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]
        # print('inv_mass_shape after reset:', self.particle_inv_mass.shape)

    @staticmethod
    def _get_pos():
        """Get the current pos of the pickers and the particles, along with the inverse
        mass of each particle"""
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    @staticmethod
    def set_picker_pos(picker_pos):
        """Caution! Should only be called during the reset of the environment
        Used only for cloth drop environment."""
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)

    def step(self, action) -> Optional[int]:
        """action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one, for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        action = np.reshape(action, [-1, 4])
        # pick_flag = np.random.random(self.num_picker) < action[:, 3]
        pick_flag = action[:, 3] > 0.5
        picker_pos, particle_pos = self._get_pos()
        new_picker_pos, new_particle_pos = picker_pos.copy(), particle_pos.copy()

        # Un-pick the particles
        # print(
        #     "check pick id:",
        #     self.picked_particles,
        #     new_particle_pos.shape,
        #     self.particle_inv_mass.shape,
        # )
        for i in range(self.num_picker):
            if not pick_flag[i] and self.picked_particles[i] is not None:
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[
                    self.picked_particles[i]
                ]  # Revert the mass
                self.picked_particles[i] = None

        # Pick new particles and update the mass and the positions
        for i in range(self.num_picker):
            new_picker_pos[i, :] = self._apply_picker_boundary(picker_pos[i, :] + action[i, :3])
            if pick_flag[i]:
                if (
                    self.picked_particles[i] is None
                ):  # No particle is currently picked and thus need to select a particle to pick
                    dists = scipy.spatial.distance.cdist(
                        picker_pos[i].reshape((-1, 3)),
                        particle_pos[:, :3].reshape((-1, 3)),
                    )
                    idx_dists = np.hstack([
                        np.arange(particle_pos.shape[0]).reshape((-1, 1)),
                        dists.reshape((-1, 1)),
                    ])
                    mask = (
                        dists.flatten()
                        <= self.picker_threshold + self.picker_radius + self.particle_radius
                    )
                    idx_dists = idx_dists[mask, :].reshape((-1, 2))
                    if idx_dists.shape[0] > 0:
                        pick_id, pick_dist = None, None
                        for j in range(idx_dists.shape[0]):
                            if idx_dists[j, 0] not in self.picked_particles and (
                                pick_id is None or idx_dists[j, 1] < pick_dist
                            ):
                                pick_id = idx_dists[j, 0]
                                pick_dist = idx_dists[j, 1]
                        if pick_id is not None:
                            self.picked_particles[i] = int(pick_id)

                if self.picked_particles[i] is not None:
                    # The position of the particle needs to be updated
                    # such that it is close to the picker particle
                    new_particle_pos[self.picked_particles[i], :3] = (
                        particle_pos[self.picked_particles[i], :3]
                        + new_picker_pos[i, :]
                        - picker_pos[i, :]
                    )
                    new_particle_pos[self.picked_particles[i], 3] = 0  # Set the mass to infinity

        # check for e.g., rope, the picker is not dragging the particles too far away
        # that violates the actual physicals constraints.
        if self.init_particle_pos is not None:
            picked_particle_idices = []
            active_picker_indices = []
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    picked_particle_idices.append(self.picked_particles[i])
                    active_picker_indices.append(i)

            l = len(picked_particle_idices)
            for i in range(l):
                for j in range(i + 1, l):
                    init_distance = np.linalg.norm(
                        self.init_particle_pos[picked_particle_idices[i], :3]
                        - self.init_particle_pos[picked_particle_idices[j], :3]
                    )
                    now_distance = np.linalg.norm(
                        new_particle_pos[picked_particle_idices[i], :3]
                        - new_particle_pos[picked_particle_idices[j], :3]
                    )
                    if (
                        now_distance >= init_distance * self.spring_coef
                    ):  # if dragged too long, make the action has no effect; revert it
                        new_picker_pos[active_picker_indices[i], :] = picker_pos[
                            active_picker_indices[i], :
                        ].copy()
                        new_picker_pos[active_picker_indices[j], :] = picker_pos[
                            active_picker_indices[j], :
                        ].copy()
                        new_particle_pos[picked_particle_idices[i], :3] = particle_pos[
                            picked_particle_idices[i], :3
                        ].copy()
                        new_particle_pos[picked_particle_idices[j], :3] = particle_pos[
                            picked_particle_idices[j], :3
                        ].copy()

        self._set_pos(new_picker_pos, new_particle_pos)


class PickerPickPlace(Picker):
    def __init__(self, num_picker, env=None, picker_low=None, picker_high=None, **kwargs):
        super().__init__(
            num_picker=num_picker,
            picker_low=picker_low,
            picker_high=picker_high,
            **kwargs,
        )
        assert picker_low is not None and picker_high is not None
        picker_low, picker_high = list(picker_low), list(picker_high)
        self.delta_move = 0.01
        self.env = env

    def step(self, action, step_sim_fn=lambda: pyflex.step()):
        """
        action: Array of pick_num x 4. For each picker, the action should be [x, y, z, pick/drop].
        The picker will then first pick/drop, and keep
        the pick/drop state while moving towards x, y, z.
        """
        total_steps = 0
        action = action.reshape(-1, 4)
        curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[: self.num_picker, :3]
        end_pos = np.vstack(
            [self._apply_picker_boundary(picker_pos) for picker_pos in action[:, :3]]
        )
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        for i in range(
            int(min(num_step, 300))
        ):  # The maximum number of steps allowed for one pick and place
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.all(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            step_sim_fn()
            total_steps += 1
            if np.all(dist < self.delta_move):
                break
        return total_steps

    def get_model_action(self, action, picker_pos):
        """Input the action and return the action used for model prediction"""
        action = action.reshape(-1, 4)
        curr_pos = picker_pos
        end_pos = np.vstack(
            [self._apply_picker_boundary(picker_pos) for picker_pos in action[:, :3]]
        )
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return [], curr_pos
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        model_actions = []
        for i in range(
            int(min(num_step, 300))
        ):  # The maximum number of steps allowed for one pick and place
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.all(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            model_actions.append(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            curr_pos += delta
            if np.all(dist < self.delta_move):
                break
        return model_actions, curr_pos


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
    rotate_particles([180, 0, 90])
    move_to_pos([0, 0.05, 0])


def set_square_scene(config, state=None):
    render_mode = 2
    camera_params = config["camera_params"][config["camera_name"]]
    env_idx = 0
    mass = config["mass"] if "mass" in config else 0.5
    scene_params = np.array([
        *config["ClothPos"],
        *config["ClothSize"],
        *config["ClothStiff"],
        render_mode,
        *camera_params["pos"][:],
        *camera_params["angle"][:],
        camera_params["width"],
        camera_params["height"],
        mass,
        config["flip_mesh"],
    ])

    pyflex.set_scene(env_idx, scene_params, 0)

    if state is not None:
        set_state(state)

    for _ in range(50):
        pyflex.step()

    if state is not None:
        set_state(state)


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


def set_state(state_dict):
    pyflex.set_positions(state_dict["particle_pos"])
    pyflex.set_velocities(state_dict["particle_vel"])
    pyflex.set_shape_states(state_dict["shape_pos"])
    pyflex.set_phases(state_dict["phase"])
    camera_params = deepcopy(state_dict["camera_params"])
    update_camera(camera_params, "default_camera")
