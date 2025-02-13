import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pyflex
import torch
from PIL import Image
from tqdm import trange

from bifold.data.single_dataset import get_mask_from_depth
from bifold.metrics.utils import iou
from bifold.utils.visualization import save_predictions, visualize_action

from .softgym_cloth_env import SoftgymClothEnv, rotate_particles
from .softgym_demonstrators import Demonstrator

task_to_cloth_type = {
    "CornerFold": "Square",
    "TriangleFold": "Square",
    "StraightFold": "Rectangular",
    "TshirtFold": "Tshirt",
    "TrousersFold": "Trousers",
}


class SoftgymEvaluator:
    def __init__(self, cfg, model, processor, task=None):
        self.model = model
        self.processor = processor
        self.cache = cfg.softgym_cache

        self.visualize_predictions = cfg.visualize_predictions

        self.env = SoftgymClothEnv(
            render_dim=cfg.model.image_size,
            dump_visualizations=self.visualize_predictions,
        )

        self.K = self.env.intrinsic_from_fov(
            height=cfg.model.image_size, width=cfg.model.image_size
        )

        self.error_threshold = self.env.particle_radius * 2
        self.iou_thresholds = [50, 80, 90]

        self.success = {}
        self.additional_metrics = {}

    def reset(
        self,
        config,
        state,
        task=None,
        random_angle=None,
        max_wait_step=300,
        stable_vel_threshold=0.2,
    ):
        self.demonstrator = Demonstrator[task]()
        self.env.reset(
            config=config,
            state=state,
            cloth3d=self.cloth3d,
            pick_speed=self.demonstrator.pick_speed,
            move_speed=self.demonstrator.move_speed,
            place_speed=self.demonstrator.place_speed,
            lift_height=self.demonstrator.lift_height,
        )
        self.task = task if task is not None else ""
        if random_angle:
            rotate_particles([0, random_angle, 0])
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.all(np.abs(curr_vel) < stable_vel_threshold):
                    break

    def evaluate(self, cloth_type: Optional[str] = None, cloth3d=True, **kwargs):
        self.cloth3d = cloth3d
        # load configs
        assert cloth_type is not None
        with open(os.path.join(self.cache, cloth_type + ".pkl"), "rb") as f:
            config_data = pickle.load(f)
        self.cached_configs = config_data["configs"]
        self.cached_states = config_data["states"]
        if self.cloth3d:
            self.cached_keypoints = config_data["keypoints"]

    def close(self):
        self.env.close()

    def save_visuals(self, out_file_name, **kwargs):
        if self.visualize_predictions:
            save_predictions(
                out_folder=os.path.join("eval", "softgym", self.task),
                out_file_name=out_file_name,
                **kwargs,
            )

    def add_to_success(self, oracle_results, model_results, unseen_flags):
        raise NotImplementedError

    def summary(self):
        return_dict = {}
        average_success = []

        for task, task_dict in self.success.items():
            if isinstance(task_dict, dict):
                for k in task_dict.keys():
                    average = float(np.array(self.success[task][k]).mean() * 100)
                    return_dict[task + " " + k] = average
                    average_success.append(average)
            else:
                average = float(np.array(self.success[task]).mean() * 100)
                average_success.append(average)
                return_dict[task] = average

        for metric, metric_dicts in self.additional_metrics.items():
            for task, task_dict in metric_dicts.items():
                if isinstance(task_dict, dict):
                    for k in task_dict.keys():
                        average = np.array(metric_dicts[task][k]).mean()
                        return_dict[metric + " " + task + " " + k] = float(average)
                else:
                    average = np.array(metric_dicts[task]).mean()
                    return_dict[metric + " " + task] = float(average)
        return_dict["average_success"] = float(np.array(average_success).mean())
        return return_dict


class SoftgymSingleEvaluator(SoftgymEvaluator):
    def __init__(self, cfg, model, processor):
        super().__init__(cfg, model, processor)

    def evaluate(
        self,
        cloth_type=None,
        cloth3d=None,
        num_evals: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs,
    ):
        if cloth_type is None:
            assert task is not None
            cloth_type = task_to_cloth_type[task]
            cloth3d = False if (cloth_type == "Square" or cloth_type == "Rectangular") else True
        else:
            assert cloth3d is not None

        super().evaluate(cloth_type=cloth_type, cloth3d=cloth3d)

        if task not in self.success:
            self.success[task] = {}
            self.additional_metrics = {
                k: {task: {}}
                for k in ["error", "iou"]
                + [f"iou_success_{thresh}" for thresh in self.iou_thresholds]
            }

        assert num_evals is not None
        for i in trange(num_evals, desc=f"Evaluating {task}"):
            rand_idx = np.random.randint(len(self.cached_configs))
            config = self.cached_configs[rand_idx]
            state = self.cached_states[rand_idx]

            if task == "StraightFold":
                random_angle = np.random.uniform(-80, 80)
            elif cloth3d:
                random_angle = np.random.uniform(-40, 40)
            else:
                random_angle = np.random.uniform(0, 40)

            # reset env
            self.reset(
                config=config,
                state=state,
                task=task,
                random_angle=random_angle,
            )

            if self.cloth3d:
                keypoints_index = self.cached_keypoints[rand_idx]
            else:
                keypoints_index = self.env.get_square_keypoints_idx()

            if task == "StraightFold":
                angle_mode = int(abs(random_angle) > 45) + int(random_angle < -45)
                # -45 - 45 mode 0
                # 45 - 90 mode 1
                # -90 - -45 mode 2
                eval_seen_instructions, eval_unseen_instructions, eval_unseen_tasks = (
                    self.demonstrator.get_eval_instruction(angle_mode)
                )
            else:
                eval_seen_instructions, eval_unseen_instructions, eval_unseen_tasks = (
                    self.demonstrator.get_eval_instruction()
                )
            eval_datas = [
                eval_seen_instructions,
                eval_unseen_instructions,
                eval_unseen_tasks,
            ]
            eval_name_list = ["si", "usi", "ut"]

            for eval_index in range(3):
                # eval seen instructions, unseen instructions, unseent task
                eval_data = eval_datas[eval_index]
                eval_name = eval_name_list[eval_index]
                pick_idxs = eval_data["pick"]
                place_idxs = eval_data["place"]
                gammas = eval_data["gammas"]
                unseen_flags = eval_data["flags"]
                instructions = eval_data["instructions"]

                if eval_name not in self.success[task]:
                    self.success[task][eval_name] = []
                    for k in self.additional_metrics.keys():
                        self.additional_metrics[k][task][eval_name] = []

                self.reset(
                    config=config,
                    state=state,
                    task=task,
                    random_angle=random_angle,
                )

                oracle_results, oracle_masks = self.execute_oracle(
                    pick_idxs=pick_idxs,
                    place_idxs=place_idxs,
                    gammas=gammas,
                    keypoints_index=keypoints_index,
                )

                # reset env
                self.reset(
                    config=config,
                    state=state,
                    task=task,
                    random_angle=random_angle,
                )

                self.execute_model(
                    pick_idxs=pick_idxs,
                    place_idxs=place_idxs,
                    gammas=gammas,
                    keypoints_index=keypoints_index,
                    instructions=instructions,
                    unseen_flags=unseen_flags,
                    eval_index=eval_index,
                    eval_name=eval_name,
                    trial_index=i,
                    oracle_results=oracle_results,
                    oracle_masks=oracle_masks,
                )

    def execute_oracle(self, pick_idxs, place_idxs, gammas, keypoints_index):
        oracle_results = []
        oracle_masks = []
        for pick_idx, place_idx, gamma in zip(pick_idxs, place_idxs, gammas):
            keypoints_pos = self.env.get_keypoints(keypoints_index)
            pick_pos = keypoints_pos[pick_idx]
            place_pos = keypoints_pos[place_idx]
            place_pos = pick_pos + gamma * (place_pos - pick_pos)
            self.env.pick_and_place_single(pick_pos.copy(), place_pos.copy())
            rgb, depth = self.env.render_image()
            mask = get_mask_from_depth(depth)
            particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
            oracle_results.append(particle_pos)
            oracle_masks.append(mask)
        return oracle_results, oracle_masks

    def execute_model(
        self,
        pick_idxs,
        place_idxs,
        gammas,
        keypoints_index,
        instructions,
        unseen_flags,
        eval_index,
        eval_name,
        trial_index,
        oracle_results,
        oracle_masks,
    ):
        rgb, depth = self.env.render_image()
        mask = get_mask_from_depth(depth)
        context = []

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(gammas)
            == len(instructions)
            == len(unseen_flags)
            == len(oracle_results)
            == len(oracle_masks)
        )

        action_index = None
        for action_index, (
            pick_idx,
            place_idx,
            gamma,
            instruction,
            unseen_flag,
        ) in enumerate(zip(pick_idxs, place_idxs, gammas, instructions, unseen_flags)):
            sample = self.processor(
                depth=depth,
                instruction=instruction,
                rgb=rgb,
                mask=mask,
                context=context,
                matrix_world_to_camera=self.env.camera_matrix,
                K=self.K,
            )
            for k, val in sample.items():
                if isinstance(val, torch.Tensor):
                    sample[k] = val.unsqueeze(0).to(self.model.device)
                elif k == "graph":
                    sample[k] = val.to(self.model.device)
                    sample[k]["batch"] = torch.zeros(
                        val["x"].size(0), dtype=torch.long, device=self.model.device
                    )
                    sample[k]["u"] = torch.zeros([1, 128], device=self.model.device)

            if eval_index < 2:  # eval seen instructions, unseen instructions,
                if unseen_flag == 1:  # oracle execute action
                    keypoints_pos = self.env.get_keypoints(keypoints_index)
                    pick_pos = keypoints_pos[pick_idx]
                    place_pos = keypoints_pos[place_idx]
                    place_pos = pick_pos + gamma * (place_pos - pick_pos)
                    action, raw_output = None, None
                    oracle_execution = True
                else:  # model execute action
                    action, raw_output = self.model.get_action(sample, return_raw_output=True)
                    assert len(action.pick) == 1
                    pick_pos = self.env.get_world_coord_from_pixel(action.pick[0], depth)
                    assert len(action.place) == 1
                    place_pos = self.env.get_world_coord_from_pixel(action.place[0], depth)
                    oracle_execution = False
            else:  # eval unseen tasks
                if unseen_flag == 0:  # oracle execute action
                    keypoints_pos = self.env.get_keypoints(keypoints_index)
                    pick_pos = keypoints_pos[pick_idx]
                    place_pos = keypoints_pos[place_idx]
                    place_pos = pick_pos + gamma * (place_pos - pick_pos)
                    action, raw_output = None, None
                    oracle_execution = True
                else:  # model execute action
                    action, raw_output = self.model.get_action(sample, return_raw_output=True)
                    assert len(action.pick) == 1
                    pick_pos = self.env.get_world_coord_from_pixel(action.pick[0], depth)
                    assert len(action.place) == 1
                    place_pos = self.env.get_world_coord_from_pixel(action.place[0], depth)
                    oracle_execution = False
            self.env.pick_and_place_single(pick_pos.copy(), place_pos.copy())  # take action

            old_rgb = rgb.copy()

            # render & update frames & save
            context.append({"rgb": rgb.copy(), "depth": depth.copy(), "mask": mask.copy()})

            rgb, depth = self.env.render_image()
            mask = get_mask_from_depth(depth)

            particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]

            error = np.linalg.norm(oracle_results[action_index] - particle_pos, axis=1).mean()
            success = error < self.error_threshold

            iou_value = iou(mask, oracle_masks[action_index])

            self.success[self.task][eval_name].append(success)
            self.additional_metrics["error"][self.task][eval_name].append(error)
            self.additional_metrics["iou"][self.task][eval_name].append(iou_value)

            iou_success = None
            for thresh in self.iou_thresholds:
                iou_success = (iou_value > thresh) * 100
                self.additional_metrics[f"iou_success_{thresh}"][self.task][eval_name].append(
                    iou_success
                )

            if self.visualize_predictions:
                if not oracle_execution:
                    visualizations = visualize_action(sample, action)
                    assert len(visualizations) == 1
                    self.save_visuals(
                        f"{eval_name}_{trial_index}_{action_index}_{instruction}_{success}.png",
                        viz=visualizations[0],
                        particle_pos=particle_pos,
                    )
                    if raw_output is not None:
                        self.save_visuals(
                            (
                                f"{eval_name}_{trial_index}_{action_index}_"
                                f"{instruction}_{success}_{iou_success}.png"
                            ),
                            rgb=old_rgb,
                            pick_heatmap=raw_output["pick_heatmap"],
                            place_heatmap=raw_output["place_heatmap"],
                            mask_heatmap=raw_output.get("mask_heatmap"),
                        )
                self.save_visuals(
                    f"{eval_name}_{trial_index}_{action_index + 1}.png",
                    rgb=rgb,
                    depth=depth,
                )

        out_dir = os.path.join(
            "eval",
            "softgym",
            self.task,
            "rollouts",
            f"{eval_name}_{trial_index}_{action_index}",
        )
        os.makedirs(out_dir, exist_ok=True)
        for frame_idx, frame in enumerate(self.env.frames):
            Image.fromarray(frame).save(os.path.join(out_dir, f"{frame_idx:05d}.png"))


class SoftgymBimanualEvaluator(SoftgymEvaluator):
    def __init__(self, cfg, model, processor):
        super().__init__(cfg, model, processor)

    def evaluate(
        self,
        cloth_type: Optional[str] = "bimanual",
        cloth3d: bool = True,
        samples: Optional[Dict[str, List]] = None,
        **kwargs,
    ):
        super().evaluate(cloth_type=cloth_type, cloth3d=cloth3d, **kwargs)
        assert samples is not None
        for sample_idx in range(len(samples["frame_start"])):
            sample_name = samples["frame_start"][sample_idx]

            self.task = sample_name.split("_")[1]
            if self.task not in self.success:
                self.success[self.task] = []
                if "error" not in self.additional_metrics:
                    self.additional_metrics = {
                        k: {self.task: []}
                        for k in ["error", "iou"]
                        + [f"iou_success_{thresh}" for thresh in self.iou_thresholds]
                    }
                else:
                    for k in self.additional_metrics.keys():
                        self.additional_metrics[k][self.task] = []

            config = self.cached_configs[sample_name]
            state = self.cached_states[sample_name]
            keypoints = self.cached_keypoints[sample_name]

            oracle_result, oracle_mask = self.execute_oracle(
                keypoints=keypoints, config=config, state=state
            )

            # Execute model
            self.execute_model(
                samples=samples,
                sample_idx=sample_idx,
                sample_name=sample_name,
                config=config,
                state=state,
                oracle_result=oracle_result,
                oracle_mask=oracle_mask,
            )

    def execute_model(
        self,
        samples,
        sample_idx,
        sample_name,
        config,
        state,
        oracle_result,
        oracle_mask,
    ):
        if "context" in samples:
            context = []
            for ctx in samples["context"][sample_idx].split("+"):
                if ctx != "":
                    config = self.cached_configs[ctx]
                    state = self.cached_states[ctx]
                    self.reset(config=config, state=state)
                    rgb, depth = self.env.render_image()
                    mask = get_mask_from_depth(depth)
                    context.append({"rgb": rgb, "depth": depth, "mask": mask})
        else:
            context = None

        config = self.cached_configs[sample_name]
        state = self.cached_states[sample_name]
        self.reset(config=config, state=state)
        rgb, depth = self.env.render_image()
        mask = get_mask_from_depth(depth)

        processed_inputs = self.processor(
            depth=depth,
            rgb=rgb,
            mask=mask,
            context=context,
            instruction=samples["raw_instruction"][sample_idx],
            matrix_world_to_camera=self.env.camera_matrix,
            K=self.K,
        )

        for k, val in processed_inputs.items():
            if isinstance(val, torch.Tensor):
                processed_inputs[k] = val.unsqueeze(0).to(self.model.device)
            elif k == "graph":
                processed_inputs[k] = val.to(self.model.device)
                processed_inputs[k]["batch"] = torch.zeros(
                    val["x"].size(0), dtype=torch.long, device=self.model.device
                )

        action, raw_output = self.model.get_action(processed_inputs, return_raw_output=True)
        if np.all(action.left_pick[0] >= 0) and np.all(action.left_place[0] >= 0):
            left_pick_pos = self.env.get_world_coord_from_pixel(action.left_pick[0], depth)
            left_place_pos = self.env.get_world_coord_from_pixel(action.left_place[0], depth)
        else:
            left_pick_pos = left_place_pos = None

        if np.all(action.right_pick[0] >= 0) and np.all(action.right_place[0] >= 0):
            right_pick_pos = self.env.get_world_coord_from_pixel(action.right_pick[0], depth)
            right_place_pos = self.env.get_world_coord_from_pixel(action.right_place[0], depth)
            if left_pick_pos is not None and left_place_pos is not None:
                self.env.pick_and_place_dual(
                    pick_pos_left=left_pick_pos,
                    place_pos_left=left_place_pos,
                    pick_pos_right=right_pick_pos,
                    place_pos_right=right_place_pos,
                )
            else:
                self.env.pick_and_place_single(pick_pos=right_pick_pos, place_pos=right_place_pos)
        else:
            assert left_pick_pos is not None and left_place_pos is not None
            self.env.pick_and_place_single(pick_pos=left_pick_pos, place_pos=left_place_pos)

        particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]

        old_rgb = rgb.copy()

        rgb, depth = self.env.render_image()
        mask = get_mask_from_depth(depth)

        error = np.linalg.norm(oracle_result - particle_pos, axis=1).mean()
        success = error < self.error_threshold

        iou_value = iou(mask, oracle_mask)

        self.task = sample_name.split("_")[1]
        self.success[self.task].append(success)
        self.additional_metrics["error"][self.task].append(error)
        self.additional_metrics["iou"][self.task].append(iou_value)

        iou_success = None
        for thresh in self.iou_thresholds:
            iou_success = (iou_value > thresh) * 100
            self.additional_metrics[f"iou_success_{thresh}"][self.task].append(iou_success)

        if self.visualize_predictions:
            instruction = samples["raw_instruction"][sample_idx]
            visualizations = visualize_action(processed_inputs, action)
            assert len(visualizations) == 1
            self.save_visuals(
                f"{sample_name}_{instruction}_{success}.png",
                viz=visualizations[0],
                particle_pos=particle_pos,
            )
            if raw_output is not None:
                self.save_visuals(
                    f"{sample_name}_{instruction}_{success}_{iou_success}.png",
                    rgb=old_rgb,
                    left_pick_heatmap=raw_output["left_pick_heatmap"],
                    right_pick_heatmap=raw_output["right_pick_heatmap"],
                    left_place_heatmap=raw_output["left_place_heatmap"],
                    right_place_heatmap=raw_output["right_place_heatmap"],
                    mask_heatmap=raw_output.get("mask_heatmap"),
                )
            out_dir = os.path.join(
                "eval",
                "softgym",
                self.task,
                f"{sample_name}_{instruction}_{success}_{iou_success}",
            )
            os.makedirs(out_dir, exist_ok=True)
            for frame_idx, frame in enumerate(self.env.frames):
                Image.fromarray(frame).save(os.path.join(out_dir, f"{frame_idx:05d}.png"))

    def execute_oracle(self, keypoints, config, state):
        self.reset(config=config, state=state)

        keypoints_pos = self.env.get_keypoints()
        if keypoints["left_pick_idx"] is not None:
            left_pick_pos = keypoints_pos[keypoints["left_pick_idx"]]
            left_place_pos = keypoints_pos[keypoints["left_place_idx"]]
            if keypoints["right_pick_idx"] is not None:
                # Both actions are correct
                right_pick_pos = keypoints_pos[keypoints["right_pick_idx"]]
                right_place_pos = keypoints_pos[keypoints["right_place_idx"]]
                self.env.pick_and_place_dual(
                    pick_pos_left=left_pick_pos,
                    place_pos_left=left_place_pos,
                    pick_pos_right=right_pick_pos,
                    place_pos_right=right_place_pos,
                )  # take action
            else:
                # only left action is correct
                self.env.pick_and_place_single(pick_pos=left_pick_pos, place_pos=left_place_pos)
        else:
            # only right action is correct
            assert keypoints["right_pick_idx"] is not None
            right_pick_pos = keypoints_pos[keypoints["right_pick_idx"]]
            right_place_pos = keypoints_pos[keypoints["right_place_idx"]]
            self.env.pick_and_place_single(pick_pos=right_pick_pos, place_pos=right_place_pos)

        rgb, depth = self.env.render_image()
        mask = get_mask_from_depth(depth)

        particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        return particle_pos, mask


class SoftgymBimanualRolloutEvaluator(SoftgymEvaluator):
    def evaluate(
        self,
        cloth_type: Optional[str] = "bimanual",
        cloth3d=True,
        sample_name: Optional[str] = None,
        instructions: Optional[List] = None,
        **kwargs,
    ):
        super().evaluate(cloth_type=cloth_type, cloth3d=cloth3d)
        if sample_name in self.cached_configs:
            self.execute_model(sample_name, instructions)

    def execute_model(
        self,
        sample_name,
        instructions,
    ):
        context = []

        config = self.cached_configs[sample_name]
        state = self.cached_states[sample_name]
        self.reset(config=config, state=state)
        rgb, depth = self.env.render_image()
        mask = get_mask_from_depth(depth)

        for instruction in instructions:
            processed_inputs = self.processor(
                depth=depth,
                rgb=rgb,
                mask=mask,
                context=context,
                instruction=instruction,
                matrix_world_to_camera=self.env.camera_matrix,
                K=self.K,
            )

            for k, val in processed_inputs.items():
                if isinstance(val, torch.Tensor):
                    processed_inputs[k] = val.unsqueeze(0).to(self.model.device)
                elif k == "graph":
                    processed_inputs[k] = val.to(self.model.device)
                    processed_inputs[k]["batch"] = torch.zeros(
                        val["x"].size(0), dtype=torch.long, device=self.model.device
                    )

            action, raw_output = self.model.get_action(processed_inputs, return_raw_output=True)

            if np.all(action.left_pick[0] >= 0) and np.all(action.left_place[0] >= 0):
                left_pick_pos = self.env.get_world_coord_from_pixel(action.left_pick[0], depth)
                left_place_pos = self.env.get_world_coord_from_pixel(action.left_place[0], depth)
            else:
                left_pick_pos = left_place_pos = None

            if np.all(action.right_pick[0] >= 0) and np.all(action.right_place[0] >= 0):
                right_pick_pos = self.env.get_world_coord_from_pixel(action.right_pick[0], depth)
                right_place_pos = self.env.get_world_coord_from_pixel(action.right_place[0], depth)
                if left_pick_pos is not None and left_place_pos is not None:
                    self.env.pick_and_place_dual(
                        pick_pos_left=left_pick_pos,
                        place_pos_left=left_place_pos,
                        pick_pos_right=right_pick_pos,
                        place_pos_right=right_place_pos,
                    )
                else:
                    self.env.pick_and_place_single(
                        pick_pos=right_pick_pos, place_pos=right_place_pos
                    )
            else:
                assert left_pick_pos is not None and left_place_pos is not None
                self.env.pick_and_place_single(pick_pos=left_pick_pos, place_pos=left_place_pos)

            particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]

            old_rgb = rgb.copy()
            old_mask = mask.copy()
            old_depth = depth.copy()
            context.append({"rgb": old_rgb, "mask": old_mask, "depth": old_depth})

            rgb, depth = self.env.render_image()
            mask = get_mask_from_depth(depth)

            self.task = sample_name.split("_")[1]

            if self.visualize_predictions:
                visualizations = visualize_action(processed_inputs, action)
                assert len(visualizations) == 1
                self.save_visuals(
                    f"{sample_name}_{instruction}.png",
                    viz=visualizations[0],
                    particle_pos=particle_pos,
                )
                if raw_output is not None:
                    self.save_visuals(
                        f"{sample_name}_{instruction}.png",
                        rgb=old_rgb,
                        left_pick_heatmap=raw_output["left_pick_heatmap"],
                        right_pick_heatmap=raw_output["right_pick_heatmap"],
                        left_place_heatmap=raw_output["left_place_heatmap"],
                        right_place_heatmap=raw_output["right_place_heatmap"],
                        mask_heatmap=raw_output.get("mask_heatmap"),
                    )

        if self.visualize_predictions:
            out_dir = os.path.join(
                "eval",
                "softgym_rollout",
                self.task,
                f"{sample_name}",
            )
            os.makedirs(out_dir, exist_ok=True)
            for frame_idx, frame in enumerate(self.env.frames):
                Image.fromarray(frame).save(os.path.join(out_dir, f"{frame_idx:05d}.png"))

    def save_visuals(self, out_file_name, **kwargs):
        if self.visualize_predictions:
            save_predictions(
                out_folder=os.path.join("eval", "softgym_rollout", self.task),
                out_file_name=out_file_name,
                **kwargs,
            )


class SoftgymBimanualRolloutEvaluatorDeng(SoftgymEvaluator):
    instructions = {
        "TshirtFold": [
            "Fold the Tshirt in half, left ro right.",
            "Fold the Tshirt in half, top to bottom.",
        ],
        "TrousersFold": [
            "Fold the Trousers in half, left to right.",
            "Fold the Trousers in half, top to bottom.",
        ],
    }

    def evaluate(
        self,
        cloth_type=None,
        cloth3d=None,
        num_evals: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs,
    ):
        if cloth_type is None:
            assert task is not None
            cloth_type = task_to_cloth_type[task]
            cloth3d = False if (cloth_type == "Square" or cloth_type == "Rectangular") else True
        else:
            assert cloth3d is not None

        super().evaluate(cloth_type=cloth_type, cloth3d=cloth3d)

        assert num_evals is not None
        for i in trange(num_evals, desc=f"Evaluating {task}"):
            rand_idx = np.random.randint(len(self.cached_configs))
            config = self.cached_configs[rand_idx]
            state = self.cached_states[rand_idx]

            self.reset(
                config=config,
                state=state,
                task=task,
                random_angle=0,
            )

            assert task is not None
            self.execute_model(
                instructions=self.instructions[task],
                trial_index=i,
            )

    def execute_model(self, instructions, trial_index):
        rgb, depth = self.env.render_image()
        mask = get_mask_from_depth(depth)
        context = []

        for instruction in instructions:
            processed_inputs = self.processor(
                depth=depth,
                rgb=rgb,
                mask=mask,
                context=context,
                instruction=instruction,
                matrix_world_to_camera=self.env.camera_matrix,
                K=self.K,
            )

            for k, val in processed_inputs.items():
                if isinstance(val, torch.Tensor):
                    processed_inputs[k] = val.unsqueeze(0).to(self.model.device)
                elif k == "graph":
                    processed_inputs[k] = val.to(self.model.device)
                    processed_inputs[k]["batch"] = torch.zeros(
                        val["x"].size(0), dtype=torch.long, device=self.model.device
                    )

            action, raw_output = self.model.get_action(processed_inputs, return_raw_output=True)

            if np.all(action.left_pick[0] >= 0) and np.all(action.left_place[0] >= 0):
                left_pick_pos = self.env.get_world_coord_from_pixel(action.left_pick[0], depth)
                left_place_pos = self.env.get_world_coord_from_pixel(action.left_place[0], depth)
            else:
                left_pick_pos = left_place_pos = None

            if np.all(action.right_pick[0] >= 0) and np.all(action.right_place[0] >= 0):
                right_pick_pos = self.env.get_world_coord_from_pixel(action.right_pick[0], depth)
                right_place_pos = self.env.get_world_coord_from_pixel(action.right_place[0], depth)
                if left_pick_pos is not None and left_place_pos is not None:
                    self.env.pick_and_place_dual(
                        pick_pos_left=left_pick_pos,
                        place_pos_left=left_place_pos,
                        pick_pos_right=right_pick_pos,
                        place_pos_right=right_place_pos,
                    )
                else:
                    self.env.pick_and_place_single(
                        pick_pos=right_pick_pos, place_pos=right_place_pos
                    )
            else:
                assert left_pick_pos is not None and left_place_pos is not None
                self.env.pick_and_place_single(pick_pos=left_pick_pos, place_pos=left_place_pos)

            old_rgb = rgb.copy()
            old_mask = mask.copy()
            old_depth = depth.copy()
            context.append({"rgb": old_rgb, "mask": old_mask, "depth": old_depth})

            rgb, depth = self.env.render_image()
            mask = get_mask_from_depth(depth)

            if self.visualize_predictions:
                visualizations = visualize_action(processed_inputs, action)
                assert len(visualizations) == 1
                self.save_visuals(
                    f"{trial_index}_{instruction}.png",
                    viz=visualizations[0],
                )
                if raw_output is not None:
                    self.save_visuals(
                        f"{trial_index}_{instruction}.png",
                        rgb=old_rgb,
                        left_pick_heatmap=raw_output["left_pick_heatmap"],
                        right_pick_heatmap=raw_output["right_pick_heatmap"],
                        left_place_heatmap=raw_output["left_place_heatmap"],
                        right_place_heatmap=raw_output["right_place_heatmap"],
                        mask_heatmap=raw_output.get("mask_heatmap"),
                    )

        if self.visualize_predictions:
            out_dir = os.path.join(
                "eval",
                "softgym_rollout_deng",
                self.task,
                f"{trial_index}",
            )
            os.makedirs(out_dir, exist_ok=True)
            for frame_idx, frame in enumerate(self.env.frames):
                Image.fromarray(frame).save(os.path.join(out_dir, f"{frame_idx:05d}.png"))

    def save_visuals(self, out_file_name, **kwargs):
        if self.visualize_predictions:
            save_predictions(
                out_folder=os.path.join("eval", "softgym_rollout", self.task),
                out_file_name=out_file_name,
                **kwargs,
            )
