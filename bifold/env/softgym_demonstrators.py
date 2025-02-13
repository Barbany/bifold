import random


class CornerFold:  # sequence
    def __init__(self):
        self.gammas = [0.9] * 4
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.1
        self.primitives = ["single"] * 4
        self.unseen_flags = [0, 0, 0, 0]

        self.seen_lang_templates = [
            "Fold the {which} corner of the fabric towards the center.",
            "Bring the {which} corner of the cloth to the middle with a fold.",
            "Create a fold from the {which} corner of the fabric towards the center.",
            "Make a crease at the {which} corner of the cloth and fold it inwards.",
            "Fold the {which} corner of the cloth towards the center.",
            "Bring the {which} corner of the fabric to the middle with a fold.",
            "Create a fold from the {which} corner of the cloth towards the center.",
            "Make a crease at the {which} corner of the fabric and fold it inwards.",
            "Fold the {which} corner of the cloth towards the center.",
            "Bring the {which} corner of the fabric to the center with a fold.",
            "Create a fold from the {which} corner of the cloth towards the center.",
            "Make a crease at the {which} corner of the fabric and fold it inwards.",
            "Fold the {which} corner of the cloth towards the middle.",
            "Bring the {which} corner of the fabric to the center with a fold.",
            "Create a fold from the {which} corner of the cloth towards the center.",
            "Make a crease at the {which} corner of the cloth and fold it inwards.",
        ]

        self.unseen_lang_templates = [
            "Fold the {which} corner of the fabric towards the midpoint.",
            "Bring the {which} corner of the cloth to the center with a fold.",
            "Create a fold from the {which} corner of the fabric towards the center.",
            "Make a crease at the {which} corner of the cloth and fold it towards the center.",
        ]

        self.seen_tasks = ["top_left", "top_right", "bottom_left"]
        self.unseen_tasks = ["bottom_right"]

        self.position_templates = {
            "top_left": [
                "upper left",
                "leftmost top",
                "topmost left",
                "left upper",
                "top left-hand",
                "left-hand top",
            ],
            "top_right": [
                "upper right",
                "rightmost top",
                "topmost right",
                "right upper",
                "top right-hand",
                "right-hand top",
            ],
            "bottom_left": [
                "lower left",
                "leftmost bottom",
                "bottommost left",
                "left lower",
                "bottom left-hand",
                "left-hand bottom",
            ],
            "bottom_right": [
                "lower right",
                "rightmost bottom",
                "bottommost right",
                "right lower",
                "bottom right-hand",
                "right-hand bottom",
            ],
        }

        # one-step action-instruction pairs
        self.act_templates = {
            "top_left": 0,
            "top_right": 2,
            "bottom_left": 6,
            "bottom_right": 8,
        }

    def get_action_instruction(self):
        self.unseen_flags = [0, 0, 0, 0]
        pick_corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        random.shuffle(pick_corners)
        pick_idxs = [self.act_templates[key] for key in pick_corners]
        place_idxs = [4, 4, 4, 4]
        instructions = []
        for i in range(4):
            pick_corner = pick_corners[i]
            if pick_corner in self.unseen_tasks:
                self.unseen_flags[i] = 1
            pick_position = random.choice(self.position_templates[pick_corner])
            lang = random.choice(self.seen_lang_templates).format(which=pick_position)
            instructions.append(lang)

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(self.gammas)
            == len(instructions)
            == len(self.primitives)
            == len(self.unseen_flags)
            == 4
        )
        return (
            pick_idxs,
            place_idxs,
            self.gammas,
            instructions,
            self.primitives,
            self.unseen_flags,
        )

    def get_eval_instruction(self):
        self.unseen_flags = [0, 0, 0, 0]
        pick_corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        random.shuffle(pick_corners)
        pick_idxs = [self.act_templates[key] for key in pick_corners]
        place_idxs = [4, 4, 4, 4]
        instructions = []
        unseen_instructions = []
        for i in range(4):
            pick_corner = pick_corners[i]
            if pick_corner in self.unseen_tasks:
                self.unseen_flags[i] = 1
            pick_position = random.choice(self.position_templates[pick_corner])
            lang = random.choice(self.seen_lang_templates).format(which=pick_position)
            instructions.append(lang)
            unseen_lang = random.choice(self.unseen_lang_templates).format(which=pick_position)
            unseen_instructions.append(unseen_lang)

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(self.gammas)
            == len(instructions)
            == len(self.primitives)
            == len(self.unseen_flags)
            == 4
        )

        eval_seen_instruction = {
            "pick": pick_idxs,
            "place": place_idxs,
            "gammas": self.gammas,
            "flags": self.unseen_flags,
            "instructions": instructions,
        }

        eval_unseen_instruction = {
            "pick": pick_idxs,
            "place": place_idxs,
            "gammas": self.gammas,
            "flags": self.unseen_flags,
            "instructions": unseen_instructions,
        }

        eval_unseen_tasks = {
            "pick": pick_idxs,
            "place": place_idxs,
            "gammas": self.gammas,
            "flags": self.unseen_flags,
            "instructions": instructions,
        }

        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


class TriangleFold:  # choose
    def __init__(self):
        self.gammas = [1.0] * 2
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.1
        self.primitives = ["single"] * 2
        self.unseen_flags = [0, 0]

        self.seen_lang_templates1 = [
            "Fold the {which} corner of the fabric to its diagonal corner.",
            "Fold the {which} corner of the cloth to its opposite point.",
            (
                "Take the {which} corner of the material and fold it "
                "to the corner on the opposite side."
            ),
            "Fold the {which} corner of the cloth to its diagonal counterpart.",
            "Fold the {which} point of the fabric to its opposite vertex.",
            "Take the corner at the {which} of the cloth and fold it to its opposite corner.",
            "Fold the {which} corner of the material to the corner on the opposite side.",
            "Fold the corner at the {which} of the fabric to its opposite corner.",
        ]

        self.unseen_lang_templates1 = [
            (
                "Take the corner at the {which} of the cloth and fold it to "
                "the corner on the opposite side."
            ),
            (
                "Bring the {which} corner of the cloth to its opposite corner "
                "by folding it diagonally."
            ),
        ]

        self.seen_lang_templates2 = [
            "Fold the {which1} corner of the fabric towards the {which2}.",
            "Bring the {which1} corner of the cloth to the {which2} corner.",
            "Make a fold from the {which1} corner of the fabric to the {which2}.",
            "Fold the {which1} corner of the cloth towards the {which2} corner.",
            "Create a diagonal fold by folding the {which1} corner of the fabric to the {which2}.",
            "Fold the {which1} corner of the cloth to meet the {which2} corner.",
            "Fold the {which1} corner of the fabric downwards to the {which2} corner.",
            "Create a triangle by folding the {which1} corner of the cloth to the {which2} corner.",
        ]

        self.unseen_lang_templates2 = [
            "Fold the {which1} corner of the fabric towards the {which2} in a diagonal shape.",
            (
                "Bring the {which1} corner of the cloth down to the {which2} corner, "
                "folding it in half diagonally."
            ),
        ]

        self.seen_tasks = [
            ["top_left", "top_right"],
            ["top_left", "bottom_left"],
            ["top_right", "top_left"],
            ["bottom_left", "bottom_right"],
            ["bottom_left", "top_left"],
            ["bottom_right", "bottom_left"],
        ]

        self.unseen_tasks = [
            ["top_right", "bottom_right"],
            ["bottom_right", "top_right"],
        ]

        self.position_templates = {
            "top_left": [
                "upper left",
                "leftmost top",
                "topmost left",
                "left upper",
                "top left-hand",
                "left-hand top",
            ],
            "top_right": [
                "upper right",
                "rightmost top",
                "topmost right",
                "right upper",
                "top right-hand",
                "right-hand top",
            ],
            "bottom_left": [
                "lower left",
                "leftmost bottom",
                "bottommost left",
                "left lower",
                "bottom left-hand",
                "left-hand bottom",
            ],
            "bottom_right": [
                "lower right",
                "rightmost bottom",
                "bottommost right",
                "right lower",
                "bottom right-hand",
                "right-hand bottom",
            ],
        }

        self.corner_pairs = {
            "top_left": "bottom_right",
            "top_right": "bottom_left",
            "bottom_left": "top_right",
            "bottom_right": "top_left",
        }

        # one-step action-instruction pairs
        self.act_templates = {
            "top_left": 0,
            "top_right": 2,
            "bottom_left": 6,
            "bottom_right": 8,
        }

    def get_action_instruction(self):
        pick_corners_list = [
            ["top_left", "top_right"],
            ["top_left", "bottom_left"],
            ["top_right", "top_left"],
            ["top_right", "bottom_right"],
            ["bottom_left", "bottom_right"],
            ["bottom_left", "top_left"],
            ["bottom_right", "bottom_left"],
            ["bottom_right", "top_right"],
        ]
        self.unseen_flags = [0, 0]

        pick_corners = random.choice(pick_corners_list)
        pick_idxs = [self.act_templates[key] for key in pick_corners]
        place_idxs = [self.act_templates[self.corner_pairs[key]] for key in pick_corners]

        instructions = []
        if pick_corners in self.unseen_tasks:
            self.unseen_flags = [1, 1]
        for pick_corner in pick_corners:
            prob = random.uniform(0, 1)
            pick_position = random.choice(self.position_templates[pick_corner])
            if prob < 0.5:
                lang = random.choice(self.seen_lang_templates1).format(which=pick_position)
            else:
                place_position = random.choice(
                    self.position_templates[self.corner_pairs[pick_corner]]
                )
                lang = random.choice(self.seen_lang_templates2).format(
                    which1=pick_position, which2=place_position
                )
            instructions.append(lang)

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(self.gammas)
            == len(instructions)
            == len(self.primitives)
            == len(self.unseen_flags)
            == 2
        )
        return (
            pick_idxs,
            place_idxs,
            self.gammas,
            instructions,
            self.primitives,
            self.unseen_flags,
        )

    def get_eval_instruction(self):

        instructions = []
        unseen_instructions = []
        seen_pick_corners = random.choice(self.seen_tasks)
        seen_pick_idxs = [self.act_templates[key] for key in seen_pick_corners]
        seen_place_idxs = [self.act_templates[self.corner_pairs[key]] for key in seen_pick_corners]
        seen_flags = [0, 0]

        for pick_corner in seen_pick_corners:
            prob = random.uniform(0, 1)
            pick_position = random.choice(self.position_templates[pick_corner])
            if prob < 0.5:
                unseen_lang = random.choice(self.unseen_lang_templates1).format(which=pick_position)
                seen_lang = random.choice(self.seen_lang_templates1).format(which=pick_position)
            else:
                place_position = random.choice(
                    self.position_templates[self.corner_pairs[pick_corner]]
                )
                unseen_lang = random.choice(self.unseen_lang_templates2).format(
                    which1=pick_position, which2=place_position
                )
                seen_lang = random.choice(self.seen_lang_templates2).format(
                    which1=pick_position, which2=place_position
                )
            instructions.append(seen_lang)
            unseen_instructions.append(unseen_lang)

        unseen_tasks_instructions = []
        unseen_pick_corners = random.choice(self.unseen_tasks)
        unseen_pick_idxs = [self.act_templates[key] for key in unseen_pick_corners]
        unseen_place_idxs = [
            self.act_templates[self.corner_pairs[key]] for key in unseen_pick_corners
        ]
        unseen_flags = [1, 1]

        for pick_corner in unseen_pick_corners:
            prob = random.uniform(0, 1)
            pick_position = random.choice(self.position_templates[pick_corner])
            if prob < 0.5:
                seen_lang = random.choice(self.seen_lang_templates1).format(which=pick_position)
            else:
                place_position = random.choice(
                    self.position_templates[self.corner_pairs[pick_corner]]
                )
                seen_lang = random.choice(self.seen_lang_templates2).format(
                    which1=pick_position, which2=place_position
                )
            unseen_tasks_instructions.append(seen_lang)

        assert (
            len(seen_pick_idxs)
            == len(unseen_pick_idxs)
            == len(unseen_instructions)
            == len(instructions)
            == len(unseen_tasks_instructions)
            == 2
        )

        eval_seen_instruction = {
            "pick": seen_pick_idxs,
            "place": seen_place_idxs,
            "gammas": self.gammas,
            "flags": seen_flags,
            "instructions": instructions,
        }

        eval_unseen_instruction = {
            "pick": seen_pick_idxs,
            "place": seen_place_idxs,
            "gammas": self.gammas,
            "flags": seen_flags,
            "instructions": unseen_instructions,
        }

        eval_unseen_tasks = {
            "pick": unseen_pick_idxs,
            "place": unseen_place_idxs,
            "gammas": self.gammas,
            "flags": unseen_flags,
            "instructions": unseen_tasks_instructions,
        }

        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


class StraightFold:  # choose
    def __init__(self):
        self.gammas = [0.9, 0.9, 1.0]
        self.pick_speed = 0.006
        self.move_speed = 0.006
        self.place_speed = 0.005
        self.lift_height = 0.125
        self.primitives = ["multi", "multi", "single"]
        self.unseen_flags = [0, 0, 0]

        self.seen_lang_templates1 = [
            "Crease the cloth in half from {which1} to {which2}.",
            "Make a fold in the cloth from {which1} to {which2}.",
            "Create a crease in the cloth from {which1} to {which2}.",
            "Create a fold in the cloth by halving it from {which1} to {which2}.",
            "Create a central fold in the cloth by folding it in half from {which1} to {which2}.",
            "Make a crease down the middle of the cloth from {which1} to {which2}.",
            (
                "Bring the {which1} and {which2} sides of the cloth together "
                "to make a fold down the middle."
            ),
            "Halve the cloth by folding it from {which1} to {which2}.",
        ]

        self.unseen_lang_templates1 = [
            "Make a fold in the cloth by halving it from {which1} to {which2}.",
            "Fold the cloth in half, starting from the {which1} side and meeting the {which2}.",
        ]

        self.seen_lang_templates2 = [
            "Fold the fabric in half, starting from the {which} side.",
            "Bend the material in half, beginning from the {which} side.",
            "Fold the textile symmetrically, starting on the {which}.",
            "Fold the cloth in half, beginning from the {which} edge.",
            "Fold the fabric in half, starting from the {which} part.",
            "Fold the fabric in half, beginning from the {which} side.",
            "Fold the textile equally, starting from the {which} side.",
            "Fold the material in half, symmetrically starting from the {which}.",
        ]

        self.unseen_lang_templates2 = [
            "Fold the cloth in half, starting on the {which} side.",
            "Fold the material equally, beginning from the {which} edge.",
        ]

        self.position_templates = {
            "left": ["left", "leftmost", "left-hand"],
            "right": ["right", "rightmost", "right-hand"],
            "up": ["upper", "top", "uppermost"],
            "down": ["lower", "bottom", "lowermost"],
        }

        self.seen_tasks = ["left", "right", "up"]
        self.unseen_tasks = ["down"]

        self.edge_pairs = {"left": "right", "right": "left", "up": "down", "down": "up"}

        self.act_templates = [
            # -45 - 45 mode 0
            {"up": [0, 2], "down": [6, 8], "left": 3, "right": 5},
            # 45 - 90 mode 1
            {"left": [0, 2], "right": [6, 8], "up": 5, "down": 3},
            # -90 - -45 mode 2
            {"left": [6, 8], "right": [0, 2], "up": 3, "down": 5},
        ]

    def get_action_instruction(self, random_angle):
        angle_mode = 0
        angle_mode = int(abs(random_angle) > 45) + int(random_angle < -45)

        if angle_mode > 0:
            pick_edges_list = [
                ["left", "up"],
                ["left", "down"],
                ["right", "up"],
                ["right", "down"],
            ]
        else:
            pick_edges_list = [
                ["up", "left"],
                ["up", "right"],
                ["down", "left"],
                ["down", "right"],
            ]

        act_templates_local = self.act_templates[angle_mode]
        self.unseen_flags = [0, 0, 0]

        pick_idxs = []
        place_idxs = []
        instructions = []

        pick_edges = random.choice(pick_edges_list)

        # multi step
        multi_action = pick_edges[0]
        for i in range(2):
            pick_idxs.append(act_templates_local[multi_action][i])
            place_idxs.append(act_templates_local[self.edge_pairs[multi_action]][i])

            pick_position = random.choice(self.position_templates[multi_action])
            place_position = random.choice(self.position_templates[self.edge_pairs[multi_action]])
            lang = random.choice(self.seen_lang_templates1).format(
                which1=pick_position, which2=place_position
            )
            instructions.append(lang)

        # single step
        single_action = pick_edges[1]
        pick_idxs.append(act_templates_local[single_action])
        place_idxs.append(act_templates_local[self.edge_pairs[single_action]])

        pick_position = random.choice(self.position_templates[single_action])
        lang = random.choice(self.seen_lang_templates2).format(which=pick_position)

        instructions.append(lang)

        if pick_edges[0] in self.unseen_tasks:
            self.unseen_flags[0] = 1
            self.unseen_flags[1] = 1

        if pick_edges[1] in self.unseen_tasks:
            self.unseen_flags[2] = 1

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(self.gammas)
            == len(instructions)
            == len(self.primitives)
            == len(self.unseen_flags)
            == 3
        )
        return (
            pick_idxs,
            place_idxs,
            self.gammas,
            instructions,
            self.primitives,
            self.unseen_flags,
        )

    def get_eval_instruction(self, angle_mode):  # bug
        if angle_mode > 0:
            seen_pick_edges_list = [["left", "up"], ["right", "up"]]
            unseen_pick_edges_list = [["left", "down"], ["right", "down"]]
        else:
            seen_pick_edges_list = [["up", "left"], ["up", "right"]]
            unseen_pick_edges_list = [["down", "left"], ["down", "right"]]

        act_templates_local = self.act_templates[angle_mode]

        seen_flags = [0, 0, 0]
        seen_pick_idxs = []
        seen_place_idxs = []
        instructions = []
        unseen_instructions = []
        seen_pick_edges = random.choice(seen_pick_edges_list)

        # multi step
        seen_multi_action = seen_pick_edges[0]
        for i in range(2):
            seen_pick_idxs.append(act_templates_local[seen_multi_action][i])
            seen_place_idxs.append(act_templates_local[self.edge_pairs[seen_multi_action]][i])

            pick_position = random.choice(self.position_templates[seen_multi_action])
            place_position = random.choice(
                self.position_templates[self.edge_pairs[seen_multi_action]]
            )
            seen_lang = random.choice(self.seen_lang_templates1).format(
                which1=pick_position, which2=place_position
            )
            unseen_lang = random.choice(self.unseen_lang_templates1).format(
                which1=pick_position, which2=place_position
            )
            instructions.append(seen_lang)
            unseen_instructions.append(unseen_lang)

        # single step
        seen_single_action = seen_pick_edges[1]
        seen_pick_idxs.append(act_templates_local[seen_single_action])
        seen_place_idxs.append(act_templates_local[self.edge_pairs[seen_single_action]])

        pick_position = random.choice(self.position_templates[seen_single_action])
        seen_lang = random.choice(self.seen_lang_templates2).format(which=pick_position)
        unseen_lang = random.choice(self.unseen_lang_templates2).format(which=pick_position)

        instructions.append(seen_lang)
        unseen_instructions.append(unseen_lang)

        unseen_flags = [0, 0, 0]
        unseen_pick_idxs = []
        unseen_place_idxs = []
        unseen_tasks_instructions = []
        unseen_pick_edges = random.choice(unseen_pick_edges_list)

        # multi step
        unseen_multi_action = unseen_pick_edges[0]
        for i in range(2):
            unseen_pick_idxs.append(act_templates_local[unseen_multi_action][i])
            unseen_place_idxs.append(act_templates_local[self.edge_pairs[unseen_multi_action]][i])
            pick_position = random.choice(self.position_templates[unseen_multi_action])
            place_position = random.choice(
                self.position_templates[self.edge_pairs[unseen_multi_action]]
            )
            seen_lang = random.choice(self.seen_lang_templates1).format(
                which1=pick_position, which2=place_position
            )
            unseen_tasks_instructions.append(seen_lang)

        # single step
        unseen_single_action = unseen_pick_edges[1]
        unseen_pick_idxs.append(act_templates_local[unseen_single_action])
        unseen_place_idxs.append(act_templates_local[self.edge_pairs[unseen_single_action]])

        pick_position = random.choice(self.position_templates[unseen_single_action])
        seen_lang = random.choice(self.seen_lang_templates2).format(which=pick_position)
        unseen_tasks_instructions.append(seen_lang)

        if unseen_multi_action in self.unseen_tasks:
            unseen_flags[0] = 1
            unseen_flags[1] = 1

        if unseen_single_action in self.unseen_tasks:
            unseen_flags[2] = 1

        assert (
            len(seen_pick_idxs)
            == len(unseen_pick_idxs)
            == len(unseen_instructions)
            == len(instructions)
            == len(unseen_tasks_instructions)
            == 3
        )

        eval_seen_instruction = {
            "pick": seen_pick_idxs,
            "place": seen_place_idxs,
            "gammas": self.gammas,
            "flags": seen_flags,
            "instructions": instructions,
        }

        eval_unseen_instruction = {
            "pick": seen_pick_idxs,
            "place": seen_place_idxs,
            "gammas": self.gammas,
            "flags": seen_flags,
            "instructions": unseen_instructions,
        }

        eval_unseen_tasks = {
            "pick": unseen_pick_idxs,
            "place": unseen_place_idxs,
            "gammas": self.gammas,
            "flags": unseen_flags,
            "instructions": unseen_tasks_instructions,
        }

        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


class TshirtFold:  # sequence
    def __init__(self):
        self.gammas = [1.0, 1.0, 1.1, 1.1]
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.125
        self.primitives = ["single", "single", "multi", "multi"]
        self.unseen_flags = [0, 0, 0, 0]

        self.seen_lang_templates1 = [
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
        ]

        self.unseen_lang_templates1 = [
            "Fold the {which} sleeve towards the midpoint of the shirt.",
            "Bring the {which} sleeve to the center seam.",
            "Fold the {which} sleeve to the centerline of the shirt.",
            "Fold the {which} sleeve to the centerline of the shirt.",
        ]

        self.seen_lang_templates2 = [
            "Bring the bottom of the T-shirt up towards the neckline.",
            "Fold the shirt's hem up towards the top.",
            "Flip the bottom of the T-shirt towards the top.",
            "Roll the bottom of the T-shirt up towards the top.",
            "Fold the lower part of the T-shirt towards the top.",
            "Tuck the bottom of the T-shirt upwards.",
            "Fold the lower edge of the T-shirt up to the top.",
            "Raise the bottom of the T-shirt to the top.",
            "Fold the shirt's tail up towards the neckline.",
            "Lift the bottom of the T-shirt towards the top.",
            "Fold the hem of the T-shirt towards the top.",
            "Turn up the bottom of the T-shirt towards the top.",
            "Crease the bottom of the T-shirt towards the top.",
            "Bring the lower part of the T-shirt up towards the neckline.",
            "Fold the shirt's bottom edge towards the top.",
            "Flip up the bottom of the T-shirt towards the top.",
        ]

        self.unseen_lang_templates2 = [
            "Fold the shirt's tail end towards the top.",
            "Roll up the bottom of the T-shirt towards the top.",
            "Fold the bottom of the T-shirt towards the top edge.",
            "Fold the shirt's lower portion up towards the neckline.",
        ]

        self.seen_tasks = ["right"]
        self.unseen_tasks = ["left"]

        self.position_templates = {
            "left": ["left", "leftmost", "left-hand"],
            "right": ["right", "rightmost", "right-hand"],
        }

        # one-step action-instruction pairs
        self.single_templates = {
            "left": [2, 3],
            "right": [5, 4],
        }

        self.multi_templates = {
            "upwards": [[6, 7], [0, 1]],
            "left-to-right": [[0, 6], [1, 7]],
            "right-to-left": [[1, 7], [0, 6]],
        }

    def get_action_instruction(self):
        single_list = ["left", "right"]
        random.shuffle(single_list)
        multi_action = "upwards"
        pick_idxs, place_idxs = [], []
        instructions = []
        self.unseen_flags = [0, 0, 0, 0]

        # single action
        for i in range(2):
            action = single_list[i]
            if action in self.unseen_tasks:
                self.unseen_flags[i] = 1
            pick_idx, place_idx = (
                self.single_templates[action][0],
                self.single_templates[action][1],
            )
            pick_idxs.append(pick_idx)
            place_idxs.append(place_idx)
            lang = random.choice(self.seen_lang_templates1).format(
                which=random.choice(self.position_templates[action])
            )
            instructions.append(lang)

        # multi action
        pick_idx, place_idx = (
            self.multi_templates[multi_action][0],
            self.multi_templates[multi_action][1],
        )
        for i in range(2):
            pick_idxs.append(pick_idx[i])
            place_idxs.append(place_idx[i])
            lang = random.choice(self.seen_lang_templates2)
            instructions.append(lang)

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(self.gammas)
            == len(instructions)
            == len(self.primitives)
            == len(self.unseen_flags)
            == 4
        )
        return (
            pick_idxs,
            place_idxs,
            self.gammas,
            instructions,
            self.primitives,
            self.unseen_flags,
        )

    def get_eval_instruction(self):
        single_list = ["left", "right"]
        random.shuffle(single_list)
        multi_action = "upwards"
        pick_idxs, place_idxs = [], []
        instructions = []
        unseen_instructions = []
        self.unseen_flags = [0, 0, 0, 0]

        # single action
        for i in range(2):
            action = single_list[i]
            if action in self.unseen_tasks:
                self.unseen_flags[i] = 1
            pick_idx, place_idx = (
                self.single_templates[action][0],
                self.single_templates[action][1],
            )
            pick_idxs.append(pick_idx)
            place_idxs.append(place_idx)
            lang = random.choice(self.seen_lang_templates1).format(
                which=random.choice(self.position_templates[action])
            )
            unseen_lang = random.choice(self.unseen_lang_templates1).format(
                which=random.choice(self.position_templates[action])
            )
            instructions.append(lang)
            unseen_instructions.append(unseen_lang)

        # multi action
        pick_idx, place_idx = (
            self.multi_templates[multi_action][0],
            self.multi_templates[multi_action][1],
        )
        for i in range(2):
            pick_idxs.append(pick_idx[i])
            place_idxs.append(place_idx[i])
            lang = random.choice(self.seen_lang_templates2)
            unseen_lang = random.choice(self.unseen_lang_templates2)
            instructions.append(lang)
            unseen_instructions.append(unseen_lang)

        eval_seen_instruction = {
            "pick": pick_idxs,
            "place": place_idxs,
            "gammas": self.gammas,
            "flags": self.unseen_flags,
            "instructions": instructions,
        }

        eval_unseen_instruction = {
            "pick": pick_idxs,
            "place": place_idxs,
            "gammas": self.gammas,
            "flags": self.unseen_flags,
            "instructions": unseen_instructions,
        }

        eval_unseen_tasks = {
            "pick": pick_idxs,
            "place": place_idxs,
            "gammas": self.gammas,
            "flags": self.unseen_flags,
            "instructions": instructions,
        }

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(self.gammas)
            == len(instructions)
            == len(unseen_instructions)
            == len(self.unseen_flags)
            == 4
        )
        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


class TrousersFold:  # choose
    def __init__(self):
        self.gammas = [1.0, 1.0, 1.0]
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.15
        self.primitives = ["multi", "multi", "single"]
        self.unseen_flags = [0, 0, 0]

        self.seen_lang_templates1 = [
            "Fold the Trousers in half, {which1} to {which2}.",
            "Fold the Trousers from the {which1} side towards the {which2} side.",
            "Fold the Trousers in half, starting from the {which1} and ending at the {which2}.",
            "Fold the Trousers, {which1} side over {which2} side.",
            "Bend the Trousers in half, from {which1} to {which2}.",
            "Fold the Trousers, making sure the {which1} side touches the {which2} side.",
            "Fold the Trousers, bringing the {which1} side to meet the {which2} side.",
            "Crease the Trousers down the middle, from {which1} to {which2}.",
            "Fold the Trousers in half horizontally, {which1} to {which2}.",
            "Make a fold in the Trousers, starting from the {which1} and ending at the {which2}.",
            "Fold the Trousers in half, aligning the {which1} and {which2} sides.",
            "Fold the Trousers, ensuring the {which1} side meets the {which2} side.",
            "Fold the Trousers, orientating from the {which1} towards the {which2}.",
            "Fold the Trousers cleanly, from the {which1} side to the {which2} side.",
            "Fold the Trousers in half, with the {which1} side overlapping the {which2}.",
            "Create a fold in the Trousers, going from {which1} to {which2}.",
        ]

        self.unseen_lang_templates1 = [
            (
                "Bring the {which1} side of the Trousers towards the {which2} side "
                "and fold them in half."
            ),
            "Fold the waistband of the Trousers in half, from {which1} to {which2}.",
            "Fold the Trousers neatly, from the {which1} side to the {which2} side.",
            "Fold the Trousers, making a crease from the {which1} to the {which2}.",
        ]

        self.seen_lang_templates2 = [
            "Fold the Trousers in half vertically from top to bottom.",
            "Create a fold in the Trousers from the waistband to the hem.",
            "Fold the Trousers along the vertical axis, starting from the top.",
            "Fold the Trousers in half lengthwise, beginning at the waistband.",
            "Fold the Trousers in half, bringing the top down to the hem.",
            "Fold the Trousers vertically, starting at the waistband.",
            "Fold the Trousers in half, starting from the top edge.",
            "Fold the Trousers by bringing the waistband down to meet the hem.",
            "Make a crease in the Trousers running from the top to the bottom.",
            "Fold the Trousers in half, starting from the waistband.",
            "Fold the Trousers in half by bringing the top down to the hem.",
            "Fold the Trousers in half lengthwise, starting from the waistband.",
            "Fold the Trousers in half vertically, starting at the upper edge.",
            "Fold the Trousers by bringing the waistband down to meet the bottom.",
            "Fold the Trousers in half, starting from the top seam.",
            "Fold the Trousers in half, bringing the top towards the hem.",
        ]

        self.unseen_lang_templates2 = [
            "Fold the Trousers lengthwise, starting at the waistband.",
            "Fold the Trousers in half vertically, beginning at the upper edge.",
            "Fold the Trousers in two, from top to bottom.",
            "Fold the Trousers in half, starting at the top of the garment.",
        ]

        self.seen_tasks = ["left"]
        self.unseen_tasks = ["right"]

        self.position_templates = {
            "left": ["left", "leftmost", "left-hand"],
            "right": ["right", "rightmost", "right-hand"],
        }
        self.position_pairs = {"left": "right", "right": "left"}

        self.dual_templates = {"left": [[0, 4], [3, 7]], "right": [[3, 7], [0, 4]]}

        self.single_templates = {
            "left": [2, 6],
            "right": [1, 5],
        }

    def get_action_instruction(self):
        self.unseen_flags = [0, 0, 0]
        fold_action = random.choice(["left", "right"])
        if fold_action in self.unseen_tasks:
            self.unseen_flags = [1, 1, 0]

        pick_idxs, place_idxs = [], []
        instructions = []

        # multi action
        pick_idx, place_idx = (
            self.dual_templates[fold_action][0],
            self.dual_templates[fold_action][1],
        )
        for i in range(2):
            pick_idxs.append(pick_idx[i])
            place_idxs.append(place_idx[i])
            lang = random.choice(self.seen_lang_templates1).format(
                which1=random.choice(self.position_templates[fold_action]),
                which2=random.choice(self.position_templates[self.position_pairs[fold_action]]),
            )
            instructions.append(lang)

        # single action
        lang = random.choice(self.seen_lang_templates2)
        pick_idxs.append(self.single_templates[fold_action][0])
        place_idxs.append(self.single_templates[fold_action][1])
        instructions.append(lang)

        assert (
            len(pick_idxs)
            == len(place_idxs)
            == len(self.gammas)
            == len(instructions)
            == len(self.primitives)
            == len(self.unseen_flags)
            == 3
        )
        return (
            pick_idxs,
            place_idxs,
            self.gammas,
            instructions,
            self.primitives,
            self.unseen_flags,
        )

    def get_eval_instruction(self):

        instructions = []
        unseen_instructions = []
        seen_pick_idxs, seen_place_idxs = [], []
        seen_fold_action = "left"
        seen_flags = [0, 0, 0]

        pick_idxs, place_idxs = (
            self.dual_templates[seen_fold_action][0],
            self.dual_templates[seen_fold_action][1],
        )
        for i in range(2):
            unseen_lang = random.choice(self.unseen_lang_templates1).format(
                which1=random.choice(self.position_templates[seen_fold_action]),
                which2=random.choice(
                    self.position_templates[self.position_pairs[seen_fold_action]]
                ),
            )
            lang = random.choice(self.seen_lang_templates1).format(
                which1=random.choice(self.position_templates[seen_fold_action]),
                which2=random.choice(
                    self.position_templates[self.position_pairs[seen_fold_action]]
                ),
            )
            instructions.append(lang)
            unseen_instructions.append(unseen_lang)
            seen_pick_idxs.append(pick_idxs[i])
            seen_place_idxs.append(place_idxs[i])

        seen_pick_idxs.append(self.single_templates[seen_fold_action][0])
        seen_place_idxs.append(self.single_templates[seen_fold_action][1])
        unseen_instructions.append(random.choice(self.unseen_lang_templates2))
        instructions.append(random.choice(self.seen_lang_templates2))

        unseen_tasks_instructions = []
        unseen_pick_idxs, unseen_place_idxs = [], []
        unseen_fold_action = "right"
        unseen_flags = [1, 1, 0]
        pick_idxs, place_idxs = (
            self.dual_templates[unseen_fold_action][0],
            self.dual_templates[unseen_fold_action][1],
        )
        for i in range(2):
            lang = random.choice(self.seen_lang_templates1).format(
                which1=random.choice(self.position_templates[unseen_fold_action]),
                which2=random.choice(
                    self.position_templates[self.position_pairs[unseen_fold_action]]
                ),
            )
            unseen_tasks_instructions.append(lang)
            unseen_pick_idxs.append(pick_idxs[i])
            unseen_place_idxs.append(place_idxs[i])

        unseen_pick_idxs.append(self.single_templates[unseen_fold_action][0])
        unseen_place_idxs.append(self.single_templates[unseen_fold_action][1])
        unseen_tasks_instructions.append(random.choice(self.seen_lang_templates2))

        assert (
            len(seen_pick_idxs)
            == len(unseen_pick_idxs)
            == len(unseen_instructions)
            == len(instructions)
            == len(unseen_tasks_instructions)
            == 3
        )

        eval_seen_instruction = {
            "pick": seen_pick_idxs,
            "place": seen_place_idxs,
            "gammas": self.gammas,
            "flags": seen_flags,
            "instructions": instructions,
        }

        eval_unseen_instruction = {
            "pick": seen_pick_idxs,
            "place": seen_place_idxs,
            "gammas": self.gammas,
            "flags": seen_flags,
            "instructions": unseen_instructions,
        }

        eval_unseen_tasks = {
            "pick": unseen_pick_idxs,
            "place": unseen_place_idxs,
            "gammas": self.gammas,
            "flags": unseen_flags,
            "instructions": unseen_tasks_instructions,
        }

        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


Demonstrator = {
    "CornerFold": CornerFold,
    "TriangleFold": TriangleFold,
    "StraightFold": StraightFold,
    "TshirtFold": TshirtFold,
    "TrousersFold": TrousersFold,
    None: TshirtFold,
}
