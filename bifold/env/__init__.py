class Action:
    def __init__(self, **kwargs):
        if len(kwargs) == 2:
            self.init_action(**kwargs)
        elif len(kwargs) == 4:
            self.init_bimanual_action(**kwargs)
        else:
            raise ValueError(f"Cannot instantiate action with {kwargs}")

    def init_action(self, pick, place):
        self.pick = pick
        self.place = place

    def init_bimanual_action(self, left_pick, right_pick, left_place, right_place):
        self.left_pick = left_pick
        self.right_pick = right_pick

        self.left_place = left_place
        self.right_place = right_place
