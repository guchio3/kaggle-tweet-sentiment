class pass_scheduler:
    def __init__(self):
        self._state_dict = {}

    def step(self):
        pass

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict
