class Manipulator:

    def __init__(self, config):
        self.config = config

    def manipulate(rl_agent, state, action, reward, next_state, done):
        raise NotImplementedError()


