from common.preprocessors.preprocessor import Preprocessor
import numpy as np

class Denoiser(Preprocessor):

    def __init__(self, state_mapper, config_str):
        super().__init__(state_mapper, config_str)
        self.parse_config(config_str)
        self.counter = 0
        # Build model



    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        self.training_interval = 1000
        self.training_epochs = 10
        return float(config_str.split(';')[1])

    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        if deploy:
            # Use denoiser
            pass
        else:
            # Train denoiser
            if self.counter % self.training_interval == 0 and self.counter > 0:
                self.train(rl_agent, action_mapper)

            pass
        self.counter+=1
        return state / self.denominator

    def save(self):
        """
        Saves the preprocessor in the MLFlow experiment.
        """
        # Save model
        pass

    def load(self, root_folder:str):
        """
        Loads the preprocessor from the folder

        Args:
            root_folder ([str]): Path to the folder
        """
        # Load model if exists
        pass

