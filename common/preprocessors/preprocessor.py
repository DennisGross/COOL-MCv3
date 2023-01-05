class Preprocessor:

    def __init__(self, state_mapper, config_str):
        self.state_mapper = state_mapper
        self.config_str = config_str
        self.buffer = {}

    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        raise NotImplementedError()

    def preprocess(self, rl_agent, state:np.ndarray, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param state: The state.
        :return: The preprocessed state.
        """
        raise NotImplementedError()

    def save(self):
        """
        Saves the preprocessor in the MLFlow experiment.
        """
        pass

    def load(self, root_folder:str):
        """
        Loads the preprocessor from the folder

        Args:
            root_folder ([str]): Path to the folder
        """
        pass

