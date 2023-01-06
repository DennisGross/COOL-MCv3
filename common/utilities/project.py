from common.rl_agents.agent_builder import AgentBuilder
from common.preprocessors.preprocessor_builder import PreprocessorBuilder
from common.manipulators.manipulator_builder import ManipulatorBuilder
from common.utilities.mlflow_bridge import MlFlowBridge

class Project():

    def __init__(self, command_line_arguments):
        self.command_line_arguments = command_line_arguments
        self.mlflow_bridge = None
        self.agent = None
        self.preprocessor = None
        self.manipulator = None


    def init_mlflow_bridge(self, project_name, task, parent_run_id):
        self.mlflow_bridge = MlFlowBridge(project_name, task, parent_run_id)

    def load_saved_command_line_arguments(self):
        saved_command_line_arguments = self.mlflow_bridge.load_command_line_arguments()
        if saved_command_line_arguments != None:
            old_task = saved_command_line_arguments['task']
            try:
                del saved_command_line_arguments['prop']
            except:
                pass
            try:
                del saved_command_line_arguments['task']
            except:
                pass
            try:
                del saved_command_line_arguments['parent_run_id']
            except:
                pass
            try:
                del saved_command_line_arguments['constant_definitions']
            except:
                pass
            if self.command_line_arguments['preprocessor'] != '':
                # Only delete it if it is not set by the command line (in this case take new one)
                try:
                    del saved_command_line_arguments['preprocessor']
                except:
                    pass
            else:
                print("USE old preprocessor: " + saved_command_line_arguments['preprocessor'])
            try:
                del saved_command_line_arguments['epsilon']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_dec']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_min']
            except:
                pass
            try:
                del saved_command_line_arguments['seed']
            except:
                pass
            try:
                del saved_command_line_arguments['deploy']
            except:
                pass
            try:
                del saved_command_line_arguments['num_episodes']
            except:
                pass
            try:
                del saved_command_line_arguments['eval_interval']
            except:
                pass
            try:
                del saved_command_line_arguments['prop_type']
            except:
                pass
            try:
                del saved_command_line_arguments['manipulator']
            except:
                pass
            try:
                del saved_command_line_arguments['range_plotting']
            except:
                pass

            for key in saved_command_line_arguments.keys():
                self.command_line_arguments[key] = saved_command_line_arguments[key]



    def create_agent(self, command_line_arguments, observation_space, number_of_actions):
        agent = None
        try:
            model_folder_path = self.mlflow_bridge.get_agent_path()
            # Build agent with the model and the hyperparameters
            agent = AgentBuilder.build_agent(model_folder_path, command_line_arguments, observation_space, number_of_actions)
        except Exception as msg:
            # If Model was not saved
            agent = AgentBuilder.build_agent(None, command_line_arguments, observation_space, number_of_actions)
        self.agent = agent
        return self.agent

    def create_preprocessor(self, command_line_arguments, observation_space, number_of_actions, state_mapper):
        processor = None
        try:
            preprocessor_path = self.mlflow_bridge.get_agent_path().replace("model", "")
            print(model_folder_path)
            # Build agent with the model and the hyperparameters
            processor = PreprocessorBuilder.build_preprocessor(preprocessor_path, command_line_arguments, observation_space, number_of_actions, state_mapper)
        except Exception as msg:
            # If Model was not saved
            preprocessor = PreprocessorBuilder.build_preprocessor(None, command_line_arguments, observation_space, number_of_actions, state_mapper)
        self.preprocessor = preprocessor
        return self.preprocessor


    def create_manipulator(self, command_line_arguments, observation_space, number_of_actions, state_mapper):
        manipulator = None
        try:
            manipulator_path = self.mlflow_bridge.get_agent_path().replace("model", "")
            print(model_folder_path)
            # Build agent with the model and the hyperparameters
            manipulator = ManipulatorBuilder.build_manipulator(manipulator_path, command_line_arguments, observation_space, number_of_actions, state_mapper)
        except Exception as msg:
            # If Model was not saved
            manipulator = ManipulatorBuilder.build_manipulator(None, command_line_arguments, observation_space, number_of_actions, state_mapper)
        self.manipulator = manipulator
        return self.manipulator


    def save(self):
        # Agent
        self.agent.save()
        # Preprocessor
        if self.preprocessor != None:
            self.preprocessor.save()
        # Manipulator
        if self.manipulator != None:
            self.manipulator.save()
        # Save Command Line Arguments
        self.mlflow_bridge.save_command_line_arguments(self.command_line_arguments)

    def close(self):
        self.mlflow_bridge.close()
