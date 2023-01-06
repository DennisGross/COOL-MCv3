import os
from common.manipulators.random_done import *
'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class ManipulatorBuilder():


    @staticmethod
    def build_manipulator(manipulator_path, command_line_arguments,observation_space, action_space,state_mapper):
        #print('Build model with', model_root_folder_path, command_line_arguments)
        #print('Environment', observation_space.shape, action_space.n)
        try:
            state_dimension = observation_space.shape[0]
        except:
            state_dimension = 1
        manipulator = None
        manipulator_name = command_line_arguments['manipulator'].split(",")[0]
        if manipulator_name == "random_done":
            manipulator = RandomDone(state_mapper, command_line_arguments['manipulator'])
            manipulator.load(manipulator_path)

        return manipulator
