import os
from common.postprocessors.random_done import *
'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class PostprocessorBuilder():


    @staticmethod
    def build_postprocessor(manipulator_path, command_line_arguments,observation_space, action_space,state_mapper):
        #print('Build model with', model_root_folder_path, command_line_arguments)
        #print('Environment', observation_space.shape, action_space.n)
        try:
            state_dimension = observation_space.shape[0]
        except:
            state_dimension = 1
        postprocessor = None
        postprocessor_name = command_line_arguments['postprocessor'].split(";")[0]
        if postprocessor_name == "random_done":
            postprocessor = RandomDone(state_mapper, command_line_arguments['postprocessor'])
            postprocessor.load(manipulator_path)

        return postprocessor
