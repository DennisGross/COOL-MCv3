import os
import sys
sys.path.insert(0, '../')
from common.utilities.helper import *
from common.utilities.training import *
from typing import Any, Dict
from common.utilities.project import Project
from common.safe_gym.safe_gym import SafeGym

def prepare_prop(prop):
    prepared = False
    original_prop = prop
    if prop.find("max") == -1 and prop.find("min") == -1:
        query = prop
        # Insert min at second position
        operator_str = query[:1]
        min_part = "min"
        prop = operator_str + min_part + query[1:]
        prepared = True
    return prop, prepared, original_prop




if __name__ == '__main__':
    # Get command line arguments
    command_line_arguments = get_arguments()
    # Set seed
    set_random_seed(command_line_arguments['seed'])
    # Command line arguments set up
    command_line_arguments['task'] = RL_MODEL_CHECKING_TASK
    # Get full prism_file_path
    prism_file_path = os.path.join(
        command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])

    # Load last run, if specified
    if command_line_arguments['parent_run_id'] == "last":
        command_line_arguments['project_name'], command_line_arguments['parent_run_id'] = LastRunManager.read_last_run()
        print(f"Loaded last run {command_line_arguments['parent_run_id']} from project {command_line_arguments['project_name']}")

    # Project
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'], command_line_arguments['task'],
        command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()

    # Project Environment
    prism_file_path = os.path.join(
        m_project.command_line_arguments['prism_dir'], m_project.command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path, m_project.command_line_arguments['constant_definitions'],
                10, 1,
                  True,
                  m_project.command_line_arguments['seed'],
                  m_project.command_line_arguments['disabled_features'])

    # Create rest of project
    m_project.create_agent(m_project.command_line_arguments,
                           env.observation_space, env.action_space)
    m_project.create_preprocessor(m_project.command_line_arguments, env.observation_space, env.action_space, env.storm_bridge.state_mapper)
    m_project.create_manipulator(m_project.command_line_arguments, env.observation_space, env.action_space, env.storm_bridge.state_mapper)

    # Prepare property
    m_project.command_line_arguments['prop'], prepared, original_prop = prepare_prop(m_project.command_line_arguments['prop'])

    m_project.mlflow_bridge.set_property_query_as_run_name(
        m_project.command_line_arguments['prop'] + " for " + original_prop)

    # Model checking
    mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessor, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'])

    run_id = m_project.mlflow_bridge.get_run_id()
    print(f'{original_prop}:\t{mdp_reward_result}')
    print(f'Model Size:\t\t{model_checking_info["model_size"]}')
    print(f'Model Building Time:\t{model_checking_info["model_building_time"]}')
    print(f'Model Checking Time:\t{model_checking_info["model_checking_time"]}')
    print("Run ID: " + run_id)
    LastRunManager.write_last_run(m_project.command_line_arguments['project_name'], run_id)
    m_project.close()
