# COOL-MC




## Architecture
We will first describe the main components of COOL-MC and then delve into the details.

### Training
The following code runs an agent in an environment. The env variable represents the environment, and the reset method is called to reset the environment to its initial state. The step method is used to take an action in the environment, and observe the resulting next state, reward, and whether the episode has finished (indicated by the done flag).

The Preprocessor object preprocesses the raw state. Preprocessing the raw state is often necessary in reinforcement learning because the raw state may be difficult for the agent to work with directly. Preprocessing can also be used to apply adversarial attacks and countermeasure simulations to the environment.

The select_action method of the agent object is called to select an action based on the current state.
The Postprocessor object has a manipulate method that takes in the current state, action, reward, next state, and done flag, and returns modified versions of these variables.
The step_learn method is then called to update the agent's knowledge based on the observed reward and next state.

Finally, the episode_learn method is used by the agent for episodic learning (for example in REINFORCE).
```
state = env.reset()
done = False
episode_reward = 0
while done == False:
    state = Preprocessors.preprocess_state(state, agent, env)
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    episode_reward+=reward
    state, action, reward, next_state, done = Postprocessor.manipulate(state,action,reward,next_state,done)
    agent.step_learn(next_state, reward, done)
agent.episode_learn()
if prop_query == "":
    print("Reward:",episode_reward)
else:
    model_checking_result = model_checking(env,agent,Preprocessors, Postprocessor)
    print("Model Checking Result:", model_checking_result)
```

To run the safe_training component via MLflow, execute `mlflow run safe_training --env-manager=local`


#### RL Wrapper
We wrap all RL agents into a RL wrapper. This RL wrapper handles the interaction with the environment.
Via our generic interface, we can model check any kind of memoryless policy.
Our tool also supports probabilistic policies by always choosing the action with the highest probability in the probability distribution at each state and makes the policy therefore deterministic.


#### Preprocessors
A preprocessor is a tool that is used to modify the states that an RL agent receives before they are passed to the agent for learning. There are several types of preprocessors that can be used, depending on the user's preferences and goals. These include normal preprocessors, which are designed to improve the learning process; adversarial preprocessors, which are used to perform adversarial training or attacks; and defensive preprocessors, which are used to evaluate defense methods.

To use an existing preprocessor, you can use the preprocessor command-line argument `preprocessor`. For example, the following command would divide each state feature by 10 before it is observed by the RL agent: `--preprocessor="normalizer,10"`.
Note, that preprocessors get loaded automatically into the child runs. Use `--preprocessor="None"` to remove the preprocessor in the child run.
Use `--preprocessor="normalizer,12"`, to use another preprocessor in the child run.
For more information about how to use preprocessors, you can refer to the examples and to the preprocessors package.

1. If you want to create your own custom preprocessor, you can follow these steps:
2. Create a new Python script called PREPROCESSORNAME.py in the preprocessors package, and define a new class called PREPROCESSORNAME inside it.
3. Inherit the preprocessor class from the preprocessor.py script. This will give your custom preprocessor all of the necessary methods and attributes of a preprocessor.
4. Override any methods that you want to customize in your custom preprocessor.
5. Import the PREPROCESSORNAME.py script into the preprocessor builder script, which is responsible for building and configuring the preprocessor for your RL agent.
6. Add the new PREPROCESSORNAME to the build_preprocessor function, which is responsible for constructing the preprocessor object. You will need to pass any necessary arguments to the constructor of your PREPROCESSORNAME class when building the preprocessor.

It is important to make sure that your custom preprocessor is compatible with the rest of the RL agent's code, and that it performs the preprocessing tasks that you expect it to. You may need to test your custom preprocessor to ensure that it is working correctly.

It is possible to concat multiple preprocessors after each other: `--preprocessor="normalizer,10#fgsm,1"`.



### Postprocessor
Postprocessors can be used to postprocess, for example the observed state (before being passed to the replay buffer), or to render the environment.
Poisoning attacks in reinforcement learning (RL) are a type of adversarial attack that can be used to manipulate the training process of an RL agent. In a poisoning attack, the attacker injects malicious data into the training process in order to cause the RL agent to learn a suboptimal or malicious policy.
There are several ways in which poisoning attacks can be carried out in RL. One common method is to manipulate the rewards that the RL agent receives during training. For example, the attacker could artificially inflate the rewards for certain actions, causing the RL agent to prioritize those actions and learn a suboptimal policy.
This can cause the RL agent to learn a policy that is suboptimal or even harmful.

The **Postprocessor** allows the simulation of poissioning attacks during training.
It manipulates the replay buffers of the RL agents.
With the tight integration between RL and model checking, it is possible to analyze the effetivness of poissoning attacks.

To use an existing Postprocessor, you can use the Postprocessor command-line argument `postprocessor`. For example, the following command would randomly change the value if the next state is a terminal state and stores the change value into the RL agent experience: `--postprocessor="random_done"`.
Note, that postprocessors get loaded automatically into the child runs. Use `--postprocessor="None"` to remove the preprocessor in the child run.
Use `--postprocessor="OTHERPOSTPROCESSOR"`, to use another postprocessor in the child run.
For more information about how to use postprocessors, you can refer to the examples and to the postprocessors package.

1. If you want to create your own custom postprocessor, you can follow these steps:
2. Create a new Python script called POSTPROCESSORNAME.py in the postprocessors package, and define a new class called POSTPROCESSORNAME inside it.
3. Inherit the postprocessor class from the postprocessor.py script. This will give your custom postprocessor all of the necessary methods and attributes of a postprocessor.
4. Override any methods that you want to customize in your custom postprocessor.
5. Import the POSTPROCESSORNAME.py script into the postprocessor builder script, which is responsible for building and configuring the postprocessor for your RL agent.
6. Add the new POSTPROCESSORNAME to the build_postprocessor function, which is responsible for constructing the postprocessor object. You will need to pass any necessary arguments to the constructor of your POSTPROCESSORNAME class when building the postprocessor.

It is important to make sure that your custom postprocessor is compatible with the rest of the RL agent's code, and that it performs the manipulating tasks that you expect it to. You may need to test your custom postprocessor to ensure that it is working correctly.

### Model Checking
The callback function is used to incrementally build the induced DTMC, which is then passed to the model checker Storm. The callback function is called for every available action at every reachable state by the policy. It first gets the available actions at the current state. Second, it preprocesses the state and then queries the RL policy for an action. If the chosen action is not available, the callback function chooses the first action in the available action list. The callback function then checks if the chosen action was also the trigger for the current callback function and builds the induced DTMC from there if it was.
```
def callback_function(state_valuation, action_index):
    simulator.restart(state_valuation)
    available_actions = sorted(simulator.available_actions())
    current_action_name = prism_program.get_action_name(action_index)
    # conditions on the action
    state = self.__get_clean_state_dict(state_valuation.to_json(),env.storm_bridge.state_json_example)
    state = Preprocessor.preprocess_state(state, agent, env)
    selected_action = self.__get_action_for_state(env, agent, state)
    if (selected_action in available_actions) is not True:
        selected_action = available_actions[0]
    cond1 = (current_action_name == selected_action)
    return cond1

constructor = stormpy.make_sparse_model_builder(prism_program, options,stormpy.StateValuationFunctionActionMaskDouble(callback_function))
```

## Setup

Install Docker.

Install VSCode.

Add VSCode extensions: docker, Visual Studio Code Dev Containers

Open Remote Explorer, add (+), clone repository in container volume, add GITHUB-REPOSITORY URL, write coolmc volume, and coolmc target.
Afterwards, the docker container will be created (it takes time).

Verify that everything works: `python run_tests.py`

Start MLFlow server in the background: `mlflow server -h 0.0.0.0 &`

## Experiments
To run the experiments from our paper, use the bash scripts in examples (*_experiments.sh).


## Command Line Arguments
The following list contains all the major COOL-MC command line arguments. It does not contain the arguments which are related to the RL algorithms. For a detailed description, we refer to the common.rl_agents package.

| Argument             | Description                                                                                                                                                                                                                                                                                 | Options                                           | Default Value  |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|----------------|
| task                 | The type of task do you want to perform.                                                                                                                                                                                                                                                    | safe_training, rl_model_checking | safe_training  |
| project_name         | The name of your project.                                                                                                                                                                                                                                                                   |                                                   | defaultproject |
| parent_run_id        | Reference to previous experiment for retraining or verification.                                                                                                                                                                                                                            | PROJECT_IDs                                       |                |
| num_episodes         | The number of training episodes.                                                                                                                                                                                                                                                            | INTEGER NUMBER                                    | 1000           |
| eval_interval        | Interval for verification while safe_training.                                                                                                                                                                                                                                              | INTEGER NUMBER                                    | 9            |
| sliding_window_size  | Sliding window size for reward averaging over episodes.                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | 100            |
| rl_algorithm         | The name of the RL algorithm.                                                                                                                                                                                                                                                               | [SEE ommon.rl_agents.agent_builder]                               | dqn_agent      |
| prism_dir            | The directory of the PRISM files.                                                                                                                                                                                                                                                           | PATH                                              | ../prism_files |
| prism_file_path      | The name of the PRISM file.                                                                                                                                                                                                                                                                 | STR                                               |                |
| constant_definitions | Constant definitions seperated by a commata.                                                                                                                                                                                                                                                | For example: xMax=4,yMax=4,slickness=0            |                |
| prop                 | Property Query. **For safe_training:** Pmax tries to save RL policies that have higher probabilities. Pmin tries to save RL policies that have  lower probabilities. **For rl_model_checking:** In the case of induced DTMCs min/max  yield to the same property result (do not remove it). |                                                   |                |
| max_steps            | Maximal steps in the safe gym environment.                                                                                                                                                                                                                                                  |                                                   | 100            |
| disabled_features    | Disable features in the state space.                                                                                                                                                                                                                                                        | FEATURES SEPERATED BY A COMMATA                   |                |
| preprocessor     | Preprocessor configuration (each preprocessor is seperated by an  hashtag)                                                                                                                                                                            |                                                   |                |
| postprocessor    | Postprocessor configuration.                                                                                                                                                                                       |                                                   |                |
| wrong_action_penalty | If an action is not available but still chosen by the policy, return a penalty of [DEFINED HERE].                                                                                                                                                                                           |                                                   |                |
| reward_flag          | If true (1), the agent receives rewards instead of penalties.                                                                                                                                                                                                                               |                                                   | 0              |
| seed                 | Random seed for PyTorch, Numpy, Python.                                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | None (-1)      |
