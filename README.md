# COOL-MC


## Setup

Verify that everything works: `python run_tests.py`

Start MLFlow server in the background: `mlflow server -h 0.0.0.0 &`

## Architecture
We will first describe the main components of COOL-MC and then delve into the details.

### Training
The following code runs an agent in an environment. The env variable represents the environment, and the reset method is called to reset the environment to its initial state. The step method is used to take an action in the environment, and observe the resulting next state, reward, and whether the episode has finished (indicated by the done flag).

The Preprocessor object preprocesses the raw state. Preprocessing the raw state is often necessary in reinforcement learning because the raw state may be difficult for the agent to work with directly. Preprocessing can also be used to apply adversarial attacks and countermeasure simulations to the environment.

The select_action method of the agent object is called to select an action based on the current state.
The Manipulator object has a manipulate method that takes in the current state, action, reward, next state, and done flag, and returns modified versions of these variables.
The step_learn method is then called to update the agent's knowledge based on the observed reward and next state.

Finally, the episode_learn method is used by the agent for episodic learning (for example in REINFORCE).
```
state = env.reset()
done = False
episode_reward = 0
while done == False:
    state = Preprocessor.preprocess_state(state, agent, env)
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    episode_reward+=reward
    state, action, reward, next_state, done = Manipulator.manipulate(state,action,reward,next_state,done)
    agent.step_learn(next_state, reward, done)
agent.episode_learn()
if prop_query == "":
    print("Reward:",episode_reward)
else:
    model_checking_result = model_checking(env,agent,Preprocessor, Manipulator)
    print("Model Checking Result:", model_checking_result)
```

To run the safe_training component via MLflow, execute `mlflow run safe_training --env-manager=local`


#### RL Wrapper
We wrap all RL agents into a RL wrapper. This RL wrapper handles the interaction with the environment.
Via our generic interface, we can model check any kind of memoryless policy.
Our tool also supports probabilistic policies by always choosing the action with the highest probability in the probability distribution at each state and makes the policy therefore deterministic.


#### Preprocessor
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



### Manipulator
Poisoning attacks in reinforcement learning (RL) are a type of adversarial attack that can be used to manipulate the training process of an RL agent. In a poisoning attack, the attacker injects malicious data into the training process in order to cause the RL agent to learn a suboptimal or malicious policy.
There are several ways in which poisoning attacks can be carried out in RL. One common method is to manipulate the rewards that the RL agent receives during training. For example, the attacker could artificially inflate the rewards for certain actions, causing the RL agent to prioritize those actions and learn a suboptimal policy.
Another method is to manipulate the observations that the RL agent receives during training. For example, the attacker could alter the sensory input to the RL agent in order to mislead it about the state of the environment. This can cause the RL agent to learn a policy that is suboptimal or even harmful.

The **manipulator** allows the simulation of poissioning attacks during training.
It manipulates the replay buffers of the RL agents.
With the tight integration between RL and model checking, it is possible to analyze the effetivness of poissoning attacks.

To use an existing manipulator, you can use the manipulator command-line argument `manipulator`. For example, the following command would randomly change the value if the next state is a terminal state and stores the change value into the RL agent experience: `--manipulator="random_done"`.
Note, that manipulators get loaded automatically into the child runs. Use `--manipulator="None"` to remove the preprocessor in the child run.
Use `--manipulator="OTHERMANIPULATOR"`, to use another manipulator in the child run.
For more information about how to use manipulators, you can refer to the examples and to the manipulators package.

1. If you want to create your own custom manipulator, you can follow these steps:
2. Create a new Python script called MANIPULATORNAME.py in the manipulators package, and define a new class called MANIPULATORNAME inside it.
3. Inherit the manipulator class from the manipulator.py script. This will give your custom manipulator all of the necessary methods and attributes of a manipulator.
4. Override any methods that you want to customize in your custom manipulator.
5. Import the MANIPULATORNAME.py script into the manipulator builder script, which is responsible for building and configuring the manipulator for your RL agent.
6. Add the new MANIPULATORNAME to the build_manipulator function, which is responsible for constructing the manipulator object. You will need to pass any necessary arguments to the constructor of your MANIPULATORNAME class when building the manipulator.

It is important to make sure that your custom manipulator is compatible with the rest of the RL agent's code, and that it performs the manipulating tasks that you expect it to. You may need to test your custom manipulator to ensure that it is working correctly.

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
