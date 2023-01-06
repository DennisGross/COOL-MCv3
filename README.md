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

#### Preprocessor
The preprocessor is used to preprocess states before they get passed to the agent.
Depending on the user preferences, it may consists of normal preprocessing (for better learning), adversarial preprocessing (for adversarial trainings/attacks), and defensive preprocessing (for evaluating defense methods).
It is also used to disable certain state variables in cases where the state variables only needed to correctly model the environment in PRISM or to simulate partial observability (WITHOUT an belief function).


#### RL Wrapper
We wrap all RL agents into a RL wrapper. This RL wrapper handles the interaction with the environment.
Via our generic interface, we can model check any kind of memoryless policy.
Our tool also supports probabilistic policies by always choosing the action with the highest probability in the probability distribution at each state and makes the policy therefore deterministic.


#### Manipulator
The manipulator allows the simulation of poissioning attacks during training.
It manipulates the replay buffers of the RL agents.
With the tight integration between RL and model checking, it is possible to analyze the effetivness of poissoning attacks.

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


