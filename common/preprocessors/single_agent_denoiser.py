from common.preprocessors.preprocessor import Preprocessor
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import walk
import os
import numpy as np
import mlflow
import shutil
from common.utilities.helper import *
from common.preprocessors.single_agent_fgsm import FGSM
import random
import getpass


NORM_SCALE = 40
HIDDEN_LAYER_SIZE = 1048


class AEDataset(Dataset):

    def __init__(self):
        self.raw_data = {}

    def artificial_data_generation(self, agent, n, epsilon, state_mapper):
        """
        Sample from self.file_path and add a random noise to it
        """
        for i in range(n):
            random_key = random.sample(self.raw_data.keys(), 1)[0]
            x = self.raw_data[random_key]
            # Random numpy noise
            rnd_epsilon = epsilon
            m_fgsm = FGSM(state_mapper, "fgsm,"+str(rnd_epsilon))
            adv_data = m_fgsm.attack(agent, x)
            adv_data = torch.from_numpy(adv_data)
            self.X.append(adv_data/NORM_SCALE)
            self.y.append(x/NORM_SCALE)
            self.X.append(x/NORM_SCALE)
            self.y.append(x/NORM_SCALE)

    def collect_data(self, state):
        # state str to sha256
        state_key = state.encode('utf-8')
        state_key = hashlib.sha256(state_key).hexdigest()
        if state_key not in self.raw_data.keys():
            # Add to raw_data
            self.raw_data[state_key] = state

    def create_adv_data(self):
        self.X = []
        self.y = []
        for idx, key in enumerate(self.raw_data.keys()):
            # Read npy
            x = self.raw_data[key]
            x = self.coop_agent.po_manager.get_observation(x, self.agent_idx)
            # numpy to pytorch tensor
            #x = torch.from_numpy(x)
            # Create adversarial data
            rnd_epsilon = random.random()
            rnd_epsilon = 0.1
            m_fgsm = FGSM(self.coop_agent.po_manager.state_mapper, "fgsm,"+str(rnd_epsilon))
            adv_data = m_fgsm.attack(self.coop_agent.agents[self.agent_idx], x)
            adv_data = torch.from_numpy(adv_data)
            #print(adv_data)
            self.X.append(adv_data/NORM_SCALE)
            self.y.append(x/NORM_SCALE)
            self.X.append(x/NORM_SCALE)
            self.y.append(x/NORM_SCALE)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.tensor(self.X[index]), torch.tensor(self.y[index]))


class AE(torch.nn.Module):

    def __init__(self, input_output_size):
        super().__init__()
        self.input_output_size = input_output_size


        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_output_size, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, input_output_size),
            torch.nn.ReLU()
        )
        self.epsilon = None
        self.to(DEVICE)

    def forward(self, x):
        x = self.nn(x)
        return x

    def attack_and_clean(self, x, rl_agent, agent_idx):
        # Get first observation of agent
        obs = np.array(rl_agent.po_manager.get_observation(x, agent_idx), copy=True)
        if self.epsilon == None:
            obs = torch.from_numpy(obs/NORM_SCALE).to(DEVICE)
            clean_obs = torch.round(self.forward(obs.float())*NORM_SCALE)
            #x = rl_agent.po_manager.inject_observation_into_state(x, clean_obs, agent_idx)
            return clean_obs
        else:
            m_fgsm = FGSM(rl_agent.po_manager.state_mapper, "fgsm,"+str(self.epsilon))
            #print(obs)
            adv_data = m_fgsm.attack(rl_agent.agents[agent_idx], obs)
            #print("adv_data", adv_data)
            adv_data = torch.from_numpy(adv_data/NORM_SCALE).to(DEVICE)
            clean_obs = torch.round(self.forward(adv_data)*NORM_SCALE)
            #print("Cleaned", clean_obs)
            #x = rl_agent.po_manager.inject_observation_into_state(x, clean_obs, agent_idx)
            #print("Injected", x)
            return clean_obs

    def attack(self, x, rl_agent, agent_idx):
        obs = np.array(rl_agent.po_manager.get_observation(x, agent_idx), copy=True)
        if self.epsilon != None:
            m_fgsm = FGSM(rl_agent.po_manager.state_mapper, "fgsm,"+str(self.epsilon))
            adv_data = m_fgsm.attack(rl_agent.agents[agent_idx], obs)
            return adv_data
        else:
            return obs

    def clean(self, x, rl_agent):
        obs = torch.from_numpy(x/NORM_SCALE).to(DEVICE)
        clean_obs = torch.round(self.forward(obs.float())*NORM_SCALE)
        return clean_obs




    def set_attack(self, attack:str):
        self.attack_name, epsilon = attack.split(",")
        self.epsilon = float(epsilon)


    def save(self, idx):
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            username = getpass.getuser()
        torch.save(self.nn.state_dict(), 'tmp_model/nn_'+str(self.input_output_size)+'_.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path="autoencoder" + str(idx))
        shutil.rmtree('tmp_model')


    def load(self, model_path, idx):
        # replace last foldername with autoencoder of the model_path
        model_path = model_path[0].replace("model", "autoencoder" + str(idx))
        self.nn.load_state_dict(torch.load(os.path.join(model_path, 'nn_'+str(self.input_output_size)+'_.chkpt')))

class Denoiser(Preprocessor):

    def __init__(self, state_mapper, config_str):
        super().__init__(state_mapper, config_str)
        self.m_dataset = AEDataset()
        self.m_ae = AE(self.state_mapper.get_state_size())
        self.parse_config(config_str)
        self.counter = 0
        # Build model


    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        config_str = "denoiser;1000;10;0.1;100;0"
        """
        self.name, self.training_interval, self.training_epochs, self.epsilon, self.argumention, self.active = config_str.split(';')
        self.active = (self.active == "1")


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
            self.m_dataset.collect_data(state)
            # Train denoiser
            if self.counter % self.training_interval == 0 and self.counter > 0:
                self.m_dataset.create_adv_data(self.argumention)
                self.m_dataset.artificial_data_generation(rl_agent, self.argumention, self.epsilon, self.state_mapper)
                loss_function = torch.nn.MSELoss()

                # Using an Adam Optimizer with lr = 0.1
                optimizer = torch.optim.Adam(autoencoders[i].parameters(),
                                            lr = 0.0001)
                # Plot
                print("Training Autoencoder for agent")
                losses = []
                m_data_loader = DataLoader(dataset=self.m_dataset, batch_size=32, shuffle=True)
                for epoch in range(self.training_epochs):
                    epoch_loss = []
                    for (images, original_images) in m_data_loader:
                        images = images.to(DEVICE).float()

                        # Output of Autoencoder
                        reconstructs = autoencoders[i](images)
                        # numpy array to pytorch tensor
                        original_images = original_images.float()


                        # Original image to pytorch tensor on gpu
                        original_images = original_images.to(DEVICE)

                        # Calculating the loss function
                        loss = loss_function(reconstructs, original_images)

                        # The gradients are set to zero,
                        # the gradient is computed and stored.
                        # .step() performs parameter update
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Storing the losses in a list for plotting
                        losses.append(loss.cpu().detach().numpy())
                        epoch_loss.append(losses[-1])
                    print(epoch, "Average Epoch Loss",sum(epoch_loss)/len(epoch_loss))
                    m_project.mlflow_bridge.log_denoiser_loss(i, sum(epoch_loss)/len(epoch_loss),epoch)
                    torch.cuda.empty_cache()
                    gc.collect()
        self.counter+=1
        if self.active:
            return self.m_ae.clean(state, rl_agent)
        return state

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

