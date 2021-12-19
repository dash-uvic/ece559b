"""
Code based on the PyTorch DQN tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import sys,os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from typing import List, Set, Dict, Tuple, Optional, Union
import numpy as np

from replay import Transition
from model import QNetworkWithExtractor, QNetwork, QNetworkV2

class Agent(nn.Module):
    def __init__(self, num_actions:int,
                       input_shape:Union[List,Tuple] = (4,112,112), 
                       arch:str="resnet18",
                       batch_size:int = 128, 
                       gamma:float     = 0.999,
                       eps_start:float = 0.9, 
                       eps_end:float = 0.05, 
                       eps_decay:int = 200
                       ):
        self.device = "cpu"
        super(Agent, self).__init__()
        
        self.num_actions = num_actions
        self.batch_size  = batch_size
        self.gamma       = gamma
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.steps_done  = 0

        if "+" in arch:
            base,extractor = arch.split("+") 
            if base == "qnet": 
                self.policy_net = QNetworkWithExtractor(extractor, num_actions, input_shape[0])
                self.target_net = QNetworkWithExtractor(extractor, num_actions, input_shape[0])
            else:
                assert input_shape[0] == 3, f"{arch} can only be used with input channels = 3"
                self.policy_net = QNetworkV2(extractor, num_actions, input_shape[0])
                self.target_net = QNetworkV2(extractor, num_actions, input_shape[0])
        else:
            self.policy_net = QNetwork(input_shape, num_actions)
            self.target_net = QNetwork(input_shape, num_actions)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
        self.target_net.eval()
        self.policy_net.train()


    def to(self, device: Union[str,int]):
        self.device = device
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        return super(Agent, self).to(device)

    def encode(self, x:torch.Tensor, y:Optional[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net.encode(x.to(self.device), y)

    @torch.no_grad()
    def get_sample_error(self, state, action, next_state, reward, done):
        target = self.policy_net(state).data
        old_val = target[0][action]
        target_val = self.target_net(next_state).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * torch.max(target_val)

        error = abs(old_val - target[0][action])
        return error.detach().cpu()

    def select_action(self, state:torch.Tensor,
                            evaluate:bool=False) -> torch.Tensor:
       
        if evaluate:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)

    def optimize(self, memory):
        #Not enough samples in memory yet
        if len(memory) < self.batch_size:
            return
     
        self.train()

        #sample episodes from the replay memory
        transitions = memory.sample(self.batch_size)
        if len(transitions) == 3:
            transitions, idxs, is_weights = transitions
            is_weights = torch.from_numpy(is_weights).to(self.device)
        else:
            is_weights = torch.ones((self.batch_size,), device=self.device)
            idxs = [0]*self.batch_size #dummy variable
            
        #np.save("transitions.npy", transitions)
        try:
            batch = Transition(*zip(*transitions))
        except Exception as e:
            #Occassionally, prioritized memory relay sticks in a bad entry at the end and not sure why
            transitions = transitions[:-1]
            is_weights = is_weights[:-1]
            idxs = idxs[:-1]
            batch = Transition(*zip(*transitions))
            #torch.save(transitions, "transitions.pt")
            #raise e

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                  batch.next_state)), device=self.device, dtype=torch.bool) #torch.uint8)
        try:
            #torch.Size([X, 2052])
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(self.device)
        except:
            #in case there are no non-final states
            print(f"!!!! No non-final states, skipping this update.")
            return 
            

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(state_action_values.size(0), device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch


        # Compute (weighted) loss
        expected_state_action_values = expected_state_action_values.unsqueeze(1)
        loss  = is_weights.unsqueeze(1) * F.mse_loss(state_action_values, expected_state_action_values, reduction='none')
        loss  = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None: 
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        #update priority if using priority memory replay
        errors = abs(state_action_values - expected_state_action_values).detach().cpu()
        try:
            for i in range(self.batch_size):
                memory.update(idxs[i], errors[i])
        except Exception as e:
            print("!!! Error while updating the priority memory replay")
            print("idsx:", idxs)
            print("errors:", errors)

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()

    def load_checkpoint(self, filename:str, evaluate:bool=False) -> Union[Dict, List]:
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['dqn'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['step']
        self.update_target()

        self.target_net.eval()
        if evaluate:
            self.policy_net.eval()
        else:
            self.policy_net.train()

        return checkpoint["info"]

    def save_checkpoint(self, info: Union[Dict,List], 
                              filename:str):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        checkpoint = {
            'dqn': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.steps_done,
            "info" : info,
        }
        torch.save(checkpoint, filename)
