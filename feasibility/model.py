import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as M  
from typing import List, Set, Dict, Tuple, Optional, Union
import torch.nn.init as init
import numpy as np

import os
path="/home/adash/data"
if os.path.exists(path):
    os.environ['TORCH_HOME'] = os.path.join(path, 'model_zoo')
else:
    path="/home/memet/Projects/data"
    os.environ['TORCH_HOME'] = os.path.join(path, 'model_zoo')

print(f'| set TORCH_HOME: {os.path.join(path, "model_zoo")}')

# Initialize weights
def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        input_channels = input_shape[0]
        input_size = input_shape[1]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the image when squashed to a linear vector
        # We assume here that the input image is square
        conv_length = self._conv_shape(
            self._conv_shape(self._conv_shape(input_size, 4, 4), 4, 2), 3, 1
        )
        conv_shape = conv_length ** 2 * 64
        self.linear1 = nn.Linear(conv_shape, hidden_dim)
        #self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
    
    @staticmethod
    def _conv_shape(input_size, filter_size, stride, padding=0):
        return 1 + (input_size - filter_size + 2 * padding) // stride

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class QNetworkWithExtractor(nn.Module):
    def __init__(self, arch:str = "resnet50", num_actions:int = 6, num_channels:int = 3):
        super(QNetworkWithExtractor, self).__init__()
        assert hasattr(M, arch), f"{arch} ot in model_zoo"
       
        #TODO: name conventions aren't the same between models
        net = getattr(M, arch)(pretrained=True)
        num_inputs   = net.fc.in_features

        if num_channels == 3:
            for param in net.parameters():    
                param.requires_grad = False
       
        # Parameters of newly constructed modules have requires_grad=True by default
        last_layer = next(reversed(vars(net)['_modules'].items()))
        if isinstance(last_layer[1], nn.Linear):
            fc_layer = last_layer[1] 
        elif isinstance(last_layer[1], nn.Sequential):
            l = [module for module in last_layer[1].modules()]
            fc_layer = l[-1]
        
        #for param in net.parameters():
        #    param.requires_grad = False
        fc_layer.weight.requires_grad = True
        
        if isinstance(net.__dict__["_modules"][last_layer[0]], list):
            del net.__dict__["_modules"][last_layer[0]][-1]
        else:
            del net.__dict__["_modules"][last_layer[0]]
            
        if num_channels == 4:
            state = [nn.Conv2d(4,3, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1,
                                dilation=1,
                                groups=1,
                                bias=True),
                            nn.LeakyReLU(0.2, inplace=True)]
            
            modules       = [*state, *list(net.children())]
        else:
            modules       = list(net.children())

        self.encoder  = nn.Sequential(*modules)
        
        #https://github.com/TTitcombe/QNetwork/blob/master/src/models/dqn_linear.py
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, num_actions)
  
        #just for logging purposes
        self.base = arch
        self.num_channels = num_channels
        self.num_inputs = num_inputs
        self.num_actions = num_actions

    def __repr__(self):
        return f"QNetwork+Backbone: extractor: {self.base}  channels: {self.num_channels} Nd:{self.num_inputs} " \
               "Linear layers: 4 " \
               f"Actions: {self.num_actions}"

    def to(self, device:Union[str,int]):
        self.device = device
        return super(QNetworkWithExtractor, self).to(device)
    
    def encode(self, x:torch.Tensor) -> torch.Tensor:
        return self.encoder(x).view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 2:
            x = self.encoder(x).view(x.size(0), -1)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

#https://raw.githubusercontent.com/TTitcombe/QNetwork/master/src/models/dqn.py
class QNetwork(nn.Module):
    """
    A convolutional network.
    Architecture as outlined in the methods section of
    "Human-level control through deep reinforcement learning" - Mnih et. al
    There is nothing about this architecture which is specific to Deep-q-learning - in fact,
    the algorithm's performance should be fairly robust to the number and sizes of layers.
    """

    #def __init__(self, input_channels, input_size, output_size):
    def __init__(self, input_shape, num_actions):
        """
        Initialise the layers of the QNetwork
        :param input_channels: number of input channels (usually 4)
        :param input_size: width/height of the input image (we assume it's square)
        :param output_size: number out elements in the output vector
        """
        super(QNetwork, self).__init__()
        input_channels = input_shape[0]
        input_size = input_shape[1]
        output_size = num_actions

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the image when squashed to a linear vector
        # We assume here that the input image is square
        conv_length = self._conv_shape(
            self._conv_shape(self._conv_shape(input_size, 4, 4), 4, 2), 3, 1
        )
        conv_shape = conv_length ** 2 * 64
        self.linear1 = nn.Linear(conv_shape, 512)
        self.linear2 = nn.Linear(512, output_size)

        self.apply(weights_init_)
    
    @staticmethod
    def _conv_shape(input_size, filter_size, stride, padding=0):
        return 1 + (input_size - filter_size + 2 * padding) // stride
    
    def to(self, device:Union[str,int]):
        self.device = device
        return super(QNetwork, self).to(device)
    
    def encode(self, x:torch.Tensor) -> torch.Tensor:
        return x.detach()#.cpu() 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.view(x.size(0), -1)))
        return self.linear2(x)


class DDQN(QNetwork):
    """Dueling QNetwork. We inherit from QNetwork to get _conv_shape, but overwrite init and forward."""

    def __init__(self, input_channels, input_size, output_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the image when squashed to a linear vector
        # We assume here that the input image is square
        conv_length = self._conv_shape(
            self._conv_shape(self._conv_shape(input_size, 4, 4), 4, 2), 3, 1
        )
        conv_shape = conv_length ** 2 * 64

        self.linear1 = nn.Linear(conv_shape, 512)

        # The linear layers for the action stream
        self.action1 = nn.Linear(512, 256)
        self.action2 = nn.Linear(256, output_size)

        # The linear layers for the state stream
        self.state1 = nn.Linear(512, 256)
        self.state2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        # Action stream
        x_action = F.relu(self.action1(x))
        x_action = self.action2(x_action)
        x_action = x_action - torch.mean(x_action)

        # State stream
        x_state = F.relu(self.state1(x))
        x_state = self.state2(x_state)

        return x_action + x_state

class QNetworkV2(nn.Module):
    def __init__(self, arch:str = "googlenet", num_actions:int = 6, num_channels:int = 3):
        super(QNetworkV2, self).__init__()
        assert hasattr(M, arch), f"{arch} ot in model_zoo"
       
        #TODO: name conventions aren't the same between models
        net = getattr(M, arch)(pretrained=True)
        num_inputs   = net.fc.in_features + 4 #bbox
        
        self.activation = {} 
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        net.avgpool.register_forward_hook(get_activation('avgpool'))
        for param in net.parameters():    
            param.requires_grad = False
        
        #https://github.com/TTitcombe/QNetwork/blob/master/src/models/dqn_linear.py
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, num_actions)
        self.net = net

        #just for logging purposes
        self.base = arch
        self.num_inputs = num_inputs
        self.num_actions = num_actions

    def __repr__(self):
        return f"QNetworkV2: extractor: {self.base}  Nd:{self.num_inputs} " \
               "Linear layers: 4 " \
               f"Actions: {self.num_actions}"

    def to(self, device:Union[str,int]):
        self.device = device
        self.net = self.net.to(device)
        return super(QNetworkV2, self).to(device)
    
    @torch.no_grad()
    def encode(self, x:torch.Tensor) -> torch.Tensor:
        self.net(x)
        return self.activation['avgpool'].view(x.size(0), -1)

    @staticmethod
    def normalize(x, mean:float = 0.5, std:float = 0.5) -> torch.FloatTensor:
        return ((x - mean) / std).float()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        #TODO: also normalize x?
        x = self.encode(x) #N,2048
        y = self.normalize(y)#N,4
        x = torch.cat([x,y], dim=1) #N,2052
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

if __name__ == "__main__":
    x = torch.randn((1,4,112,112))
    model = QNetwork(4,112,8)
    y = model(x)
    print(y.shape)

