import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, env_params, her, hidden_layers):
        """
        Actor network: learns the policy π(s) → a (action)
        ----------------------------------------------------
        Parameters:
        - env_params: dictionary with keys:
            - 'obs_dim'    : dimensionality of the observation vector
            - 'goal_dim'   : dimensionality of the goal vector
            - 'action_dim' : number of actions (output size)
            - 'act_limit'  : scalar or array defining max action values per dimension
        - her: bool, if True → concatenate goal with observation
        - hidden_layers: list of ints, e.g. [256, 256], defining the architecture
        """
        super(Actor, self).__init__()

        # Input size depends on whether we're including the goal vector (HER)
        input_size = env_params['obs_dim'] + env_params['goal_dim'] if her else env_params['obs_dim']

        # Store the action limit (used to scale output to correct range)
        self.act_limit = env_params['act_limit']  # e.g. 1.0 or env.action_space.high

        # Create a list of fully connected layers based on the specified architecture
        layer_dims = [input_size] + hidden_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(hidden_layers))
        ])
        
        # Final output layer that maps to the action space
        self.action_out = nn.Linear(hidden_layers[-1], env_params['action_dim'])


    def forward(self, x):
        """
        Forward pass through the actor.
        Input: state vector (and goal if HER is used)
        Output: continuous action vector in [-act_limit, act_limit]
        """
        # Pass through hidden layers using ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Final layer uses tanh to restrict output to [-1, 1]
        x = torch.tanh(self.action_out(x))

        # Scale output to match the environment's action space range
        return self.act_limit * x
