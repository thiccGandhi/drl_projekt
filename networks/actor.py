import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, env_params, her, hidden_layers, seed = None, stochastic_policy=True):
        """
        Actor network: learns the policy π(s) → a (action)
        ----------------------------------------------------
        Parameters:
        - env_params: dictionary with keys:
            - 'obs_dim'    : dimensionality of the observation vector
            - 'goal_dim'   : dimensionality of the goal vector
            - 'action_dim' : number of actions (output size)
            - 'act_limit'  : scalar or array defining max action values per dimension
            -  stochastic_policy : if stochastic_policy=True: outputs Gaussian (mean, std) for SAC; else, deterministic for DDPG.
        Always supports HER if her=True
        - her: bool, if True → concatenate goal with observation
        - hidden_layers: list of ints, e.g. [256, 256], defining the architecture
        """
        super(Actor, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        
        self.stochastic_policy = stochastic_policy
        
        # Input size depends on whether we're including the goal vector (HER)
        input_size = env_params['obs_dim'] + env_params['goal_dim'] if her else env_params['obs_dim']

        # Convert act_limit to torch tensor ONCE, for model computations
        self.act_limit = torch.tensor(env_params['act_limit'], dtype=torch.float32)

        # Create a list of fully connected layers based on the specified architecture
        layer_dims = [input_size] + hidden_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(hidden_layers))
        ])
        
        # Final output layer that maps to the action space
        self.action_out = nn.Linear(hidden_layers[-1], env_params['action_dim'])
        self.action_dim = env_params['action_dim']

        if stochastic_policy:
            self.log_std_out = nn.Linear(hidden_layers[-1], env_params['action_dim'])


    def forward(self, x, deterministic=False, with_logprob=False):
        """
        Forward pass through the actor.
        - If with_logprob: (action, log_prob) tuple [for SAC]
        - Else: action only [for DDPG/TD3]
        
        - Input x: input tensor (obs or if HER is used [obs, goal])
        - deterministic: use mean (for evaluation)
        - with_logprob: return (action, log_prob)
        - Output x: continuous action vector in [-act_limit, act_limit]
        
        """
        # Pass through hidden layers using ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))


        if self.stochastic_policy:
            mu = self.action_out(x)
            log_std = self.log_std_out(x)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)
            pi_distribution = Normal(mu, std)

            if deterministic:
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()
            # # Squash to action bounds
            action = torch.tanh(pi_action) * self.act_limit

            if with_logprob:
                # Tanh correction for log-prob (see SAC paper appendix)
                logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
                # Correction: numerically stable, match dims!
                logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1, keepdim=True)
                return action, logp_pi
            else:
                return action
        else:
            # Deterministic policy (DDPG-like)
            # Final layer uses tanh to restrict output to [-1, 1]
            x = torch.tanh(self.action_out(x))

            # Scale output to match the environment's action space range
            return self.act_limit * x


