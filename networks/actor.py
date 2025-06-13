import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    The Actor network representing the policy \mu(s).
    Learns to select actions that maximize expected return as estimated by the Critic.
    """
    def __init__(self, obs_dim, act_dim, act_limit):
        """
        :param obs_dim: The dimension of the observation.
        :param act_dim: The dimension of the action.
        :param act_limit: The limit of the action space.
        """
        super(Actor, self).__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Multiplying by the limit of the action space to map the [-1,1] output of the `tanh` layer to the action space.
        """
        return self.net(x) * self.act_limit
