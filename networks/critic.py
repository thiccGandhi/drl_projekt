import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    The Critic network representing the Q-function Q(s,a).
    Leanrs to evaluate how good a state-action pair is.
    """
    def __init__(self, obs_dim, act_dim):
        """
        :param obs_dim: The dimension of the observation space.
        :param act_dim: The dimension of the action space.
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # single Q-value
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)  # concat along last dimension
        return self.net(x)
