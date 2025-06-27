import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, env_params, her, hidden_layers):
        """
        Critic network: learns the Q-function Q(s, a)
        ----------------------------------------------------
        Parameters:
        - env_params: dictionary with keys:
            - 'obs_dim'    : dimensionality of the observation vector
            - 'goal_dim'   : dimensionality of the goal vector
            - 'action_dim' : number of actions
            - 'act_limit'  : max action value (scalar or array)
        - her: bool, whether HER is used (i.e., include goal in the input)
        - hidden_layers: list of ints, defining the MLP architecture
        """
        super().__init__()

        # Convert act_limit to torch tensor ONCE, for model computations
        self.act_limit = torch.tensor(env_params['act_limit'], dtype=torch.float32)

        # Input includes state + goal (if HER) + action
        input_size = env_params['obs_dim'] + env_params['goal_dim'] + env_params['action_dim'] if her \
            else env_params['obs_dim'] + env_params['action_dim']

        # Create a list of fully connected layers based on the architecture
        layer_dims = [input_size] + hidden_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(hidden_layers))
        ])

        # Final output layer that predicts the Q-value (scalar)
        self.q_out = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x, action):
        """
        Forward pass through the critic.
        Input:
        - x: state (and goal) vector
        - action: raw action vector from the actor
        Output:
        - q-value: scalar estimate of Q(s, a)
        """
        # Normalize the action (to match the scale the critic expects)
        # Then concatenate it with the input state
        x = torch.cat([x, action / self.act_limit], dim=1)

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Output Q-value
        return self.q_out(x)
    

