import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from agents.base import BaseAgent


class DDPGAgent(BaseAgent):
    """
    RL agent implementation for the Deep Deterministic Policy Gradient (DDPG) algorithm.

    See: `DDPG Overview <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_
    """

    def __init__(self, actor, critic, actor_target, critic_target, replay_buffer, config, actor_optimizer, critic_optimizer, act_limit):
        """
        :param actor: actor network
        :param critic: critic network
        :param actor_target: actor target network
        :param critic_target: critic target network
        :param replay_buffer: replay buffer
        :param config: config
        :param actor_optimizer: actor optimizer
        :param critic_optimizer: critic optimizer
        :param act_limit: action limit
        """
        super().__init__(actor, critic, actor_target, critic_target, replay_buffer, config)

        # Synchronize target networks with main networks at the beginning
        # copies the weights from the actor (main network) to the actor target network. It's needed once at initialization so that both networks start with the same parameters.
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.act_limit = act_limit
        self.critic_loss = nn.MSELoss()
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.config = config

        # Optional: Set device and move models there
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.actor.to(self.device)
        # self.critic.to(self.device)
        # self.actor_target.to(self.device)
        # self.critic_target.to(self.device)

    def select_action(self, obs, noise=0.0):
        """
        Select an action using the actor network with optional Gaussian noise.

        :param obs: single observation (dict with "observation" and "desired_goal" if HER; else array)
        :param noise: standard deviation of Gaussian noise
        :return: clipped action
        """
        # If using HER, concatenate observation and desired_goal
        if self.config.get("her", True):
            obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])  #FIXED
        else:
            # If obs is a dict (no HER), get observation only
            obs_vec = obs["observation"] if isinstance(obs, dict) else obs

        obs_torch = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        self.actor.eval()  # Evaluation mode (no dropout, etc.)
        with torch.no_grad():
            action = self.actor(obs_torch).cpu().numpy()[0]
        self.actor.train()  # Switch back to training mode

        # Optionally add exploration noise
        if noise > 0:
            action += noise * np.random.randn(*action.shape)
        # Clip to valid action range
        return np.clip(action, -self.act_limit, self.act_limit)


    def update_target(self, main_model, target_model):
        super().update_target(main_model, target_model)

    def update(self):
        """
        Perform one training step for actor and critic using a sampled batch.
        """
        # Wait until buffer has at least batch_size entries
        if self.replay_buffer.size < self.batch_size:
            return

        # Sample batch of transitions from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # If using HER, concatenate goal to obs and next_obs
        if self.config.get("her", True):  # <--- FIXED/IMPORTANT
            obs = torch.tensor(
                np.concatenate([batch["obs1"], batch["goal"]], axis=1), dtype=torch.float32
            )
            next_obs = torch.tensor(
                np.concatenate([batch["obs2"], batch["goal"]], axis=1), dtype=torch.float32
            )
        else:
            obs = torch.tensor(batch["obs1"], dtype=torch.float32)
            next_obs = torch.tensor(batch["obs2"], dtype=torch.float32)

        act = torch.tensor(batch["action"], dtype=torch.float32)
        rew = torch.tensor(batch["reward"], dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(batch["done"], dtype=torch.float32).unsqueeze(1)

        # ---- Critic update ----
        # Compute target Q-value for next_obs and next_action
        with torch.no_grad():
            target_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, target_actions)
            # Bellman backup for Q-function
            target = rew + self.gamma * (1 - done) * target_q

        # Critic (Q-function) gradient step
        current_q = self.critic(obs, act)
        critic_loss = self.critic_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor update ----
        # Compute actor loss (maximize expected Q)
        actions_pred = self.actor(obs)
        actor_loss = -self.critic(obs, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Update target networks ----
        self.update_target(self.actor, self.actor_target)
        self.update_target(self.critic, self.critic_target)

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    
    def save(self, path):
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

