import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from agents.base import BaseAgent

class TD3Agent(BaseAgent):
    """
    RL agent implementation using TD3 (Twin Delayed DDPG).
    """

    def __init__(self, actor, critic1, critic2, actor_target, critic1_target, critic2_target, replay_buffer, config, actor_optimizer,
                 critic1_optimizer, critic2_optimizer, act_limit):
        """
        :param actor: actor network
        :param critic1: first critic network
        :param critic2: second critic network
        :param actor_target: actor target network
        :param critic1_target: first critic target network
        :param critic2_target: second critic target network
        :param replay_buffer: replay buffer
        :param config: config
        :param actor_optimizer: actor optimizer
        :param critic1_optimizer: first critic optimizer
        :param critic2_optimizer: second critic optimizer
        :param act_limit: action limit
        """
        super().__init__(actor, critic1, actor_target, critic1_target, replay_buffer, config)

        self.critic2 = critic2
        self.critic1_target = critic1_target
        self.critic2_target = critic2_target

        self.actor_optimizer = actor_optimizer
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer

        self.act_limit = torch.tensor(act_limit, dtype=torch.float32)
        self.critic_loss_fn = nn.MSELoss()

        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_delay = config.get("policy_delay", 2)
        self.config = config
        self.total_steps = 0

        # Synchronize target networks with main networks at the beginning
        # copies the weights from the actor (main network) to the actor target network. It's needed once at initialization so that both networks start with the same parameters.
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def select_action(self, obs, noise=0.0):
        """
        Select an action using the actor network with optional Gaussian noise.

        :param obs: single observation (dict with "observation" and "desired_goal" if HER; else array)
        :param noise: standard deviation of Gaussian noise
        :return: clipped action
        """
        # same as ddpg
        # If using HER, concatenate observation and desired_goal
        if self.config.get("her", True):
            obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])  # FIXED
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
        # Gym env wants numpy arrays and not tensors so convert act lim
        act_limit_np = self.act_limit.cpu().numpy() if isinstance(self.act_limit, torch.Tensor) else self.act_limit
        return np.clip(action, -act_limit_np, act_limit_np)


    def update_target(self, main_model, target_model):
        super().update_target(main_model, target_model)

    def update(self):
        """
        Perform one training step for actor and critic using a sampled batch.
        """
        # Wait until buffer has at least batch_size entries
        if self.replay_buffer.size < self.batch_size:
            return

        self.total_steps += 1

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
            # change compared to ddpg, add clipped noise to target
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            target_action = (self.actor_target(next_obs) + noise).clamp(-self.act_limit, self.act_limit)
            target_q1 = self.critic1_target(next_obs, target_action)
            target_q2 = self.critic2_target(next_obs, target_action)
            # take minimum of both q functions
            target_q = torch.min(target_q1, target_q2)
            target = rew + self.gamma * (1 - done) * target_q

        # Critic (Q-function) gradient step
        current_q1 = self.critic(obs, act)
        critic1_loss = self.critic_loss_fn(current_q1, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # second q function gradient step
        current_q2 = self.critic2(obs, act)
        critic2_loss = self.critic_loss_fn(current_q2, target)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ---- Actor update ----
        # Compute actor loss (maximize expected Q)
        # delayed actor update compared to DDPG
        if self.total_steps % self.policy_delay == 0:
            actions_pred = self.actor(obs)
            actor_loss = -self.critic(obs, actions_pred).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ---- Update target networks ----
            self.update_target(self.actor, self.actor_target)
            self.update_target(self.critic, self.critic1_target)
            self.update_target(self.critic2, self.critic2_target)

            return {"actor_loss": actor_loss.item(), "critic1_loss": critic1_loss.item(), "critic2_loss": critic2_loss.item()}

        return {"actor_loss": np.nan, "critic1_loss": critic1_loss.item(), "critic2_loss": critic2_loss.item()}

    def save(self, path):
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        torch.save(self.critic2.state_dict(), f"{path}/critic2.pth")