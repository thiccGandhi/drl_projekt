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
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.act_limit = act_limit
        self.critic_loss = nn.MSELoss()


    def select_action(self, obs, noise=0.0):
        """
        Select an action using the actor network with optional Gaussian noise.

        :param obs: single observation
        :param noise: standard deviation of Gaussian noise
        :return: clipped action
        """
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        self.actor.eval() # puts actor in evaluation mode, no training behavior (dropout, batch normalization, etc.)
        with torch.no_grad(): # no gradient calculation, only inference not backpropagation
            action = self.actor(obs).cpu().numpy()[0]
        self.actor.train() # put actor back into training mode

        if noise > 0:
            action += noise * np.random.randn(*action.shape)
        return np.clip(action, -self.act_limit, self.act_limit) # if action is bigger than action space


    def update_target(self, main_model, target_model):
        super().update_target(main_model, target_model)


    def update(self):
        """
        Perfrom one training step for actor and critic using a sampled batch.
        """
        if self.replay_buffer.size() < self.batch_size:
            return # wait until buffer has at least batch_size entries

        # sample batch of transitions
        batch = self.replay_buffer.sample(self.batch_size)
        obs = torch.tensor(batch["obs"], dtype=torch.float32)
        act = torch.tensor(batch["act"], dtype=torch.float32)
        rew = torch.tensor(batch["rew"], dtype=torch.float32)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32)
        done = torch.tensor(batch["done"], dtype=torch.float32)

        # TODO: implement below stuff, guess its done?
        # Compute target Q-value
        with torch.no_grad():
            target_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, target_actions)
            target = rew + self.gamma * (1 - done) * target_q # done signal to terminate

        # Critic (Q-function) gradient step
        current_q = self.critic(obs, act)
        critic_loss = self.critic_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor (policy) gradient ascent step
        # ascent because of the Deterministic Policy Gradient (DPG) Theorem
        # we want to increase expected return, so ascent
        actions_pred = self.actor(obs)
        actor_loss = -self.critic(obs, actions_pred).mean() # -, because ascent is just negative descent (for e.g. Adam, dont know any ascent optimizers)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.update_target(self.actor, self.actor_target)
        self.update_target(self.critic, self.critic_target)

        return actor_loss, critic_loss