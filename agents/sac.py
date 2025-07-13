import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent (PyTorch version)
    - Double Q-learning (two critics, take min)
    - Entropy regularization (max-entropy RL)
    - Polyak averaging for target networks
    """

    def __init__(
        self,
        actor,             # Policy network
        critic1,           # First Q network
        critic2,           # Second Q network
        target_critic1,    # Target Q1 network
        target_critic2,    # Target Q2 network
        replay_buffer,     # Experience replay buffer
        config,            # Config dict (gamma, tau, batch_size, alpha, etc.)
        actor_optimizer,   # Optimizer for actor
        critic1_optimizer, # Optimizer for Q1
        critic2_optimizer, # Optimizer for Q2
        alpha_optimizer=None, # Optimizer for alpha (if automatic entropy tuning)
        action_limit=1.0   # Max action magnitude
    ):
        # Networks
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2

        # Sync target networks with main networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.replay_buffer = replay_buffer
        self.config = config
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.device = config.get('device', 'cpu')
        self.alpha = config.get('alpha', 0.2)  # Entropy coefficient

        # Automatic entropy tuning
        self.automatic_entropy_tuning = config.get('automatic_entropy_tuning', False)
        if self.automatic_entropy_tuning:
            self.target_entropy = config.get('target_entropy', -actor.action_dim)
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = alpha_optimizer

        # Optimizers
        self.actor_optimizer = actor_optimizer
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer

        self.action_limit = action_limit
        self.critic_loss = nn.MSELoss()

    def select_action(self, obs, deterministic=False):
        """
        Select action according to current policy.
        If deterministic=False, sample from policy; else, take mean action.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action, _ = self.actor(obs, deterministic=True)
            else:
                action, _ = self.actor(obs)
        return action.cpu().numpy()[0] * self.action_limit

    def update(self):
        """
        Perform one SAC update step: sample batch, update critics, actor, and entropy.
        """
        # Wait until buffer has enough samples
        if self.replay_buffer.size < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        obs = torch.tensor(batch['obs1'], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch['obs2'], dtype=torch.float32, device=self.device)
        act = torch.tensor(batch['action'], dtype=torch.float32, device=self.device)
        rew = torch.tensor(batch['reward'], dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.tensor(batch['done'], dtype=torch.float32, device=self.device).unsqueeze(1)

        # ----------- Critic Update -----------
        # Compute target Q value
        with torch.no_grad():
            # Sample next action and log probability from current policy
            next_action, next_log_prob = self.actor(next_obs)
            # Compute target Q by taking min of two target critics
            target_q1 = self.target_critic1(next_obs, next_action)
            target_q2 = self.target_critic2(next_obs, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            # Soft Q backup: reward + discount * (target_q - alpha * entropy)
            target_q = rew + self.gamma * (1 - done) * (min_target_q - self.alpha * next_log_prob)

        # Current Q estimates
        current_q1 = self.critic1(obs, act)
        current_q2 = self.critic2(obs, act)
        critic1_loss = self.critic_loss(current_q1, target_q)
        critic2_loss = self.critic_loss(current_q2, target_q)

        # Optimize critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ----------- Actor Update -----------
        # Sample action and log probability for current state batch
        new_action, log_prob = self.actor(obs)
        # Q-value for new actions (use current critics)
        q1_new = self.critic1(obs, new_action)
        q2_new = self.critic2(obs, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        # Actor loss: maximize expected Q + entropy (i.e., minimize -(Q - alpha * log_pi))
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------- Entropy (Alpha) Update -----------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()  # Update alpha

        # ----------- Target Networks Update (Polyak averaging) -----------
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "alpha": self.alpha,
        }

    def _soft_update(self, source, target):
        """Polyak averaging: target <- tau * source + (1-tau) * target"""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data)

    def save(self, path):
        """Save model parameters"""
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic1.state_dict(), f"{path}/critic1.pth")
        torch.save(self.critic2.state_dict(), f"{path}/critic2.pth")

    def load(self, path):
        """Load model parameters"""
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic1.load_state_dict(torch.load(f"{path}/critic1.pth"))
        self.critic2.load_state_dict(torch.load(f"{path}/critic2.pth"))


