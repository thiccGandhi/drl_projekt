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
        actor,             # Policy network π_ϕ
        critic1,           # First Q network Q_θ1
        critic2,           # Second Q network Q_θ2
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
            # Proper scientific way: register as a parameter for optimizer
            self.log_alpha = nn.Parameter(torch.tensor(np.log(self.alpha), device=self.device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.get("lr_alpha", 0.0003))

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
            # Always request both outputs for SAC
            if hasattr(self.actor, 'stochastic_policy') and self.actor.stochastic_policy:
                action, _ = self.actor(obs, deterministic=deterministic, with_logprob=True)
            else:
                action = self.actor(obs)
        return action.cpu().numpy()[0] * self.action_limit

    def update(self):
        """
        Perform one SAC update step: sample batch, update critics, actor, and entropy.
        """
        # Wait until buffer has enough samples
        if self.replay_buffer.size < self.batch_size:
            return

        # 1. BATCH SAMPLING = Monte Carlo Estimate of E
        # E: Expectation = empirical average over the batch 𝐸𝑠𝑡,𝑎𝑡,𝑟𝑡,𝑠𝑡+1∼𝐷
        # Sample batch from replay buffer D for each gradient step 
        # D ← D ∪ {(st, at, r(st, at), st+1)}
        # samples a minibatch (s_t, a_t, r_t, s_t+1, d_t)
        batch = self.replay_buffer.sample(self.batch_size)
        obs = torch.tensor(batch['obs1'], dtype=torch.float32, device=self.device)
        goal = torch.tensor(batch['goal'], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch['obs2'], dtype=torch.float32, device=self.device)
        act = torch.tensor(batch['action'], dtype=torch.float32, device=self.device)
        rew = torch.tensor(batch['reward'], dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.tensor(batch['done'], dtype=torch.float32, device=self.device).unsqueeze(1)

        # CONCATENATE obs + goal for ALL inputs to networks (actor, critic, target_critic)
        obs_and_goal      = torch.cat([obs, goal], dim=1)
        next_obs_and_goal = torch.cat([next_obs, goal], dim=1)

        # ----------- Critic Update ----------
        # Value Function Update: Critic (Q-function) Update
        # Compute target Q value : 
        # formula Qˆ(s_t, a_t) = r(s_t, a_t) + γ E_st+1∼p [V_ψ¯(s_t+1) ]
        with torch.no_grad():
            # Sample next action and log probability from current policy
            #next_action, next_log_prob = self.actor(next_obs_and_goal)
            next_action, next_log_prob = self.actor(next_obs_and_goal, with_logprob=True)
            # Compute target Q by taking min of two target critics
            target_q1 = self.target_critic1(next_obs_and_goal, next_action)
            target_q2 = self.target_critic2(next_obs_and_goal, next_action)
            # in the double Q variant, we use min_i=1,2 Q_θ(s_t+1, a_t+1) instead of the net value
            min_target_q = torch.min(target_q1, target_q2)
            # Soft Q backup: reward + discount * (target_q - alpha * entropy)
            target_q = rew + self.gamma * (1 - done) * (min_target_q - self.alpha * next_log_prob)

        # Q-function Loss : Current Q estimates
        # critic1_loss and critic2_loss are 𝐽_𝑄(𝜃) for each critic (with empirical mean via batch).
        current_q1 = self.critic1(obs_and_goal, act)
        current_q2 = self.critic2(obs_and_goal, act)
        critic1_loss = self.critic_loss(current_q1, target_q)
        critic2_loss = self.critic_loss(current_q2, target_q)

        # Q-function Gradient Step: Optimize critics GRADIENT STEP (∇J)
        # 𝜃𝑖 <- 𝜃𝑖 - 𝜆_𝑄 .∇_𝜃𝑖 . 𝐽_𝑄(𝜃𝑖)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ----------- Policy (Actor) Update -----------
        # Sample action and log probability for current state batch
        #new_action, log_prob = self.actor(obs_and_goal) # a_t ~ π_φ(·|s_t)
        new_action, log_prob = self.actor(obs_and_goal, with_logprob=True)
        # Q-value for new actions (use current critics)
        q1_new = self.critic1(obs_and_goal, new_action)
        q2_new = self.critic2(obs_and_goal, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        # Actor loss: maximize expected Q + entropy (i.e., minimize -(Q - alpha * log_pi))
        # reparameterized stochastic policy gradient
        # self.alpha * log_prob corresponds to entropy regularization (+αH) 
        # and -min_q_new is maximizing Q-value (but minimizing the negative, so a minimization loss)
        #(α.log (π_ϕ​(at​∣st​))−min_i (​Q_θi​​(st​,at​)) )
        actor_loss = (self.alpha * log_prob - min_q_new).mean() 
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------- Entropy (Alpha) Update -----------
        # Entropy (Alpha) Update (SAC v2, not in original Eq. 5–13 but in follow-up)
        # In the 2019 follow-up, Haarnoja et al. add automatic entropy tuning.
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()  # Update alpha

        # ----------- Target Networks Q Update (Polyak averaging) -----------
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


