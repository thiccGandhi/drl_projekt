import json
import gymnasium as gym
import gymnasium_robotics
import torch
from copy import deepcopy
import numpy as np
import random 

from networks.actor import Actor
from networks.critic import Critic
from replay_buffer.replay_buffer import ReplayBuffer
from replay_buffer.her_buffer import HERBuffer
from agents.ddpg import DDPGAgent
from agents.td3 import TD3Agent
from agents.sac import SACAgent
from utils.mylogging import Logger
from trainer.trainer import Trainer
from utils.plotting import ResultsPlotter

# config = json.load(open("configs/test.json"))
#config = json.load(open("/home/ul/ul_student/ul_cep22/my_folders/drl_projekt/configs/test.json"))

# Minimal config for test (override as needed)
config = {
  "max_steps": 2_000_000,                # 2M env steps is minimal for FetchPush
  "min_steps": 50_000,                   # Allow for some min total steps
  "episode_length": 50,                  # Default for FetchPush (50 per episode)
  "update_after": 1000,                  # Start updating after 1000 steps
  "eval_freq": 10,                       # Evaluate every 10 episodes
  "exploration_noise": 0.1,              # Standard initial exploration
  "batch_size": 256,                     # Large batch for SAC; GPU preferred
  "learning_rate": 0.001,                # Not used; use lr_actor/lr_critic
  "lr_actor": 0.0003,
  "lr_critic": 0.0003,
  "discount": 0.98,                      # For robotic tasks 0.98-0.99 common
  "tau": 0.005,
  "buffer_size": 1_000_000,
  "seed": 42,
  "device": "cpu",
  "gamma": 0.98,
  "project_name": "rl_project",
  "run_name": "sac_fetchpush_full",
  "env_name": "FetchPush-v4",
  "agent": "sac",
  "alpha": 0.2,
  "automatic_entropy_tuning": True,      # Enable auto alpha tuning for SAC
  "target_entropy": -4,                  # Usually -action_dim
  "her": True,
  "hidden_layers": [256, 256]            # Deeper net for challenging envs
}

# Optionally save config to file if your script expects it:
# with open("configs/test.json", "w") as f: json.dump(config, f)

SEED = config["seed"]  # 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

gym.register_envs(gymnasium_robotics)
env = gym.make(config["env_name"])

env.action_space.seed(SEED)
env.observation_space.seed(SEED)
obs, info = env.reset(seed=SEED)  # Use seed here

obs_dim = env.observation_space["observation"].shape[0]
goal_dim = env.observation_space["desired_goal"].shape[0]
act_dim = env.action_space.shape[0]
act_lim = np.array(env.action_space.high, dtype=np.float32)


env_params = {
    "obs_dim": obs_dim,
    "goal_dim": goal_dim,
    "action_dim": act_dim,
    "act_limit": act_lim
}
her = config.get("her", True)
hidden_layers = config.get("hidden_layers", [256, 256])

actor = Actor(env_params, her, hidden_layers, stochastic_policy=True)
critic = Critic(env_params, her, hidden_layers)
critic2 = Critic(env_params, her, hidden_layers)
actor_target = deepcopy(actor)
critic_target = deepcopy(critic)
critic2_target = deepcopy(critic2)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["lr_actor"])
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["lr_critic"])
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=config["lr_critic"])


# replay_buffer = ReplayBuffer(obs_dim, act_dim, goal_dim, size=1_000_000)
buffer_sie = config.get("buffer_size", 1_000_000)
# toDo: check if replay_buffer = HERBuffer(env.unwrapped.compute_reward, ...) is better 
# according to https://github.com/openai/baselines/blob/master/baselines/her/her_sampler.py#L31
replay_buffer = HERBuffer(env.env.env.env.compute_reward, obs_dim, act_dim, goal_dim, size=buffer_sie)


if config["agent"] == "ddpg":
    agent = DDPGAgent(actor, critic, actor_target, critic_target,
                      replay_buffer, config, actor_optimizer, critic_optimizer, act_lim)
elif config["agent"] == "td3":
    agent = TD3Agent(actor, critic, critic2, actor_target, critic_target, critic2_target,
                     replay_buffer, config, actor_optimizer, critic_optimizer, critic2_optimizer, act_lim)
elif config["agent"] == "sac":
    agent = SACAgent(actor, critic, critic2, critic_target, critic2_target,
                     replay_buffer, config, actor_optimizer, critic_optimizer, 
                     critic2_optimizer, act_lim)
else:
    raise ValueError(f"Unsupported agent type: {config['agent']}")

logger = Logger(log_dir="wandb/", config=config)

trainer = Trainer(agent, env, config, logger)

# Run training step test
#trainer.test_ddpg_training_step(agent, env, actor, critic, replay_buffer)

trainer.train()
plotter = ResultsPlotter(trainer)
plotter.plot_all(show=True)
