import json

import gymnasium as gym
import gymnasium_robotics
import torch
from copy import deepcopy

from networks.actor import Actor
from networks.critic import Critic
from replay_buffer.replay_buffer import ReplayBuffer
from agents.ddpg import DDPGAgent
from utils.logging import Logger
from trainer.trainer import Trainer



gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPush-v4")

obs_dim = env.observation_space["observation"].shape[0]
act_dim = env.action_space.shape[0]
act_lim = torch.as_tensor(env.action_space.high, dtype=torch.float32)

config = json.load(open("configs/test.json"))

actor = Actor(obs_dim, act_dim, act_lim)
critic = Critic(obs_dim, act_dim)
actor_target = deepcopy(actor)
critic_target = deepcopy(critic)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

replay_buffer = ReplayBuffer(obs_dim, act_dim, size=1_000_000)

agent = DDPGAgent(actor, critic, actor_target, critic_target, replay_buffer, config, actor_optimizer, critic_optimizer, act_lim)

logger = Logger("test")

trainer = Trainer(agent, env, replay_buffer, config)

# Call the test function
trainer.test_ddpg_training_step(agent, env, actor, critic, replay_buffer)

