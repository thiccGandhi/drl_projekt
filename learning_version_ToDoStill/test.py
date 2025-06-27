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
goal_dim = env.observation_space["desired_goal"].shape[0]
act_dim = env.action_space.shape[0]
act_lim = torch.as_tensor(env.action_space.high, dtype=torch.float32)

config = json.load(open("configs/test.json"))

env_params = {
    "obs_dim": obs_dim,
    "goal_dim": goal_dim,
    "action_dim": act_dim,
    "act_limit": act_lim
}
her = config.get("her", False)
hidden_layers = config.get("hidden_layers", [256, 256])

actor = Actor(env_params, her, hidden_layers)
critic = Critic(env_params, her, hidden_layers)
actor_target = deepcopy(actor)
critic_target = deepcopy(critic)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["lr_actor"])
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["lr_critic"])

replay_buffer = ReplayBuffer(obs_dim, act_dim, goal_dim, size=1_000_000)

agent = DDPGAgent(actor, critic, actor_target, critic_target, replay_buffer, config, actor_optimizer, critic_optimizer, act_lim)

logger = Logger("test")

trainer = Trainer(agent, env, config, logger)

# Run training step test
trainer.test_ddpg_training_step(agent, env, actor, critic, replay_buffer)