# main_pars.py — unified, minimal, reproducible

import os, sys, json, logging, random
from copy import deepcopy

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import mujoco

# repo-local imports
sys.path.append("/home/ul/ul_student/ul_cep22/my_folders/drl_projekt")

from utils.change_parameters import (
    jprint, snapshot_geom_and_body,
    apply_override, apply_friction_override,
    contact_friction_product,
)

from networks.actor import Actor
from networks.critic import Critic
from replay_buffer.her_buffer import HERBuffer
from agents.ddpg import DDPGAgent
from agents.td3 import TD3Agent
from agents.sac import SACAgent
from drl_projekt.utils.mylogging import Logger
from trainer.trainer import Trainer
from utils.plotting import ResultsPlotter


# ---------- system / mujoco setup ----------
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
logging.getLogger("mujoco").setLevel(logging.ERROR)


# ---------- config ----------
DEFAULT_CONFIG = "/home/ul/ul_student/ul_cep22/my_folders/drl_projekt/configs/test.json"
CONFIG_PATH = os.environ.get("CFG_PATH", DEFAULT_CONFIG)
config = json.load(open(CONFIG_PATH))


# ---------- seeding ----------
SEED = int(config.get("seed", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------- env ----------
gym.register_envs(gymnasium_robotics)
render_mode = config.get("render_mode", "rgb_array")
env = gym.make(config["env_name"], render_mode=render_mode)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
obs, info = env.reset(seed=SEED)  # seed sim state


# ---------- BEFORE: compiled (XML-derived) state ----------
snap_before = snapshot_geom_and_body(env, "object0")
jprint("[BEFORE  XML/COMPILED]", snap_before)

# persist initial info (useful for W&B / logs)
config["object_gid"]            = snap_before["geom"]["gid"]
config["object_type_initial"]   = snap_before["geom"]["type"]
config["object_size_initial"]   = snap_before["geom"]["size"]
config["body_mass_initial"]     = snap_before["body"]["mass"]
config["body_inertia_initial"]  = snap_before["body"]["inertia_diag"]
config["friction_initial"]      = snap_before["geom"]["friction"]
config["condim_initial"]        = snap_before["geom"]["condim"]


# ---------- SHAPE OVERRIDE (type/size [+ optional inertials]) ----------
ovr = config.get("override_object")
if ovr:
    jprint("[REQUESTED SHAPE OVERRIDE]", {
        "name": ovr.get("name", "object0"),
        "type": ovr.get("type"),
        "size": ovr.get("size"),
        "mass": ovr.get("mass", None),
        "update_inertia": bool(ovr.get("update_inertia", False)),
    })
    res_shape = apply_override(
        env,
        name=ovr.get("name", "object0"),
        typ=ovr["type"],
        size=ovr["size"],
        mass=ovr.get("mass", None),
        update_inertia=bool(ovr.get("update_inertia", False)),
        atol=1e-9,
    )
    jprint("[AFTER  SHAPE]", {"ok": res_shape["ok"], "after": res_shape["after"]})

    snap_shape = snapshot_geom_and_body(env, ovr.get("name", "object0"))
    config["object_type_final"]      = snap_shape["geom"]["type"]
    config["object_size_final"]      = snap_shape["geom"]["size"]
    config["body_mass_final"]        = snap_shape["body"]["mass"]
    config["body_inertia_final"]     = snap_shape["body"]["inertia_diag"]
    config["object_inertia_updated"] = res_shape["inertia_updated"]


# ---------- FRICTION OVERRIDE ([slide, spin, roll] + condim) ----------
# e.g. "override_friction": {"name":"object0","friction":[0.8,0.0,0.0],"condim":1}
fovr = config.get("override_friction")
if fovr:
    jprint("[REQUESTED FRICTION OVERRIDE]", fovr)
    res_fric = apply_friction_override(
        env,
        name=fovr.get("name", "object0"),
        friction=fovr["friction"],
        condim=fovr.get("condim", None),
        atol=1e-9,
    )
    jprint("[AFTER  FRICTION]", res_fric)
    config["friction_final"] = res_fric["after"]["friction"]
    config["condim_final"]   = res_fric["after"]["condim"]

    # optional diagnostic: effective friction vs table geom (adjust name if needed)
    try:
        eff = contact_friction_product(env, fovr.get("name", "object0"), "geom_22")
        jprint("[CONTACT FRICTION object0 x table]", {"effective": eff.tolist()})
    except Exception:
        pass


# ---------- RL wiring ----------
obs_dim  = env.observation_space["observation"].shape[0]
goal_dim = env.observation_space["desired_goal"].shape[0]
act_dim  = env.action_space.shape[0]
act_lim  = np.array(env.action_space.high, dtype=np.float32)

env_params = {
    "obs_dim":   obs_dim,
    "goal_dim":  goal_dim,
    "action_dim": act_dim,
    "act_limit": act_lim,
}

her = bool(config.get("her", True))
hidden_layers = config.get("hidden_layers", [256, 256])

# stochastic_policy=True is typical for SAC; DDPG/TD3 ignore it internally
actor  = Actor(env_params, her, hidden_layers, stochastic_policy=True)
critic = Critic(env_params, her, hidden_layers)
critic2 = Critic(env_params, her, hidden_layers)  # used by TD3/SAC

actor_target  = deepcopy(actor)
critic_target = deepcopy(critic)
critic2_target = deepcopy(critic2)

actor_optimizer  = torch.optim.Adam(actor.parameters(), lr=config["lr_actor"])
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["lr_critic"])
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=config["lr_critic"])

# HER buffer (uses env's compute_reward). unwrap chain depends on Gym wrappers stack.
buffer_size = int(config.get("buffer_size", 1_000_000))
replay_buffer = HERBuffer(env.env.env.env.compute_reward, obs_dim, act_dim, goal_dim, size=buffer_size)

agent_name = config.get("agent", "sac").lower()
if agent_name == "ddpg":
    agent = DDPGAgent(actor, critic, actor_target, critic_target,
                      replay_buffer, config, actor_optimizer, critic_optimizer, act_lim)
elif agent_name == "td3":
    agent = TD3Agent(actor, critic, critic2, actor_target, critic_target, critic2_target,
                     replay_buffer, config, actor_optimizer, critic_optimizer, critic2_optimizer, act_lim)
elif agent_name == "sac":
    agent = SACAgent(actor, critic, critic2, critic_target, critic2_target,
                     replay_buffer, config, actor_optimizer, critic_optimizer, critic2_optimizer, act_lim)
else:
    raise ValueError(f"Unsupported agent type: {agent_name}")

logger  = Logger(log_dir="wandb/", config=config)
trainer = Trainer(agent, env, config, logger)

try:
    trainer.train()
    plotter = ResultsPlotter(trainer)
    plotter.plot_all(show=True)
finally:
    logger.close()
