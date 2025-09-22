# main_pars.py — unified, minimal, reproducible (curved-shape safe)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")

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
    apply_overrides_from_config, apply_friction_override,
    contact_friction_product,
    # extra utils for physics guards
    get_geom_info, get_geom_friction, set_geom_friction,
)

from networks.actor import Actor
from networks.critic import Critic
from replay_buffer.her_buffer import HERBuffer
from agents.ddpg import DDPGAgent
from agents.td3 import TD3Agent
from agents.sac import SACAgent
from utils.mylogging import Logger
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

# ---------- apply geometry overrides ----------
snaps = apply_overrides_from_config(env, config, geom_name="object0",
                                    reseed_after=True, seed=SEED)
jprint("[BEFORE  XML/COMPILED]", snaps["before"])
jprint("[AFTER   OVERRIDES]",    snaps["after"])

# =========================================================
# Curved-object contact guards: ensure spin/roll active on BOTH sides
# =========================================================

def _is_curved(t: str) -> bool:
    return t in ("sphere", "ellipsoid")

def _find_geoms(env, substr):
    m = env.unwrapped.model
    out = []
    for gid in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if substr in nm:
            out.append(nm)
    return out

# 1) If object0 is curved, enforce condim=3 and nonzero spin/roll
obj = get_geom_info(env, "object0")
if _is_curved(obj["type"]):
    fr = obj["friction"].copy()
    # minima that work robustly with rolling bodies
    fr[1] = max(fr[1], 0.10)  # spin
    fr[2] = max(fr[2], 0.05)  # roll
    set_geom_friction(env, "object0", friction=fr, condim=3)

# 2) Make table/floor provide spin/roll too (condim=3)
support_names = list(set(_find_geoms(env, "table") + _find_geoms(env, "floor")))
for nm in support_names:
    try:
        set_geom_friction(env, nm, friction=[1.0, 0.10, 0.05], condim=3)
    except Exception:
        pass

# 3) If config gave an explicit friction override, sanitize for curved objects
fovr = config.get("override_friction")
if fovr and _is_curved(get_geom_info(env, "object0")["type"]):
    fr = list(map(float, fovr.get("friction", [0.8, 0.0, 0.0])))
    if fr[1] <= 0.0: fr[1] = 0.10
    if fr[2] <= 0.0: fr[2] = 0.05
    fovr["friction"] = fr
    fovr["condim"]   = max(int(fovr.get("condim", 1)), 3)
    res_fric = apply_friction_override(
        env,
        name=fovr.get("name", "object0"),
        friction=fovr["friction"],
        condim=fovr["condim"],
        atol=1e-9,
    )
    jprint("[AFTER  FRICTION (SANITIZED)]", res_fric)
    config["friction_final"] = res_fric["after"]["friction"]
    config["condim_final"]   = res_fric["after"]["condim"]
elif fovr:
    # non-curved: apply as-is (your original behavior)
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

# 4) Quick diagnostics and assert: condim_used must be 3 for curved
if _is_curved(get_geom_info(env, "object0")["type"]):
    def _contact_effective(ga, gb):
        fa = get_geom_friction(env, ga);  ca = get_geom_info(env, ga)["condim"]
        fb = get_geom_friction(env, gb);  cb = get_geom_info(env, gb)["condim"]
        return (fa * fb, min(ca, cb))
    used = []
    for nm in support_names:
        eff, used_cd = _contact_effective("object0", nm)
        jprint(f"[CONTACT object0 x {nm}]", {
            "fa": get_geom_friction(env, "object0").tolist(),
            "fb": get_geom_friction(env, nm).tolist(),
            "eff": eff.tolist(),
            "condim_used": int(used_cd),
        })
        used.append(int(used_cd))
    assert all(u == 3 for u in used), f"condim_used!=3 somewhere: {used}"

# 5) Re-seed + settle to remove micro-bounces contaminating HER
obs, info = env.reset(seed=SEED)
_zero = np.zeros(env.action_space.shape, dtype=np.float32)
for _ in range(300):
    env.step(_zero)

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
