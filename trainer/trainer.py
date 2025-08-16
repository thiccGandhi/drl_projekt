import copy
import torch
from tqdm import tqdm
import numpy as np
from collections import deque

def flatten_obs(obs, her=True):
    """
    Converts a dict obs from the env to a flat vector for the agent.
    If HER is True, concatenate observation + desired_goal (the usual setup).
    """
    if isinstance(obs, dict):
        if her:
            return np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)
        else:
            return obs["observation"]
    # If not a dict, just return as array
    return np.asarray(obs)

class Trainer:
    """
    Trainer class
    """
    def __init__(self, agent, env, config, logger):
        self.agent = agent
        self.env = env
        self.eval_env = copy.deepcopy(env) # to avoid resetting the training env, if more than evaluation at end is needed?
        self.replay_buffer = agent.replay_buffer
        self.config = config
        self.logger = logger
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.success_rate = 0
        self.training_history = []   # List of dicts, one per episode
        self.eval_history = []       # List of (step, eval dict)
        self.last_100_successes = deque([0.0] * 100, maxlen=100)
        self.last_100_time_in_goal = deque([0.0] * 100, maxlen=100)
        self.current_best_success_rate = 0


    def train(self):
        """
        Train the agent in the environment.
        """

        while self.total_steps < self.config["max_steps"]:
            obs, _ = self.env.reset() # for whatever reason this returns a tuple
            #ag = obs["achieved_goal"]
            #g = obs["desired_goal"]
            #print(f"Episode {self.total_episodes} reset: AG = {ag}, G = {g}, dist = {np.linalg.norm(ag - g):.4f}")
            # print(obs)
            done = False
            episode_metrics = []
            final_info = {}
            time_in_goal = []
            current_episode_data = []

            while self.episode_length < self.config["episode_length"]:
                obs_flat = flatten_obs(obs, her=self.config.get("her", True))
                # Action selection
                if self.total_steps < self.config["min_steps"]:
                    act = np.array(self.env.action_space.sample(), dtype=np.float32) # random action because not enough in buffer
                else:
                    act = np.array(self.agent_select_action(obs_flat, explore=True), dtype=np.float32) # agent policy with noise for exploration

                # Environment Step
                # result = self.env.step(self.env.action_space.sample())
                # print(f"Step output length: {len(result)}")
                next_obs, rew, terminated, truncated, info = self.env.step(act)
                done = terminated or truncated
                final_info = info
                #if self.episode_length == 0:
                #    print(f"Step 1 reward: {rew}, is_success: {info.get('is_success')}")

                # Store in buffer
                self.replay_buffer.store(obs, next_obs, act, rew, done)
                current_episode_data.append((obs, next_obs, act, rew, done))

                # Update current observation and stats
                obs = next_obs
                self.total_steps += 1
                self.episode_reward += rew
                self.episode_length += 1
                
                timestep_success = info.get("is_success", 0.0)
                time_in_goal.append(timestep_success)

                # Training step
                if self.total_steps < self.config["update_after"]:
                    metrics = {"actor_loss": 0, "critic_loss": 0}
                else:
                    metrics = self.agent.update()
                if metrics:  # sometimes it might return None early in training
                    episode_metrics.append(metrics)

            # End of episode
            episode_end_success = final_info.get("is_success", 0.0)
            self.last_100_successes.append(episode_end_success)
            self.last_100_time_in_goal.append(np.sum(time_in_goal))


            # Evaluate agent every n episodes
            if self.total_episodes % self.config["eval_freq"] == 0:
                success_rate, eval_reward = self.evaluate() # or reward
                self.logger.log_eval({"eval/success_rate": success_rate, "eval/reward": eval_reward}, step=self.total_episodes) # idk
                self.eval_history.append((self.total_episodes, success_rate))
                
            
            save_options = []
            
            # Check for new best agent
            new_best = np.mean(self.last_100_successes)
            if self.current_best_success_rate < new_best:
                self.current_best_success_rate = new_best
                save_options.append("best")
            
            
            # Check for freq episode agent
            if self.total_episodes > 0 and self.total_episodes % self.config["save_agent_freq"] == 0:
                num_freqk = self.total_episodes // self.config["save_agent_freq"]
                save_options.append(num_freqk)
                
            
            # If there is something to save, save/animate it
            if save_options:
                print(f"Number of episodes: {self.total_episodes}, save options: {save_options}")
                current_animation = self.animate(self.agent)
                for option in save_options:
                    self.logger.save_agent(option, self.agent, current_episode_data, self.total_episodes)
                    self.logger.save_animation(option, current_animation, self.total_episodes)
            

            # train loss, success rate
            if episode_metrics:
                avg_metrics = {}
                for key in episode_metrics[0].keys():
                    values = [m[key] for m in episode_metrics if key in m and m[key] is not None]
                    if values:
                        avg_metrics[key] = np.nanmean(values)
                # avg_metrics["train/success_rate_100"] = np.mean(self.last_100_successes)
                # self.logger.log_episode(avg_metrics, step=self.total_episodes)
                self.logger.log_episode({
                    **avg_metrics,
                    "train/success_rate_100": np.mean(self.last_100_successes),
                    "train/success_raw": episode_end_success,  # (optional to debug individual episodes)
                    "train/avg_reward": self.episode_reward,
                    "train/time_in_goal_100": np.mean(self.last_100_time_in_goal)
                }, step=self.total_episodes)
                self.training_history.append({"step": self.total_episodes, **avg_metrics})

            # increase episode counter
            self.total_episodes += 1
            self.episode_reward = 0
            self.episode_length = 0
            current_episode_data = []  # Reset for next episode

    # this ecaluates only one episode, is that correct? 
    # suggestions:
    def evaluate(self, num_episodes=5):
        num_successes = 0
        total_reward = []

        for _ in range(num_episodes):
            #obs = self.eval_env.reset()
            obs, _ = self.eval_env.reset()
            obs_flat = flatten_obs(obs, her=self.config.get("her", True))
            done = False
            info = {}
            episode_reward = 0

            while not done:
                act = np.array(self.agent_select_action(obs_flat, explore=False), dtype=np.float32)
                #obs, rew, done, info = self.eval_env.step(act)
                obs, reward, terminated, truncated, info = self.eval_env.step(act)
                obs_flat = flatten_obs(obs, her=self.config.get("her", True))
                done = terminated or truncated
                episode_reward += reward

            total_reward.append(episode_reward)

            if info.get("is_success", 0.0) == 1.0:
                num_successes += 1

        return num_successes / num_episodes, np.mean(total_reward)


    def agent_select_action(self, obs, explore=True):
        """
        General action selection that handles exploration and evaluation mode
        for both SAC (deterministic arg) and DDPG/TD3 (noise arg).
        """
        # Use type or config to select logic (recommended: set 'agent' in config)
        algo = self.config.get("agent", "sac").lower()
        if algo == "sac":
            # For SAC: Exploration is handled by stochastic policy; set deterministic for evaluation
            return self.agent.select_action(obs, deterministic=not explore)
        elif algo in {"ddpg", "td3"}:
            # For DDPG/TD3: Use noise during exploration, zero during evaluation
            return self.agent.select_action(obs, noise=self.config["exploration_noise"] if explore else 0.0)
        else:
            # fallback: try both
            try:
                return self.agent.select_action(obs, deterministic=not explore)
            except TypeError:
                return self.agent.select_action(obs, noise=self.config["exploration_noise"] if explore else 0.0)


    def test_ddpg_training_step(self, agent, env, actor, critic, replay_buffer):
        # Fill buffer
        print("Filling replay buffer with 100 random transitions...")
        for _ in tqdm(range(100), "Filling replay buffer"):
            obs_dict = env.reset()[0]
            obs = obs_dict["observation"]
            done = False
            while not done:
                action = env.action_space.sample()
                next_obs_dict, reward, terminated, truncated, _ = env.step(action)
                next_obs = next_obs_dict["observation"]
                done = terminated or truncated
                #replay_buffer.store(obs, action, reward, next_obs, done)
                ###############
                goal = obs_dict["desired_goal"]
                replay_type = 0  # Standard transition
                replay_buffer.store(obs, next_obs, action, reward, goal, done)#, replay_type)
                ########################
                obs = next_obs
                if done:
                    break
        print("Replay buffer filled.")

        # Save weights
        actor_weights_before = [p.clone() for p in actor.parameters()]
        critic_weights_before = [p.clone() for p in critic.parameters()]

        out = agent.update()
        if out is None:
            print("not enough in buffer")
            return
        actor_loss, critic_loss = out

        actor_changed = any(not torch.equal(p0, p1) for p0, p1 in zip(actor_weights_before, actor.parameters()))
        critic_changed = any(not torch.equal(p0, p1) for p0, p1 in zip(critic_weights_before, critic.parameters()))

        print("Actor changed:", actor_changed)
        print("Critic changed:", critic_changed)
        print(f"Actor loss: {actor_loss.item():.4f} | Critic loss: {critic_loss.item():.4f}")

        if actor_changed and critic_changed:
            print("training does something")
        else:
            print("nothing changed")


    def animate(self, agent):
        """Animate the agent's behavior in the environment as a gif.

        :param agent: The trained agent to animate.
        """
        
        obs, _ = self.env.reset()
        obs_flat = flatten_obs(obs, her=self.config.get("her", True))
        done = False
        frames = []

        while not done:
            act = np.array(self.agent_select_action(obs_flat, explore=False), dtype=np.float32)
            #obs, rew, done, info = self.env.step(act)
            obs, reward, terminated, truncated, info = self.env.step(act)
            obs_flat = flatten_obs(obs, her=self.config.get("her", True))
            done = terminated or truncated
            frame = self.env.render()
            frames.append(frame)
        
        return frames
    
    