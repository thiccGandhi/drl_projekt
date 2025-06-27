import copy
import torch
from tqdm import tqdm
import numpy as np
from collections import deque

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
        self.last_100_successes = deque(maxlen=100)


    def train(self):
        """
        Train the agent in the environment.
        """

        while self.total_steps < self.config["max_steps"]:
            obs, _ = self.env.reset() # for whatever reason this returns a tuple
            # print(obs)
            done = False
            episode_metrics = []
            final_info = {}

            while not done:
                # Action selection
                if self.total_steps < self.config["min_steps"]:
                    act = np.array(self.env.action_space.sample(), dtype=np.float32) # random action because not enough in buffer
                else:
                    act = np.array(self.agent.select_action(obs, noise=self.config["exploration_noise"]), dtype=np.float32) # agent policy with noise for exploration

                # Environment Step
                # result = self.env.step(self.env.action_space.sample())
                # print(f"Step output length: {len(result)}")
                next_obs, rew, terminated, truncated, info = self.env.step(act)
                done = terminated or truncated
                final_info = info

                # Store in buffer
                self.replay_buffer.store(obs, next_obs, act, rew, done)

                # Update current observation and stats
                obs = next_obs
                self.total_steps += 1
                self.episode_reward += rew
                self.episode_length += 1

                # Training step
                metrics = self.agent.update()
                if metrics:  # sometimes it might return None early in training
                    episode_metrics.append(metrics)

            # End of episode
            episode_success = final_info.get("is_success", 0.0)
            self.last_100_successes.append(episode_success)

            # Evaluate agent every n episodes
            if self.total_episodes % self.config["eval_freq"] == 0:
                success_rate = self.evaluate() # or reward
                self.logger.log_eval({"eval/success_rate": success_rate}, step=self.total_episodes) # idk
                self.eval_history.append((self.total_episodes, success_rate))

            # train loss, success rate
            if episode_metrics:
                avg_metrics = {
                    key: np.mean([m[key] for m in episode_metrics]) for key in episode_metrics[0].keys()
                }
                avg_metrics["train/success_rate_100"] = np.mean(self.last_100_successes)
                # self.logger.log_episode(avg_metrics, step=self.total_episodes)
                self.logger.log_episode({
                    **avg_metrics,
                    "train/success_rate_100": np.mean(self.last_100_successes),
                    "train/success_raw": episode_success  # (optional to debug individual episodes)
                }, step=self.total_episodes)
                self.training_history.append({"step": self.total_episodes, **avg_metrics})

            # increase episode counter
            self.total_episodes += 1

    # this ecaluates only one episode, is that correct? 
    # suggestions:
    def evaluate(self, num_episodes=5, return_success_rate=True):
        num_successes = 0
        total_reward = 0

        for _ in range(num_episodes):
            #obs = self.eval_env.reset()
            obs, _ = self.eval_env.reset()
            done = False
            info = None
            episode_reward = 0

            while not done:
                act = self.agent.select_action(obs, noise=0.0)
                #obs, rew, done, info = self.eval_env.step(act)
                obs, reward, terminated, truncated, info = self.eval_env.step(act)
                done = terminated or truncated
                episode_reward += reward

            total_reward += episode_reward

            if return_success_rate:
                if info.get("is_success", 0.0) == 1.0:
                    num_successes += 1

        if return_success_rate:
            return num_successes / num_episodes
        else:
            return total_reward / num_episodes


    #def evaluate(self):
    #    """
    #    Evaluate the agent in the evaluation environment.
    #
    #    :return: The summed reward of the episode.
    #    """
    #    obs = self.eval_env.reset()
    #    done = False
    #    total_reward = 0
    #    while not done:
    #        act = self.agent.select_action(obs, noise=0.0) # no noise for evaluation
    #        obs, rew, done, _ = self.eval_env.step(act)
    #        total_reward += rew
    #    return total_reward

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




