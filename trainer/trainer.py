import copy
import numpy as np

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


    def train(self):
        """
        Train the agent in the environment.
        """

        while self.total_steps < self.config.max_steps:
            obs = self.env.reset()
            done = False
            metrics = []

            while not done:
                # Action selection
                if self.total_steps < self.config.min_steps:
                    act = self.env.action_space.sample() # random action because not enough in buffer
                else:
                    act = self.agent.select_action(obs, noise=self.config.exploration_noise) # agent policy with noise for exploration

                # Environment Step
                next_obs, rew, done, info = self.env.step(act)

                # Store in buffer
                self.replay_buffer.add(obs, act, rew, next_obs, done)

                # Update current observation and stats
                obs = next_obs
                self.total_steps += 1
                self.episode_reward += rew
                self.episode_length += 1

                # Training step
                metrics = self.agent.update()

            # End of episode
            # Evaluate agent every n episodes
            if self.total_episodes % self.config.eval_freq == 0:
                eval_reward = self.evaluate()
                self.logger.log_eval(eval_reward) # idk

            # Log stuff
            self.logger.log_episode(metrics) # idk

            self.total_episodes += 1


    def evaluate(self):
        """
        Evaluate the agent in the evaluation environment.

        :return: The summed reward of the episode.
        """
        obs = self.eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            act = self.agent.select_action(obs, noise=0.0) # no noise for evaluation
            obs, rew, done, _ = self.eval_env.step(act)
            total_reward += rew
        return total_reward
