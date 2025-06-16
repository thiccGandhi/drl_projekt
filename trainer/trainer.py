import copy
import torch
from tqdm import tqdm

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

        while self.total_steps < self.config["max_steps"]:
            obs = self.env.reset()
            done = False
            metrics = []

            while not done:
                # Action selection
                if self.total_steps < self.config["min_steps"]:
                    act = self.env.action_space.sample() # random action because not enough in buffer
                else:
                    act = self.agent.select_action(obs, noise=self.config["exploration_noise"]) # agent policy with noise for exploration

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
            if self.total_episodes % self.config["eval_freq"] == 0:
                eval_reward = self.evaluate()
                self.logger.log_eval(eval_reward) # idk

            # Log stuff
            if metrics is not None:
                self.logger.log_episode(metrics)

            self.total_episodes += 1

    # this ecaluates only one episode, is that correct? 
    # suggestions:
    # def evaluate(self, num_episodes=5):
    #     total_reward = 0
    #     for _ in range(num_episodes):
    #         obs = self.eval_env.reset()
    #         obs = obs["observation"] if isinstance(obs, dict) else obs
    #         done = False
    #         while not done:
    #             act = self.agent.select_action(obs, noise=0.0)
    #             obs, reward, done, _ = self.eval_env.step(act)
    #             obs = obs["observation"] if isinstance(obs, dict) else obs
    #             total_reward += reward
    #     return total_reward / num_episodes
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
                replay_buffer.store(obs, next_obs, action, reward, goal, done, replay_type)
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




