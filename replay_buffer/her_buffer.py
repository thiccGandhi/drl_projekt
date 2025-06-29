import numpy as np

"""
A class that contains logic for applying HER on experience data. It has two main methods:

    1) sample_goals_her: randomly relabels some transitions with future goals
    2)_apply_hindsight: deterministically changes all goals in an episode to the final achieved goal
     this is a more advanced flexible version of HER
"""


class HERBuffer:
    def __init__(self, reward_fn, obs_dim, act_dim, goal_dim, size, replay_k=4, strategy='future'):
        """
        Initialize the HER (Hindsight Experience Replay) object.
        ( Hindsight Experience Replay (HER) — a technique to help agents learn from failures by pretending they succeeded at a different goal)
        Replay which uses Hindsight Experience Replay (HER) to relable the desired goal to the achieved goal.
        
        Parameters:
        - reward_fn: a callable function to compute rewards, typically env.compute_reward(achieved_goal, desired_goal, info)
    
        """
        # This line is commented out, but in most HER strategies, this sets the ratio of HER replays to normal ones.
        # Example: if replay_k = 4, then 80% of samples would be HER replays.
        # self.future_p = 1 - (1. / (1 + replay_k))

        # Save the reward function to recompute rewards after changing the goal
        # reward_fn should be like env.compute_reward(ag, g, info)
        self.idx = 0
        self.size = 0
        self.max_size = size

        self.obs1_buffer = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buffer = np.zeros([size, obs_dim], dtype=np.float32)
        self.action_buffer = np.zeros([size, act_dim], dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.goal_buffer = np.zeros([size, goal_dim], dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.float32)

        self.reward_fn = reward_fn
        self.replay_k = replay_k
        self.future_p = 1 - (1. / (1 + replay_k))  # e.g. 0.8 if replay_k = 4
        self.strategy = strategy

        # Internal episode buffer
        self.episode_transitions = []

    def store(self, obs, next_obs, action, reward, done):
        """
        Stores a step and triggers HER augmentation when done=True.
        """
        # print(obs)
        goal = obs['desired_goal']
        ag = obs['achieved_goal']
        ag_next = next_obs['achieved_goal']

        # Append to internal episode buffer
        self.episode_transitions.append((obs, next_obs, action, reward, goal, done, ag, ag_next))

        if done:
            self._store_episode()
            self.episode_transitions = []  # Reset for next episode


    def _store_episode(self):
        """
        Takes the episode from the internal buffer and saves the original? and with HER relabeled transitions.
        """
        episode = self.episode_transitions
        T = len(episode)

        obs, next_obs, acts, rewards, goals, dones, ags, ag_nexts = zip(*episode)

        for t in range(T):
            # Store original transition
            self._store_single(obs[t]["observation"], next_obs[t]["observation"], acts[t], rewards[t], goals[t], dones[t])

        # HER relabeling
        for t in range(T):
            if np.random.uniform(0,1) > self.future_p:
                continue
            for _ in range(self.replay_k):
                if self.strategy == 'future':
                    if t + 1 >= T:
                        continue
                    future_t = np.random.randint(t + 1, T)
                    new_goal = ags[future_t]
                elif self.strategy == 'final':
                    new_goal = ags[-1]
                elif self.strategy == 'episode':
                    new_goal = ags[np.random.randint(0, T)]
                else:
                    raise ValueError("Invalid HER strategy")

                # Recompute reward with new goal
                new_reward = self.reward_fn(ag_nexts[t], new_goal, None)

                self._store_single(obs[t]["observation"], next_obs[t]["observation"], acts[t], new_reward, new_goal, dones[t])

    def _store_single(self, obs, next_obs, action, reward, goal, done):
        self.obs1_buffer[self.idx] = obs
        self.obs2_buffer[self.idx] = next_obs
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.goal_buffer[self.idx] = goal
        self.done_buffer[self.idx] = done

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return dict(
            obs1=self.obs1_buffer[idxs],
            obs2=self.obs2_buffer[idxs],
            action=self.action_buffer[idxs],
            reward=self.reward_buffer[idxs],
            goal=self.goal_buffer[idxs],
            done=self.done_buffer[idxs],
        )
