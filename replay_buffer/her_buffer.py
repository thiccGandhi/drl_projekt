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

    def sample_goals_her(self, buffer_temp, num_transitions):
        """
        This function applies HER by randomly choosing transitions and replacing their goal with a future achieved goal.

        Parameters:
        - buffer_temp: a dictionary containing full episode of transitions (obs, ag, g, actions, ag_next, etc.)
        - num_transitions: number of transitions to sample for HER

        Returns:
        - transitions: a dictionary with some goals replaced by achieved goals and rewards recomputed accordingly
        """
        # Length of the episode (number of transitions)
        T = buffer_temp['actions'].shape[0]

        # Number of transitions we want to generate via HER
        batch_size = num_transitions

        # Sample random time indices from the episode [0, T-1)
        t_samples = np.random.randint(T - 1, size=batch_size)

        # Create a new dictionary of sampled transitions by selecting timesteps
        transitions = {key: buffer_temp[key][t_samples].copy() for key in buffer_temp.keys()}

        # Choose which of the sampled transitions will be HER relabeled (based on probability self.future_p)
        # her_indexes contains the indices in the batch that will have their goal replaced
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # For HER transitions, choose a future timestep offset to get a future achieved goal
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)  # offset is how far in future to look
        future_offset = future_offset.astype(int)

        # Compute the actual future timesteps where the achieved goal will be taken from
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Get the achieved goals from those future steps
        future_ag = buffer_temp['ag'][future_t]

        # Replace the original goals at HER transition indices with the future achieved goals
        transitions['g'][her_indexes] = future_ag

        # Recalculate the reward after the goal change, using the reward function
        transitions['r'] = np.expand_dims(
            self.reward_fn(transitions['ag_next'], transitions['g'], None),
            axis=1
        )

        # Ensure all arrays have consistent shape: (batch_size, ...)
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }

        return transitions

    # It takes a dictionary buffer_temp, representing a full episode of transitions, and creates a new version where all goals are replaced with the last achieved goal.
    def _apply_hindsight(self, buffer_temp):
        """
        A simpler HER method: replace every goal in the episode with the final achieved goal.

        Parameters:
        - buffer_temp: dictionary containing a full episode's data (obs, ag, g, actions, ag_next)

        Returns:
        - hind_experiences: modified dictionary where every desired goal is replaced with the final achieved goal,
                            and the rewards are recomputed manually (or can be done with reward_fn)
        """
        # step 1) - pretend that the final achieved goal was what the agent actually intended.
        # Number of transitions/time steps in the episode
        num_transitions = len(buffer_temp['actions'])

        # Get the final achieved goal of the episode (what the agent actually accomplished at the end)
        new_desired_goal = buffer_temp['ag_next'][-1]

        # Make a shallow copy of the episode to modify and return
        hind_experiences = buffer_temp.copy()

        # Clear the reward list (we will manually fill it)
        hind_experiences['r'] = []

        # Loop through each timestep in the episode
        for i in range(num_transitions):
            # Replace the original goal at every step with the final achieved goal
            hind_experiences['g'][i] = new_desired_goal

            # # toDo: It doesn't call reward_fn() here, maybe i should call it here? 
            # # Recompute reward — currently hardcoded:
            # if i == num_transitions - 1:
            #     # If agent achieves the goal in the last step, reward = 0 (success)
            #     reward = 0   # Success: reward is 0 at the final step (matches Fetch's sparse reward logic)
            # else:
            #     reward = -1  # Failure: reward is -1 otherwise (treat it as an incomplete attempt)
            """            
            toDo: reviewMe, the attempt to fix it,  Inputs to reward_fn in reward = self.reward_fn(ag_next, goal, None):
            "If the agent achieved ag_next, and was supposed to reach goal, what reward would it get?" 
                hind_experiences['ag_next'][i]	What the agent actually achieved at step i + 1
                hind_experiences['g'][i]	The (possibly replaced) goal — what the agent was supposed to achieve
                None	Placeholder for info (some environments ignore it)
            """
            reward = self.reward_fn(hind_experiences['ag_next'][i], hind_experiences['g'][i], None)

            hind_experiences['r'].append(reward)  # Add computed reward to the list

        # Add the next-step goals (shifted one step forward)
        hind_experiences['g_next'] = hind_experiences['g'][1:, :]  # like shift the goal array for next steps

        # Return the updated episode with relabeled goals and updated rewards
        return hind_experiences
