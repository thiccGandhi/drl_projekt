# toDo: delete these notes later:
# names changed: ptr->idx, curr_size -> size,  next_obs -> obs2

import numpy as np

class ReplayBuffer:
    """
    Normal Replay Buffer to sample from for training.
    """

    def __init__(self, obs_dim, act_dim, goal_dim, size):
        """
        Initializes the Replay Buffer.

        Parameters:
        - obs_dim:    Dimension of the observation vector.
        - act_dim:    Dimension of the action vector.
        - goal_dim:   Dimension of the goal vector.
        - size:       Maximum number of experiences to store (buffer size).
        """
        # idx = pointer where to store (self.ptr + 1) % self.max_size (overwrite beginning if full)
        self.idx = 0           # Points to the next index to overwrite in the buffer.
        self.size = 0          # Current number of stored experiences (size of the buffer).
        self.max_size = size   # Maximum number of experiences the buffer can hold.

        # Preallocate memory for each component of the transition (experience).
        # These are numpy arrays, not tensors yet, to save memory and CPU-GPU overhead.
    
        # Stores the observation before taking an action.
        self.obs1_buffer   = np.zeros([size, obs_dim], dtype=np.float32)

        # Stores the observation after taking the action.
        self.obs2_buffer   = np.zeros([size, obs_dim], dtype=np.float32)

        # Stores the action taken by the agent.
        self.action_buffer = np.zeros([size, act_dim], dtype=np.float32)

        # Stores the reward received after the action.
        self.reward_buffer = np.zeros(size, dtype=np.float32)

        # Stores the goal (used in goal-conditioned RL).
        self.goal_buffer   = np.zeros([size, goal_dim], dtype=np.float32)

        # Stores whether the episode ended at this step (1.0 for done, 0.0 otherwise).
        self.done_buffer   = np.zeros(size, dtype=np.float32)

        # Stores the type of replay: 0 = normal, 1 = HER (Hindsight Experience Replay).
        self.type_buffer   = np.zeros(size, dtype=np.float32)
     
        self.curr_size = 0

    def store(self, obs, next_obs, action, reward, goal, done, replay_type):
        """
        Stores one transition (experience) into the buffer.

        Parameters:
        - obs:         The observation before taking the action.
        - next_obs:    The observation after taking the action. (Next Observation)
        - action:      The action taken by the agent.
        - reward:      The reward received after taking the action.
        - goal:        The goal the agent was trying to achieve.
        - done:        Flag (0 or 1) indicating if the episode ended.
        - replay_type: Type of transition: 0 = standard, 1 = HER.
        """
        # Write the data into the appropriate buffer slots at current index
        self.obs1_buffer[self.idx]   = obs
        self.obs2_buffer[self.idx]   = next_obs
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.goal_buffer[self.idx]   = goal
        self.done_buffer[self.idx]   = done
        self.type_buffer[self.idx]   = replay_type

        # Move to the next index, wrap around if at the end (circular buffer)
        self.idx  = (self.idx + 1) % self.max_size

        # Update the current size (stops growing when full)
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the buffer for training.

        Parameters:
        - batch_size: Number of transitions to sample.
    
        Returns:
        A dictionary with sampled arrays for each component.
        """
        # Sample random indices from the filled portion of the buffer.
        random_idxs = np.random.randint(0, self.size, batch_size)

        # Return a batch dictionary with matching keys
        return dict(
            obs1        = self.obs1_buffer[random_idxs],     # batch of obs before action
            obs2        = self.obs2_buffer[random_idxs],     # batch of obs after action
            action      = self.action_buffer[random_idxs],   # batch of actions
            reward      = self.reward_buffer[random_idxs],   # batch of rewards
            goal        = self.goal_buffer[random_idxs],     # batch of goals
            done        = self.done_buffer[random_idxs],     # batch of done flags
            replay_type = self.type_buffer[random_idxs],     # batch of types (HER or not)
        )
    
    
    def clear(self):
        """
        Clears the replay buffer by resetting all arrays and indices.
        """
        self.obs1_buffer.fill(0)
        self.obs2_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.goal_buffer.fill(0)
        self.done_buffer.fill(0)
        self.type_buffer.fill(0)
        self.idx = 0
        self.size = 0
        
        
        