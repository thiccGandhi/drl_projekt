import numpy as np


class ReplayBuffer:
    """
    Normal Replay Buffer to sample from for training.
    """

    def __init__(self, obs_dim, act_dim, size):
        """
        :param obs_dim: Observation dimension
        :param act_dim: Action dimension
        :param size: Buffer size
        """
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs_next_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size = size
        self.ptr = 0  # pointer where to store (self.ptr + 1) % self.max_size (overwrite beginning if full)
        self.curr_size = 0

    def store(self, obs, act, rew, next_obs, done):
        """
        Stores a transition in the buffer.

        :param obs: Observation
        :param act: Action
        :param rew: Reward
        :param next_obs: Next Observation
        :param done: Done
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs_next_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.

        :param batch_size: Batch size
        """
        idxs = np.random.randint(0, self.curr_size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.obs_next_buf[idxs],
            done=self.done_buf[idxs],
        )

    def size(self):
        """
        Returns the size of the buffer.
        """
        return self.curr_size
