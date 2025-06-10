from replay_buffer import ReplayBuffer


class HERBuffer(ReplayBuffer):
    """
    Replay which uses Hindsight Experience Replay (HER) to relable the desired goal to the achieved goal.
    """

    def __init__(self, obs_dim, act_dim, size, her_k=4):
        super().__init__(obs_dim, act_dim, size)
        self.her_k = her_k


    def store(self, obs, act, rew, next_obs, done):
        # relabel here or keep raw data and relable during sampling
        super().store(obs, act, rew, next_obs, done)


    def sample(self, batch_size):
        # TODO: Relabel transitions using the HER strategy or relabel at storing
        return super().sample(batch_size)