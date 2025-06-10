from abc import ABC, abstractmethod


class BaseBuffer(ABC):
    """
    Abstract base class for replay buffers.
    """

    @abstractmethod
    def store(self, obs, act, rew, next_obs, done):
        """
        Stores a transition in the buffer.

        :param obs: observation
        :param act: action
        :param rew: reward
        :param next_obs: next observation
        :param done: done
        """
        pass


    @abstractmethod
    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        :param batch_size: number of transitions to sample
        :return: batch of transitions
        """
        pass


    @abstractmethod
    def __len__(self):
        """
        Return the number of entries in the buffer.
        """
        pass