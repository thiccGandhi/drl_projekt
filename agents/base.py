from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    def __init__(self, actor, critic, actor_target, critic_target,
                 replay_buffer, config):
        """

        :param actor: the actor (policy) network
        :param critic: the critic (Q-function) network
        :param actor_target: target actor network
        :param critic_target: target critic network
        :param replay_buffer: replay buffer to save observations
        and sample from for training
        :param config: hyperparameters (tau, gamma, batch_size, ...)
        """
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.replay_buffer = replay_buffer

        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]


    @abstractmethod
    def select_action(self, obs, noise=0.0):
        """
        Select an action given an observation (optionally add noise).
        :param obs: current observation
        :param noise: standard deviation of exploration noise
        :return: action to take
        """
        pass


    @abstractmethod
    def update_target(self, main_model, target_model):
        """
        Update target network with Polyak Averaging towards main network:

        θ_target ← τ * θ_main + (1 - τ) * θ_target

        Tau is between 0 and 1, usually close to 0.
        :param main_model: The main network.
        :param target_model: The target network.
        """
        for target_param, main_param in zip(target_model.parameters(), main_model.parameters()):
            target_param.data.copy_(
                self.tau * main_param.data + (1.0 - self.tau) * target_param.data
            )


    @abstractmethod
    def update(self):
        """
        Perform a learning update of actor and critic networks using a batch from the replay buffer.
        """
        pass