import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# this class is to encourage exploration
# Ornstein-Uhlenbeck noise for exploration
# a noise from physics that models the motion of of brownian particles, which is a random motion of particles
# a particle sucject to a random walk based on interactions with other nearby partcles
# it gives you a temorally correlated noise in time that centers around a mean of 0

# this will be used in our actor class, to add in some explorations noiseto the action selection 
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset() # reset the tempral correclation
    
    def __call__(self):
        # overwrite the call function
        # allows u to noise = OUActionNoise() and noise() to return	
        # just a type of random normal noise that is correlated in time
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape) 
        self.x_prev = x
        return x
    
    def reset(self):
        # check to make sure x0 exists, if it does not then set it to zero value
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# numy arrays of the shape of the action space, the obsevation space and the rewards so that
# we can have a memeory of the events that have happened so we can sample them during the learning step
# matricies that keep track of the state reward action transitions
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0 # keeps track of the last memeory that we have stored
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        # when the episode is over, the agent enters ther terminal state from which it recieves no future rewards, so the val of this terminal state is zero, so the way we wanna keep track of when we transition into terminal state is by saving the done flags from the openai gem environment
    
    def store_transition(self, state, action, reward, state_, done):
        # when mem_cntr is greater than mem_size, just wraps around from 0 to mem_size
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        #--# self.terminal_memory[index] = 1 - done
        self.terminal_memory[index] = done # when we get to the update equation, the bellman equation for our learning function, we want to multiple by whether the epside is over or not
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class CriticNetwork(nn.Module):
    # beta: learning rate
    # fc1_dims : num if dims for the 1st fully connected layer
    # name: for saving the network
    # chkpt_dir: where to save the network
    # the parameters for deep NN that approximates the value function
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        # fisrt layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # var to initialize the weights and biases of the neural network f1
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        # initialize that layer's weisghts and biases with uniform distribution
        # to constrain the initial weights of the network  to a very narrow region of parameter space
        # to help you get better convergence
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1) # parameter u wanna initialze and then lower and upper bounadries
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        # batch normal layer: batch normalization helps with convegence of the model
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # the critic gets the state and the action as input
        # but we're gonna add them at the end of the network
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = T.optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # if you have a 2nd gpu then  'cuda:1'
        self.to(self.device)
    
    # action is continuous so it's a vector
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))





class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)	

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003 # from paper
        # mu is represenation of the policy, it is a real vector of shape n_actions (actual actions not the probabilites)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


# gamma: agent's discount factor
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        
        # since it's an off-policy agent, just like deep Q network, we need to have target networks as well as the base networks
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetActor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Critic')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        #++# STORE ENV FOR GOAL REWARD RECOMPUTATION, This allows reward recomputation inside the learning loop based on achieved/desired goals
        self.env = env
        #++#
        # TimeLimit doesn’t implement compute_reward, so grab the real Fetch env here:
        self._reward_env = getattr(env, 'unwrapped', env)
        #++#
        self.update_network_parameters(tau=1)

    # you have to put the actor into evaluation mode, it doesn't perform an evaulation step
    # but just tells pytorch that you don't wanna calculate statistics for the batch norm layers
    # otherwise the agent won't learn --> only needed if u have batch normalization or dropout
    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # get the action from the actor network and send it to the device
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()

        # this is pytorch specific, otherwise it won#t give u the numpy value
        # and you can't pass an tensor into the openai gem
        #--# return mu_prime.cpu().detach().numpy()
        #++# #### This ensures the action stays within the environment's valid range (e.g., [-1, 1])
        return np.clip(mu_prime.cpu().detach().numpy(), -1.0, 1.0)


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def learn(self):
        # you dont wanna learn if u havent filled up at least batch size of your memory buffer
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        # turn all into tensors, since it all come back as numpy arrays
        # as long they're on the same device it's fine
        #--# reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        #++#  #### This computes goal-conditioned sparse reward based on actual goal distance
        goal_dim = self.env.observation_space.spaces['desired_goal'].shape[0]
        achieved_goals = new_state[:, -goal_dim:]
        desired_goals = state[:, -goal_dim:]
        # rewards = np.array([
        #     self.env.compute_reward(ag, dg, info={})
        #     for ag, dg in zip(achieved_goals, desired_goals)
        # ], dtype=np.float32)
        rewards = np.array([
            # call compute_reward on the base Fetch env, not the TimeLimit wrapper
            self._reward_env.compute_reward(ag, dg, info={})
            for ag, dg in zip(achieved_goals, desired_goals)
        ], dtype=np.float32)
        reward = T.tensor(rewards, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        #++#

        # pytorch specific
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # calulate the target actions just like bellman equation for deep Q learning
        target_actions = self.target_actor.forward(new_state)
        # we get the target_actiona from the target actor network and plug it in the state value function for the target critic network
        critic_value_ = self.target_critic.forward(new_state, target_actions).detach()
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            # ### Your current Bellman update uses `* done` — it should be `* (1 - done)`
            #--# target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
            #++#
            target.append(reward[j] + self.gamma * critic_value_[j] * (1 - done[j]))

        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        # calculations of the loss functions
        self.critic.train()
        # In pytorch, whenever you calculate the loss function, you have to zero out the gradients so that the gradients from previous dont accumulate and interfere with the current calculations
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    # tau is a parameter that allows the update of the target network to gradually approach the evaluation networks, important for slow convergence, not too large steps in between updates
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # get the names and turn them into dicts
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params) 
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


