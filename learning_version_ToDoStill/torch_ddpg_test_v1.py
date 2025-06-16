from ddpg_torch import Agent
import gym
import numpy as np
import torch

env = gym.make("CartPole-v1")
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []
for i in range(1000):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        new_state, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, new_state, int(done))
        agent.learn()
        score += reward
        observation = new_state
    
    score_history.append(score)
    print('episode ', i, 'score %.2f' % score
          , '100 game average %.2f' % np.mean(score_history[-100:]))
    
    if i % 25 == 0:
        agent.save_models()
    
    filename = 'scores.png'
    x = [i+1 for i in range(len(score_history))]
    running_avg = np.zeros(len(score_history))