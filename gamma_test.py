import gym
import myenv
import argparse
import numpy as np
import _pickle as pickle

import torch
from torch.autograd import Variable
from agent import Agent, load_agent
from replay_memory import ReplayMemory, Transition
from noise import Noise, OUNoise

env_name = 'Firefly-v0' # 'Pendulum-v0'
env = gym.make(env_name)
env.goal_radius = 0.4
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
memory_size = 1000000
num_episodes = 2000
num_steps = env.episode_len
batch_size = 64
std = 0.1
#agent = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001)
noise = Noise(action_dim, mean=0., std=std)
#replay = ReplayMemory(memory_size)

gamma = Variable(torch.Tensor([0.99]), requires_grad=True)
rewards = []
times = []
agent = load_agent(file='pretrained/model_3.0.pth.tar', gamma=gamma)
for episode in range(20):
    state = torch.Tensor([env.reset()])
    episode_reward = 0.
    #std *= 0.9985
    noise.reset(0., std)
    for t in range(num_steps):
        action = agent.select_action(state, noise)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        episode_reward += reward
        #action = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])
        agent.memory.push(state, action, mask, next_state, reward)
        state = next_state
        if len(agent.memory) > batch_size * 2:
            print("True")
            agent.learn(epochs=2, batch_size=batch_size)
        if done:
            #env.goal_radius -= 2e-4
            break
    rewards.append(episode_reward)
    times.append(t+1)
    print("Episode: {}, steps: {}, noise: {:0.2f}, reward: {:0.4f}, average reward: {:0.4f}".format(episode, t+1, noise.scale, rewards[-1], np.mean(rewards[-100:])))
