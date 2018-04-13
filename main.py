import gym
import myenv
import argparse
import numpy as np
import _pickle as pickle

import torch
from agent import Agent, load_agent
from replay_memory import ReplayMemory, Transition
from exploration_noise import Noise, OUNoise

parser = argparse.ArgumentParser(description='PyTorch DDPG example')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--play', action='store_true', help='set to play mode, no learning')
parser.add_argument('--load', action='store_true', help='set to play mode, no learning')
args = parser.parse_args()
render = args.render
play = args.play
load = args.load

env_name = 'Firefly-v1' # 'Pendulum-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
memory_size = 1000000
num_episodes = 2000
num_steps = env.episode_len
batch_size = 64
std = 1.
agent = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001)
noise = Noise(action_dim, mean=0., std=std)
#replay = ReplayMemory(memory_size)
if play:
    agent = load_agent(file='pretrained/model_7.pth.tar')
if load:
    agent = load_agent(file='pretrained/model_6.0.pth.tar') #6.0

rewards =[]
for episode in range(num_episodes):
    state = torch.Tensor([env.reset()])
    episode_reward = 0.
    std *= 0.9985
    noise.reset(0., std)
    for t in range(num_steps):
        if not play:
            action = agent.select_action(state, noise)
        else:
            action = agent.select_action(state)
            #print(action)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        episode_reward += reward
        #action = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        if render:
            env.render()

        if not play:
            agent.memory.push(state, action, mask, next_state, reward)

        state = next_state

        if len(agent.memory) > batch_size * 5 and not play:
            agent.learn(epochs=2, batch_size=batch_size)

        if done:
            #env.goal_radius -= 2e-4
            if play:
                print(action)
                print("radius: {:0.2f}".format(env.goal_radius))
            break
    rewards.append(episode_reward)
    print("Episode: {}, steps: {}, noise: {:0.2f}, reward: {:0.4f}, average reward: {:0.4f}".format(episode, t+1, noise.scale, rewards[-1], np.mean(rewards[-100:])))

    if episode%10 == 0 and not play:
        agent.save()
env.close()
