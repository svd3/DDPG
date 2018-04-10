import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import os
import copy
from filename_gen import next_path

from replay_memory import ReplayMemory

def variable(x, **kwargs):
    if torch.cuda.is_available():
        return Variable(x, **kwargs).cuda()
    return Variable(x, **kwargs)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    #target = copy.deepcopy(source)

MSELoss = nn.MSELoss()

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        num_outputs = action_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        mu = F.tanh(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear_action = nn.Linear(action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, actions):
        x = F.relu(self.linear1(inputs))
        a = F.relu(self.linear_action(actions))
        x = torch.cat((x, a), dim=1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        V = self.V(x)
        return V

class Agent(object):
    def __init__(self, input_dim, action_dim, hidden_dim=128, gamma=0.99, tau=0.01, memory_size=100000):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        if torch.cuda.is_available():
            self.actor = Actor(input_dim, action_dim, hidden_dim).cuda()
            self.actor_target = Actor(input_dim, action_dim, hidden_dim).cuda()

            self.critic = Critic(input_dim, action_dim, hidden_dim).cuda()
            self.critic_target = Critic(input_dim, action_dim, hidden_dim).cuda()
        else:
            self.actor = Actor(input_dim, action_dim, hidden_dim)
            self.actor_target = Actor(input_dim, action_dim, hidden_dim)

            self.critic = Critic(input_dim, action_dim, hidden_dim)
            self.critic_target = Critic(input_dim, action_dim, hidden_dim)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.memory = ReplayMemory(memory_size)

        self.memory_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.overwrite = False
        self.args = (input_dim, action_dim, hidden_dim, gamma, tau, memory_size)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, exploration=None):
        #self.actor.eval()
        mu = self.actor(variable(state))
        #self.actor.train()
        mu = mu.data
        if exploration is not None:
            if torch.cuda.is_available():
                mu += torch.Tensor(exploration.noise()).cuda()
            else:
                mu += torch.Tensor(exploration.noise())

        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        state_batch = variable(torch.cat(batch.state))
        if "0.4" in torch.__version__:
            next_state_batch = variable(torch.cat(batch.next_state))
        else:
            next_state_batch = variable(torch.cat(batch.next_state), volatile=True)
        action_batch = variable(torch.cat(batch.action))
        reward_batch = variable(torch.cat(batch.reward))
        mask_batch = variable(torch.cat(batch.mask))
        if "0.4" in torch.__version__:
            with torch.no_grad():
                next_action_batch = self.actor_target(next_state_batch)
                next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

                reward_batch = torch.unsqueeze(reward_batch, 1)
                expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)
        else:
            next_action_batch = self.actor_target(next_state_batch)
            next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

            reward_batch = torch.unsqueeze(reward_batch, 1)
            expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic(state_batch, action_batch)

        value_loss = MSELoss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic(state_batch, self.actor(state_batch))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def learn(self, epochs, batch_size=64):
        for epoch in range(epochs):
            # sample new batch here
            batch, _ = self.memory.sample(batch_size)
            self.update_parameters(batch)

    def save(self, memory=False):
        path = './pretrained'
        os.makedirs(path, exist_ok=True)
        if not self.overwrite:
            self.filename = next_path(os.path.join(path, 'model_%s.pth.tar'))
        self.overwrite = True
        state = {
            'args': self.args,
            'actor_dict': self.actor.state_dict(),
            'actor_target_dict': self.actor_target.state_dict(),
            'critic_dict': self.critic.state_dict(),
            'critic_target_dict': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict()
        }
        torch.save(state, self.filename)
        if memory:
            memory_buffer = {
                'memory': self.memory
            }
            torch.save(memory_buffer, './pretrained/memory.pkl')
        print("Saved to " + self.filename)

def load_agent(file='pretrained/model.pth.tar', hard=True, optim_reset=True, memory=False, **kwargs):
    print("Loading...")
    state = torch.load(file, map_location=lambda storage, loc: storage)
    args = state['args']
    #agent = Agent(*args)
    agent = Agent(input_dim=args[0], action_dim=args[1], hidden_dim=args[2], **kwargs)
    agent.actor.load_state_dict(state['actor_dict'])
    agent.critic.load_state_dict(state['critic_dict'])
    if not optim_reset:
        agent.actor_optim.load_state_dict(state['actor_optim'])
        agent.critic_optim.load_state_dict(state['critic_optim'])
    if hard:
        hard_update(agent.actor_target, agent.actor)
        hard_update(agent.critic_target, agent.critic)
    else:
        agent.actor_target.load_state_dict(state['actor_target_dict'])
        agent.critic_target.load_state_dict(state['critic_target_dict'])
    if memory:
        memory_buffer = torch.load('./pretrained/memory.pkl')
        agent.memory = memory_buffer['memory']
    print("Loaded.")
    return agent
