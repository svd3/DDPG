import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.input_dim = 2
        self.action_dim = 1
        self.hidden_dim = 64
        self.l1 = nn.Linear(1, 16)
        self.l1_2 = nn.Linear(2,16)
        self.l2 = nn.Linear(16, 64)
        self.l3 = nn.Linear(64,64)
        self.out = nn.Linear(64,1)
    def forward(self, inputs):
        x1 = inputs[:,0].view(-1,1)
        x2 = inputs[:,1].view(-1,1)
        x = F.relu(self.l1(x1))
        y = F.relu(self.l1(x2))
        z = x+y
        z = F.relu(self.l1_2(inputs))
        z = F.relu(self.l2(z))
        z = F.relu(self.l3(z))
        r = F.relu(self.out(z))
        return r

radius2 = Net()
#X = torch.Tensor(np.random.uniform(-5,5, (1000,2)))
#r_true = torch.sqrt(X[:,0]**2 + X[:,1]**2).view(-1,1)

optimizer = torch.optim.Adam(radius2.parameters(), lr=1e-6)
for i in range(1000):
    #indx = np.random.randint(1000, size=64)
    X = torch.Tensor(np.random.uniform(-1,1, (1000,2)))
    r_true = torch.sqrt(X[:,0]**2 + X[:,1]**2).view(-1,1)
    r = radius2(X)
    loss = F.mse_loss(r, r_true)
    print(loss.data.numpy())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
