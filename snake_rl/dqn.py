import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim=11, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return np.stack(s), a, r, np.stack(s_next), d

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim=11, action_dim=3, lr=1e-3, gamma=0.9,
                 eps=1.0, eps_min=0.05, eps_decay=0.995, buffer_size=10000):
        self.q_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = 64
        self.update_target_steps = 500
        self.learn_step = 0

    def act(self, state, greedy=False):
        if (not greedy) and (np.random.rand() < self.eps):
            return np.random.randint(3)
        with torch.no_grad():
            q_vals = self.q_net(torch.tensor(state).float().unsqueeze(0))
            return int(torch.argmax(q_vals).item())

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s_next, d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32)
        s_next = torch.tensor(s_next, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q_vals = self.target_net(s_next).max(1)[0]
        target = r + (1 - d) * self.gamma * next_q_vals

        loss = self.loss_fn(q_vals, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
