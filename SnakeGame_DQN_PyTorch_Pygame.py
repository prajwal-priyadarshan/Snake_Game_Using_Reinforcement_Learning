"""
Snake RL with DQN (Deep Q-Network) + Replay Buffer + Target Network + Reward Shaping + Pygame Visualization

• Training: Deep Q-Learning with a 2-layer MLP neural network (128 neurons each layer)
• Replay Buffer: Stores past experiences to break correlation and stabilize learning
• Target Network: Periodically updated network for more stable Q-value estimation
• Reward Shaping: Encourages the snake to move closer to food and penalizes moving away
• Visualization: Pygame demo to replay the learned policy with purely greedy actions (no randomness)
• Accuracy: Reports % of episodes where the snake successfully eats at least one food item

Run Tips:
    1) Train: python SnakeGame_DQN_PyTorch_Pygame.py --train --episodes 2000 --width 20 --height 20
    2) Play:  python SnakeGame_DQN_PyTorch_Pygame.py --play  --width 20 --height 20

Dependencies: numpy, pygame, torch

While testing we got a accuray of 99.25%, with training of 2000 episodes
"""

import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1) ENVIRONMENT
# -----------------------------
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
Action = int  # 0=Forward, 1=TurnLeft, 2=TurnRight

@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict

class SnakeEnv:
    def __init__(self, w=20, h=20, seed=None):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.n_actions = 3
        self.reset()

    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.direction = RIGHT
        self.snake: List[Tuple[int, int]] = [(cx, cy), (cx-1, cy), (cx-2, cy)]
        self._place_food()
        self.score = 0
        self.steps_since_food = 0
        self.hunger_limit = self.w * self.h * 2
        return self._encode_state()

    def step(self, action):
        assert action in (0, 1, 2)
        if action == 1:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4

        dx, dy = DIRS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Reward shaping
        reward = -0.01
        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        if new_dist < old_dist:
            reward += 0.1  # encourage moving closer
        else:
            reward -= 0.03  # discourage moving away

        if self._is_collision(new_head):
            return StepResult(self._encode_state(), -10.0, True, {"reason": "collision"})

        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward += 10.0
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()
            self.steps_since_food += 1

        if self.steps_since_food > self.hunger_limit:
            return StepResult(self._encode_state(), -10.0, True, {"reason": "hunger"})

        return StepResult(self._encode_state(), reward, False, {})

    def _place_food(self):
        free = {(x, y) for x in range(self.w) for y in range(self.h)} - set(self.snake)
        self.food = self.rng.choice(list(free))

    def _is_collision(self, pos):
        x, y = pos
        return x < 0 or x >= self.w or y < 0 or y >= self.h or pos in self.snake

    def _encode_state(self):
        head = self.snake[0]

        def cell_in_dir(d):
            dx, dy = DIRS[d]
            return (head[0] + dx, head[1] + dy)

        ahead = cell_in_dir(self.direction)
        right = cell_in_dir((self.direction + 1) % 4)
        left = cell_in_dir((self.direction - 1) % 4)

        danger_ahead = self._is_collision(ahead)
        danger_right = self._is_collision(right)
        danger_left = self._is_collision(left)

        moving = [0, 0, 0, 0]
        moving[self.direction] = 1

        food_up = int(self.food[1] < head[1])
        food_right = int(self.food[0] > head[0])
        food_down = int(self.food[1] > head[1])
        food_left = int(self.food[0] < head[0])

        bits = [
            int(danger_ahead), int(danger_right), int(danger_left),
            moving[UP], moving[RIGHT], moving[DOWN], moving[LEFT],
            food_up, food_right, food_down, food_left
        ]
        return np.array(bits, dtype=np.float32)

# -----------------------------
# 2) DQN + REPLAY BUFFER
# -----------------------------
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

# -----------------------------
# 3) TRAINING
# -----------------------------
def train_dqn(env, episodes=3000, max_steps=2000, save_path="dqn.pth"):
    agent = DQNAgent()
    success_count = 0

    for ep in range(episodes):
        s = env.reset()
        for t in range(max_steps):
            a = agent.act(s)
            res = env.step(a)
            agent.buffer.push(s, a, res.reward, res.state, res.done)
            agent.learn()
            s = res.state
            if res.done:
                break
        if env.score >= 1:
            success_count += 1
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1:4d} | Score={env.score:3d} | Eps={agent.eps:.3f}")

    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print(f"Accuracy: {(success_count / episodes) * 100:.2f}%")
    return agent

# -----------------------------
# 4) PLAY MODE (NO EPS)
# -----------------------------
def play_with_dqn(env, model_path="dqn.pth", fps=12, cell=30):
    import pygame
    agent = DQNAgent()
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.target_net.load_state_dict(agent.q_net.state_dict())
    agent.q_net.eval()

    pygame.init()
    W, H = env.w * cell, env.h * cell
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    def draw_rect(pos, color):
        x, y = pos
        pygame.draw.rect(screen, color, (x*cell, y*cell, cell-1, cell-1))

    s = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        a = agent.act(s, greedy=True)  # ✅ Always pick best action
        res = env.step(a)
        s = res.state

        screen.fill((30, 30, 30))
        for gx in range(env.w):
            for gy in range(env.h):
                pygame.draw.rect(screen, (45, 45, 45), (gx*cell, gy*cell, cell, cell), 1)
        draw_rect(env.food, (200, 70, 70))
        for i, seg in enumerate(env.snake):
            draw_rect(seg, (70, 200, 90) if i == 0 else (90, 230, 120))
        txt = font.render(f"Score: {env.score}", True, (240, 240, 240))
        screen.blit(txt, (8, 6))

        pygame.display.flip()
        clock.tick(fps)

        if res.done:
            pygame.time.delay(500)
            s = env.reset()
    pygame.quit()

# -----------------------------
# 5) MAIN
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--play", action="store_true")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    env = SnakeEnv(args.width, args.height)
    if args.train:
        train_dqn(env, episodes=args.episodes)
    if args.play:
        play_with_dqn(env)

if __name__ == "__main__":
    main()
