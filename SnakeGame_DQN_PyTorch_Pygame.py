"""
Snake RL with DQN (Neural Network) + Pygame Visualization + Accuracy Reporting

• Training: Deep Q-Learning with small MLP (Neural Network)
• Visualization: Pygame demo to replay the learned policy
• Accuracy: % of episodes where the snake scores >= 1
• Run Tips:
    1) Train: python SnakeGame_DQN_PyTorch_Pygame.py --train --episodes 2000 --width 20 --height 20
    2) Play:  python SnakeGame_DQN_PyTorch_Pygame.py --play  --width 20 --height 20
Dependencies: numpy, pygame, torch
"""

import argparse
import random
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
    state: int
    reward: float
    done: bool
    info: dict

class SnakeEnv:
    def __init__(self, w=20, h=20, seed=None):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.n_actions = 3
        self.max_states = 2**11
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

        reward = -0.01
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
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return True
        return pos in self.snake

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
# 2) DQN AGENT
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=11, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim=11, action_dim=3, lr=1e-3, gamma=0.9, eps=1.0, eps_min=0.01, eps_decay=0.995):
        self.q_net = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(3)
        with torch.no_grad():
            q_vals = self.q_net(torch.tensor(state).float().unsqueeze(0))
            return int(torch.argmax(q_vals).item())

    def learn(self, s, a, r, s_next, done):
        s = torch.tensor(s).float().unsqueeze(0)
        s_next = torch.tensor(s_next).float().unsqueeze(0)
        r = torch.tensor(r, dtype=torch.float32)  # make scalar
        target = r + (0 if done else self.gamma * torch.max(self.q_net(s_next)))
        q_val = self.q_net(s)[0, a]
        loss = self.loss_fn(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# -----------------------------
# 3) TRAINING
# -----------------------------
def train_dqn(env, episodes=2000, max_steps=2000, save_path="dqn.pth"):
    agent = DQNAgent()
    success_count = 0

    for ep in range(episodes):
        s = env.reset()
        for t in range(max_steps):
            a = agent.act(s)
            res = env.step(a)
            agent.learn(s, a, res.reward, res.state, res.done)
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
# 4) PLAY MODE WITH PYGAME
# -----------------------------
def play_with_dqn(env, model_path="dqn.pth", fps=12, cell=30):
    import pygame
    agent = DQNAgent()
    agent.q_net.load_state_dict(torch.load(model_path))
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

        a = agent.act(s)
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
            pygame.time.delay(600)
            s = env.reset()
    pygame.quit()

# -----------------------------
# 5) MAIN
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--play", action="store_true")
    p.add_argument("--episodes", type=int, default=2000)
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
