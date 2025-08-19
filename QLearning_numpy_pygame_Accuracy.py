# Can define no of rows and columns + Accuracy Calculation
"""
Snake RL with customizable grid size and accuracy reporting:

• Training: Pure NumPy grid environment + Q-learning (Q-table)
• Visualization: Pygame demo to replay the learned policy
• Accuracy: % of episodes where the snake scores >= 1
• AI framework: Tabular Q-learning (easy to upgrade to DQN)

Run tips
--------
1) Train a Q-table policy:      python snake_qlearning_numpy_pygame.py --train --episodes 2000 --width 20 --height 20
2) Visualize the learned policy: python snake_qlearning_numpy_pygame.py --play --width 20 --height 20
   (Both use ./qtable.npy by default.)

Dependencies: numpy, pygame (pip install numpy pygame)
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# -----------------------------
# 1) PURE NUMPY GRID ENVIRONMENT
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
    def __init__(self, w: int = 20, h: int = 20, seed: int | None = None):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.n_actions = 3
        self.max_states = 2 ** 11
        self.reset()

    def reset(self) -> int:
        cx, cy = self.w // 2, self.h // 2
        self.direction = RIGHT
        self.snake: List[Tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self._place_food()
        self.score = 0
        self.steps_since_food = 0
        self.hunger_limit = self.w * self.h * 2
        return self._encode_state()

    def step(self, action: Action) -> StepResult:
        assert action in (0, 1, 2)
        if action == 1:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4

        dx, dy = DIRS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        reward = -0.01
        done = False

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

        return StepResult(self._encode_state(), reward, done, {})

    def _place_food(self) -> None:
        free = {(x, y) for x in range(self.w) for y in range(self.h)} - set(self.snake)
        self.food = self.rng.choice(list(free))

    def _is_collision(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return True
        if pos in self.snake:
            return True
        return False

    def _encode_state(self) -> int:
        head = self.snake[0]

        def cell_in_dir(dir_idx: int) -> Tuple[int, int]:
            dx, dy = DIRS[dir_idx]
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
        s = 0
        for i, b in enumerate(bits):
            s |= (b & 1) << i
        return s


# -----------------------------
# 2) Q-LEARNING AGENT
# -----------------------------

class QAgent:
    def __init__(self, n_states: int, n_actions: int, alpha=0.1, gamma=0.9,
                 eps=1.0, eps_min=0.01, eps_decay=0.995):
        self.q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def act(self, state: int) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(self.q.shape[1])
        return int(np.argmax(self.q[state]))

    def learn(self, s: int, a: int, r: float, s_next: int, done: bool):
        target = r
        if not done:
            target += self.gamma * np.max(self.q[s_next])
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def update_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# -----------------------------
# 3) TRAINING + ACCURACY REPORT
# -----------------------------

def train_q(env: SnakeEnv, episodes=2000, max_steps=2000, save_path="qtable.npy"):
    agent = QAgent(env.max_states, env.n_actions)
    scores = []
    success_count = 0  # count episodes where snake scores >= 1

    for ep in range(episodes):
        s = env.reset()
        total_r = 0.0
        for t in range(max_steps):
            a = agent.act(s)
            res = env.step(a)
            agent.learn(s, a, res.reward, res.state, res.done)
            s = res.state
            total_r += res.reward
            if res.done:
                break
        agent.update_eps()
        scores.append(env.score)

        if env.score >= 1:
            success_count += 1

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1:4d} | score={env.score:3d} | avg(last100)={np.mean(scores[-100:]):.2f} | eps={agent.eps:.3f}")

    # Save Q-table
    np.save(save_path, agent.q)
    print(f"Saved Q-table -> {save_path}")

    # Calculate accuracy
    accuracy = (success_count / episodes) * 100
    print(f"\nTraining completed over {episodes} episodes")
    print(f"Accuracy (episodes with score >=1): {accuracy:.2f}%")

    return agent


# -----------------------------
# 4) VISUALIZATION (PYGAME)
# -----------------------------

def play_with_qtable(env: SnakeEnv, q_path="qtable.npy", fps=12, cell=30):
    import pygame
    q = np.load(q_path)
    pygame.init()
    W, H = env.w * cell, env.h * cell
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    def draw_rect(pos, color):
        x, y = pos
        pygame.draw.rect(screen, color, (x * cell, y * cell, cell - 1, cell - 1))

    running, s = True, env.reset()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        a = int(np.argmax(q[s]))
        res = env.step(a)
        s = res.state

        screen.fill((30, 30, 30))
        for gx in range(env.w):
            for gy in range(env.h):
                pygame.draw.rect(screen, (45, 45, 45), (gx * cell, gy * cell, cell, cell), 1)
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
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    env = SnakeEnv(args.width, args.height, seed=args.seed)
    if args.train:
        train_q(env, episodes=args.episodes)
    if args.play:
        play_with_qtable(env)
    if not args.train and not args.play:
        print(__doc__)


if __name__ == "__main__":
    main()
