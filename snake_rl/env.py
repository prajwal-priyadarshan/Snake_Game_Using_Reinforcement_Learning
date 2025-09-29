import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
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

        reward = -0.01
        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        if new_dist < old_dist:
            reward += 0.1
        else:
            reward -= 0.03

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