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

# -----------------------------
# 5) MAIN
# -----------------------------


import argparse
from env import SnakeEnv
from train import train_dqn
from play import play_with_dqn

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