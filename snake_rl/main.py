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