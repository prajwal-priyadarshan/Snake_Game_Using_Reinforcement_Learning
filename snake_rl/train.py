# -----------------------------
# 3) TRAINING
# -----------------------------

import torch
from dqn import DQNAgent
from env import SnakeEnv

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