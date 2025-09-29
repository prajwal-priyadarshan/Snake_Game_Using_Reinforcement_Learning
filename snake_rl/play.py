# -----------------------------
# 4) PLAY MODE (NO EPS)
# -----------------------------

import torch
import pygame
from dqn import DQNAgent

def play_with_dqn(env, model_path="dqn.pth", fps=12, cell=30):
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

        a = agent.act(s, greedy=True)
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