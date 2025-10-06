import streamlit as st
import time
import torch
from dqn import DQNAgent
from env import SnakeEnv
from PIL import Image, ImageDraw

# -----------------------------
# Streamlit Settings
# -----------------------------
st.set_page_config(page_title="Snake RL Demo", layout="wide")
st.title("Snake RL - DQN Visualization")

# -----------------------------
# Sidebar options
# -----------------------------
model_path = st.sidebar.text_input("Trained Model Path", "dqn.pth")
grid_width = st.sidebar.number_input("Grid Width", min_value=10, max_value=60, value=20)
grid_height = st.sidebar.number_input("Grid Height", min_value=10, max_value=40, value=20)
cell_pixels = st.sidebar.number_input("Cell Size (px)", min_value=10, max_value=50, value=25)
fps = st.sidebar.slider("FPS (Game speed)", min_value=1, max_value=30, value=10)

level = st.sidebar.selectbox("Select Level (Initial Snake Size)", ["Level 1", "Level 2", "Level 3"])
level_sizes = {"Level 1": 3, "Level 2": 6, "Level 3": 9}
initial_snake_size = level_sizes[level]

start_button = st.sidebar.button("Start Game")

# -----------------------------
# Helper: Draw the game as an image
# -----------------------------
def draw_game(env, cell=25):
    img_size = (env.w * cell, env.h * cell)
    img = Image.new("RGB", img_size, (30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Draw grid
    for x in range(env.w):
        for y in range(env.h):
            draw.rectangle(
                [x*cell, y*cell, (x+1)*cell-1, (y+1)*cell-1],
                outline=(45, 45, 45)
            )

    # Draw food
    fx, fy = env.food
    draw.rectangle([fx*cell, fy*cell, (fx+1)*cell-1, (fy+1)*cell-1], fill=(200, 70, 70))

    # Draw snake
    for i, (sx, sy) in enumerate(env.snake):
        color = (70, 200, 90) if i == 0 else (90, 230, 120)
        draw.rectangle([sx*cell, sy*cell, (sx+1)*cell-1, (sy+1)*cell-1], fill=color)

    return img

# -----------------------------
# Run the game
# -----------------------------
if start_button:
    env = SnakeEnv(w=grid_width, h=grid_height, seed=None)
    
    # Initialize snake based on level
    cx, cy = env.w // 2, env.h // 2
    env.snake = [(cx-i, cy) for i in range(initial_snake_size)]
    env.direction = 1  # RIGHT
    env._place_food()
    env.score = 0
    env.steps_since_food = 0

    agent = DQNAgent()
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.target_net.load_state_dict(agent.q_net.state_dict())
    agent.q_net.eval()

    stframe = st.empty()  # Placeholder for game image

    # Columns for fixed labels
    col1, col2 = st.columns(2)
    score_placeholder = col1.empty()
    time_placeholder = col2.empty()

    # Fixed labels
    col1.markdown("<h3 style='text-align: center; color: white;'>Score</h3>", unsafe_allow_html=True)
    col2.markdown("<h3 style='text-align: center; color: white;'>Time</h3>", unsafe_allow_html=True)

    s = env._encode_state()
    done = False
    start_time = time.time()

    while not done:
        a = agent.act(s, greedy=True)
        res = env.step(a)
        s = res.state

        # Draw game
        img = draw_game(env, cell=cell_pixels)
        stframe.image(img)

        # Update numeric values
        elapsed_time = int(time.time() - start_time)
        score_placeholder.markdown(f"<h2 style='text-align: center; color: white;'>{env.score}</h2>", unsafe_allow_html=True)
        time_placeholder.markdown(f"<h2 style='text-align: center; color: white;'>{elapsed_time}s</h2>", unsafe_allow_html=True)

        time.sleep(1.0 / fps)

        if res.done:
            time.sleep(0.5)
            s = env.reset()
            
            # Reset snake for selected level
            env.snake = [(cx-i, cy) for i in range(initial_snake_size)]
            env.direction = 1  # RIGHT
            env._place_food()
            env.score = 0
            start_time = time.time()
            s = env._encode_state()
