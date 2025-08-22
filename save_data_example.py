import os
os.environ['MUJOCO_GL'] = 'glfw'  # Set the rendering backend to 'glfw'
import numpy as np
import robosuite as suite
import time
from robosuite.wrappers.gym_wrapper import GymWrapper
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC

# Environment and camera settings
envName = "Lift"
cameraName = "front-sideview"
env = suite.make(
    env_name=envName,
    has_renderer=False,
    camera_names=cameraName,
    render_camera=cameraName,
    robots="Panda",
    camera_heights=224,
    camera_widths=224,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    reward_shaping=True
)

# Model loading
model = SAC.load("./path/to/your/model.zip")


LOAD_START_POS = True
save_dir = "data/train" # You will need one for validation as well

os.makedirs(save_dir, exist_ok=True)

EPISODE_LENGTH = 100
N_EPISODES = 1000

instruction = "reach the cube"
# instruction = "grasp the cube"


prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# Find existing episode numbers
existing_files = {
    int(fname.split('_')[-1].split('.')[0])
    for fname in os.listdir(save_dir)
    if fname.startswith("reach_episode_") and fname.endswith(".npy")
}

total_existing = len(existing_files)
if total_existing >= N_EPISODES:
    print(f"{total_existing} episodes already exist. Nothing to generate.")
    env.close()
    exit()

ep = 0
saved = 0

while (saved + total_existing) < N_EPISODES:
    if ep in existing_files:
        ep += 1
        continue

    episode = []
    obs = env.reset()

    # episode_file = os.path.join(start_save_dir, f"episode_{ep}.npy")


    for step in range(EPISODE_LENGTH):

        img = np.asarray(obs[cameraName+"_image"], dtype=np.uint8) # expected format for openvla
        img = np.flipud(img) # Flip the image vertically to match the original orientation (Mine was rendered upside down) you might not need this
        # Store step data

        action, _ = model.predict(np.concatenate((obs['object-state'], obs['robot0_proprio-state'])))

        episode.append({
            'image': img,
            'action': np.array(action, dtype=np.float32),
            'language_instruction': prompt
        })
        obs, reward, done, _ = env.step(action)


        if done:
            print("done at step", step)
            break

    np.save(os.path.join(save_dir, f"reach_episode_{ep}.npy"), episode)
    print(f"Saved reach_episode_{ep}.npy")
    saved += 1
    ep += 1

env.close()
print(f"Saved {saved} new episodes. Total now: {saved + total_existing}")