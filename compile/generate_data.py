""" The main goal is to take an environment, get an expert policy for it, and save a .pkl
    dump of a list of Trajectory objects, each of which corresponds to an expert rollout.
    
    The environments are:
        - [*] LunarNav - custom version of LunarLander with very low gravity. See lander_nogravity.py.
        - [*] Parking - see highway_env package
    
    [*] = currently active in this project.
"""

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from tqdm import tqdm
from torch.distributions import Categorical
from torch import Tensor as T
from stable_baselines3 import SAC
from imitation.data.types import TrajectoryWithRew
from envs import highway_env
from datetime import datetime
dt = datetime.today()
timestamp = str(int(dt.timestamp()))

random.seed(0)
np.random.seed(0)

# Parking Environment
def get_parking_rollouts(num_episodes=10000):

    # Load policy
    env = gym.make("NewParking-v0")
    model = SAC.load("experts/041222_her_sac_parking", env)

    # Get rollouts
    trajectories = []
    renders = []
    for i in tqdm(range(num_episodes)):
        seed = i % 3000
        env.seed(seed)
        env.reset()
        done = False
        obss = []
        acts = []
        rews = []
        infos = []
        env.seed(seed)
        obs = env.reset()
        while True:
            obss.append(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1))
            a = model.predict(obs, deterministic=True)
            acts.append(a[0])
            obs, reward, done, info = env.step(a[0])
            rews.append(reward)
            infos.append({"seed":seed})
            if done:
                break
        obss.append(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1))
        obss = np.array(obss)
        acts = np.array(acts)
        rews = np.array(rews)
        trajectories.append(TrajectoryWithRew(obs=obss, acts=acts, rews=rews, infos=infos))

    random.shuffle(trajectories)
    split_size = int(0.25*num_episodes)
    eval_trajectories = trajectories[:split_size] 
    valid_trajectories = trajectories[split_size:2*split_size]
    train_trajectories = trajectories[2*split_size:] 
    # Write rollouts
    with open("expert-rollouts/parking_train_"+timestamp+".pkl", "wb") as f:
        pickle.dump(train_trajectories, f)
    with open("expert-rollouts/parking_valid_"+timestamp+".pkl", "wb") as f:
        pickle.dump(valid_trajectories, f)
    with open("expert-rollouts/parking_eval_"+timestamp+".pkl", "wb") as f:
        pickle.dump(eval_trajectories, f)

get_parking_rollouts(num_episodes=100) 
