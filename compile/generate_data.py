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
from lib.omniglot.digits_utils import get_data_sequence
from envs import highway_env
from datetime import datetime
dt = datetime.today()
timestamp = str(int(dt.timestamp()))

random.seed(0)
np.random.seed(0)

# Writing Environment
def get_drawing_rollouts(data_file=""):
    # Load policy
    word_sequences = pickle.load(open(data_file,"rb"))
    alphabet = pickle.load(open("lib/omniglot/final_chars_dict", "rb"))
    # Get rollouts
    trajectories = []
    max_state = 0
    for t in tqdm(range(min(500, len(word_sequences)))):
        seed = t
        word = word_sequences[seed]
        states, rews, actions = get_data_sequence(alphabet, word, width=500, height=300)
        infos = [{"seed":seed} for i in range(len(states))]
        if(len(states) > max_state):
            max_state = len(states)
        trajectories.append(TrajectoryWithRew(obs=states, acts=actions[:-1], rews=np.array(rews[:-1]), infos=infos[:-1]))
    random.shuffle(trajectories)
    split_size = int(0.5*len(trajectories))
    eval_trajectories = trajectories[:split_size]
    train_trajectories = trajectories[split_size:] 
    # Write rollouts
    with open("expert-rollouts/drawing_train_"+timestamp+".pkl", "wb") as f:
        pickle.dump(train_trajectories, f)
    with open("expert-rollouts/drawing_eval_"+timestamp+".pkl", "wb") as f:
        pickle.dump(eval_trajectories, f)
    
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
get_drawing_rollouts(data_file="lib/omniglot/1k_words_for_compile.pkl")