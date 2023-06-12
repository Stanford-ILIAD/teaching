"""
Script for processing Omniglot digits, and creating action trajectories for different character sequences. 
"""
import numpy as np
import sys
import os
import random
import torch
import math
import pickle
from sys import platform as sys_pf
import matplotlib
if sys_pf == 'darwin':
	matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath('..'))
from writing_utils import *

# these refer to file IDs in the original Omniglot dataset
CHARACTER_FNS = ["0109_04","0115_03","0119_04","0120_02","0128_02"]
# assign fake Roman alphabet letters to character fns to simplify debugging
FAKE_FNS = ["a","b","c","t","e"]

def plot_motor_to_image(I,drawing,lw=2):
	drawing = [d[:,0:2] for d in drawing] # strip off the timing data (third column)
	drawing = [space_motor_to_img(d) for d in drawing] # convert to image space
	plt.imshow(I,cmap='gray')
	ns = len(drawing)
	for sid in range(ns): # for each stroke
		plot_traj(drawing[sid],get_color(sid),lw)
	plt.xticks([])
	plt.yticks([])


def plot_traj(stk,color,lw):
	n = stk.shape[0]
	if n > 1:
		plt.plot(stk[:,0],stk[:,1],color=color,linewidth=lw)
	else:
		plt.plot(stk[0,0],stk[0,1],color=color,linewidth=lw,marker='.')

def get_color(k):	
    scol = ['r','g','b','m','c']
    ncol = len(scol)
    if k < ncol:
       out = scol[k]
    else:
       out = scol[-1]
    return out

# convert to str and add leading zero to single digit numbers
def num2str(idx):
	if idx < 10:
		return '0'+str(idx)
	return str(idx)

# Load binary image for a character
#
# fn : filename
def load_img(fn):
	I = plt.imread(fn)
	I = np.array(I,dtype=bool)
	return I

# Load stroke data for a character from text file
#
# Input
#   fn : filename
#
# Output
#   motor : list of strokes (each is a [n x 3] numpy array)
#      first two columns are coordinates
#	   the last column is the timing data (in milliseconds)
def load_motor(fn):
	motor = []
	with open(fn,'r') as fid:
		lines = fid.readlines()
	lines = [l.strip() for l in lines]
	for myline in lines:
		if myline =='START': # beginning of character
			stk = []
		elif myline =='BREAK': # break between strokes
			stk = np.array(stk)
			motor.append(stk) # add to list of strokes
			stk = [] 
		else:
			arr = np.fromstring(myline,dtype=float,sep=',')
			stk.append(arr)
	return motor

#
# Map from motor space to image space (or vice versa)
#
# Input
#   pt: [n x 2] points (rows) in motor coordinates
#
# Output
#  new_pt: [n x 2] points (rows) in image coordinates
def space_motor_to_img(pt):
	pt[:,1] = -pt[:,1]
	return pt
def space_img_to_motor(pt):
	pt[:,1] = -pt[:,1]
	return

def get_chars_dict():
    """
    Created dictionary that maps an Omniglot character to its trajectory using the corresponding fake filename.
    Fake character names are from the Roman alphabet and picked for convenience in de-bugging. 
    """
    img_dir = 'images_background'
    stroke_dir = 'strokes_background'
    full_chars_dict={}
    for fni in range(len(CHARACTER_FNS)):
        fn = CHARACTER_FNS[fni]
        fake_fn = FAKE_FNS[fni]
        fn_stk =  fn + '.txt'
        fn_img = fn + '.png'			
        char_motor = load_motor(fn_stk)
        char_image = load_img(fn_img)
        drawing = char_motor
        drawing = [d[:,0:2] for d in drawing] # strip off the timing data (third column)
        drawing = [space_motor_to_img(d) for d in drawing] # convert to image space
        drawing_x = list(drawing[0][:,0])
        drawing_y = list(drawing[0][:,1])
        curr_scale = DRAWING_SCALE_WIDTH/(max(drawing_x)-min(drawing_x))
        drawing_x = [dx*curr_scale for dx in drawing_x]
        drawing_y = [dy*curr_scale for dy in drawing_y]
        yi=0
        count = 0
        while(True):
            if(yi >= len(drawing_y)-1):
                break
            if(abs(drawing_y[yi+1]-drawing_y[yi]) > 1):
                x_new = (drawing_x[yi+1]+drawing_x[yi])/2
                y_new = (drawing_y[yi+1]+drawing_y[yi])/2
                drawing_x.insert(yi+1, x_new)
                drawing_y.insert(yi+1, y_new)
            elif(abs(drawing_x[yi+1]-drawing_x[yi]) > 1):
                x_new = (drawing_x[yi+1]+drawing_x[yi])/2
                y_new = (drawing_y[yi+1]+drawing_y[yi])/2
                drawing_x.insert(yi+1, x_new)
                drawing_y.insert(yi+1, y_new)
            else:
                yi = yi+1
        if(drawing_y[-1] > drawing_y[0]):
            for yi in range(int(drawing_y[0]), math.ceil(drawing_y[-1]), 1):
                drawing_x.insert(0, drawing_x[0])
                drawing_y.insert(0, yi)
        while(len(drawing_y) < DRAWING_TRAJ_LEN):
            random_interpol = np.random.randint(len(drawing_y)-1)
            x_new = (drawing_x[random_interpol+1]+drawing_x[random_interpol])/2
            y_new = (drawing_y[random_interpol+1]+drawing_y[random_interpol])/2
            drawing_x.insert(random_interpol, x_new)
            drawing_y.insert(random_interpol, y_new)
        actions_x = [drawing_x[i+1]-drawing_x[i] for i in range(len(drawing_x)-1)]
        actions_y = [drawing_y[i+1]-drawing_y[i] for i in range(len(drawing_y)-1)]
        states_x = [0]
        states_y = [0]
        for ai in range(len(actions_x)):
            actions_x[ai] = actions_x[ai]
            actions_y[ai] = actions_y[ai]
            states_x.append(states_x[-1]+(actions_x[ai]))
            states_y.append(states_y[-1]+(actions_y[ai]))
        char_dict = {"states":[(states_x[i], states_y[i]) for i in range(len(states_x))],
"actions": [(actions_x[i], actions_y[i]) for i in range(len(actions_x))]}
        full_chars_dict[fake_fn]=char_dict
    return full_chars_dict



def create_line_sequence_dict(skill_len):
    states_x=[0]
    states_y=[0]
    actions_x = []
    actions_y = []
    for i in range(skill_len):
        actions_x.append(1)
        actions_y.append(0)
        states_x.append(states_x[-1]+(actions_x[-1]))
        states_y.append(states_y[-1]+(actions_y[-1]))
    char_dict = {"states":[(states_x[i], states_y[i]) for i in range(len(states_x))],
"actions": [(actions_x[i], actions_y[i]) for i in range(len(actions_x))]}
    return char_dict

def create_v_sequence_dict(skill_len):
    states_x=[0]
    states_y=[0]
    actions_x = []
    actions_y = []
    for i in range(int(skill_len/2)):
        actions_x.append(1)
        actions_y.append(-2)
        states_x.append(states_x[-1]+(actions_x[-1]))
        states_y.append(states_y[-1]+(actions_y[-1]))
    for i in range(int(skill_len/2), skill_len):
        actions_x.append(1)
        actions_y.append(2)
        states_x.append(states_x[-1]+(actions_x[-1]))
        states_y.append(states_y[-1]+(actions_y[-1]))
    char_dict = {"states":[(states_x[i], states_y[i]) for i in range(len(states_x))],
"actions": [(actions_x[i], actions_y[i]) for i in range(len(actions_x))]}
    return char_dict

def generate_skill_dict():
    full_chars_dict = get_chars_dict()
    full_chars_dict["-"] = create_line_sequence_dict(skill_len=20)
    full_chars_dict["^"] = create_v_sequence_dict(skill_len=40)
    return full_chars_dict

def display_full_sequence(full_chars_dict, word, width=500, height=300):
    return display_subsequence(full_chars_dict, word, start=0, stop=None)

def display_subsequence(full_chars_dict, word, start, stop, width=500, height=300):
    states, rews, actions = get_data_sequence(full_chars_dict, word)
    start_x = int(states[start][0])
    start_y = int(states[start][1])
    full_img, full_actions = display_fixed_actions(actions[start:stop], start_x = start_x, start_y = start_y, width=500, height=300)
    return full_img, states[start:stop]

def display_fixed_actions(actions, start_x = 0, start_y = 150, width=500, height=300):
    full_img = np.ones((width,height))
    full_img[start_x][start_y] = 0.3
    curr_x = start_x
    curr_y = start_y
    for a in actions:
        curr_x = curr_x+a[0]
        curr_y = curr_y+a[1]
        if(curr_x < 500 and curr_y < 300 and curr_x > 0 and curr_y > 0):
            full_img[int(curr_x)][int(curr_y)]=0.3 
    full_img = full_img.transpose()
    return full_img, actions 

def get_data_sequence(full_chars_dict, word, width=500, height=300):
    chars_list = ["a","b","c","t","e", "-", "^"]
    sequence = list(word)
    #create goal info for state
    goal = []
    for goal_i in range(DRAWING_MAX_TRAJ):
        if(goal_i<len(sequence)):
            goal.append(chars_list.index(sequence[goal_i]))
        else:
            goal.append(-1)
    start_x = 0
    start_y = int(height/2)
    curr_x = start_x
    curr_y = start_y
    data_states = []
    data_actions = []
    data_rews = []
    timer = 0
    for s in sequence:
        actions = full_chars_dict[s]["actions"]
        for i in range(len(actions)):
            data_states.append([curr_x, curr_y, timer]+goal) #update May: added i for timestep
            data_actions.append([actions[i][0], actions[i][1]])
            data_rews.append(len(actions[i+1:]))
            curr_x = curr_x+actions[i][0]
            curr_y = curr_y+actions[i][1]
            timer += 1
    return torch.FloatTensor(data_states), torch.FloatTensor(data_rews), torch.FloatTensor(data_actions)


				