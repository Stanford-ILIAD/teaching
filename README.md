# Assistive Teaching of Motor Control Tasks to Humans

Authors: Megha Srivastava (@meghabyte), Erdem Biyik, Suvir Mirchandani, Noah Goodman, and Dorsa Sadigh

This contains the source code for the NeurIPS 2022 paper "Assistive Teaching of Motor Control Tasks to Humans". In this work, we propose an AI-assisted teaching algorithm to break down any motor task into (i) teachable skills, (ii) construct novel drill sequences, and (iii) individualize curricula to students with different capabilities. We conduct an extensive mix of synthetic and user studies in two domains: parking a car with a joystick in a simulated environment, and learning to write Balinese characters with a computer mouse. 
Please contact megha@cs.stanford.edu with any questions! 

## Repository Overview

Code for extracting skills with CompILE is in the ```compile/``` directory. Specifically:
* ```generate_data.py``` generates expert data rollouts for environments using either an RL expert (Parking) or randomly generated character sequences (Writing)
* ```train_compile.py``` trains a CompILE module
  * an example command for the Parking task is: ```python train_compile.py  --iterations=5 --rollouts_path_train=expert-rollouts/parking_train_1683450947.pkl --rollouts_path_eval=expert-rollouts/parking_eval_1683450947.pkl --latent_dist concrete --latent_dim 4 --num_segments 4 --cont_action_dim 2 --prior_rate 10 --mode state+action --run_name parking-concrete-4d-state1 --state_dim 12 --beta_s 1```
  * an example command for the Writing task is: ```python train_compile.py --iterations 5 --rollouts_path_train expert-rollouts/drawing_train_1686529312.pkl --rollouts_path_eval expert-rollouts/drawing_eval_1686529312.pkl  --latent_dist concrete --latent_dim 24 --num_segments 8 --mode action --run_name drawing-concrete-24d-action0 --batch_size 1 --learning_rate 0.01 --beta_s 0.0```
* ```parking_viz.py``` visualizes skills from a trained CompILE module over a dataset of rollouts

## Writing Task Overview
For the Writing task, we use character trajectories from the [Omniglot dataset](https://github.com/brendenlake/omniglot). All pre-processing and utility functions are located in the ```lib/omniglot/``` sub-directory. Specifically: 
* For our paper, we selected a set of Balinese characters which are located (both .txt stroke and .png image files) in the directory. One can simply copy-paste these files for new characters into the same directory. 
* ```digits_utils.py``` defines many useful utility functions for creating trajectory sequences out of stroke data from the original data. Importantly, because we deal with non-Roman alphabet characters, the global variables CHARACTER_FNS and FAKE_FNS set the correspondence between characters from the Omniglot dataset and Roman alphabet characters we can use to represent them to ease coding / de-bugging. 
* ```omniglot_runner.py``` generates two files: [1] ```final_chars_dict``` that mapes the Roman alphabet characters to the underlying Omniglot character trajectories, with all trajectory-infilling and pre-processing added, [2] ```1k_words_for_compile```, a set of randomly-generated "words", or character sequences, used to train CompILE. The current generation procedure sets different likelihoods for character sequences -- we did not optimize these, and it would be interesting to thoroughly study how the sequence distribution affects CompILE performance in future work. 
