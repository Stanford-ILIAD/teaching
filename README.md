# Assistive Teaching of Motor Control Tasks to Humans

Authors: Megha Srivastava (@meghabyte), Erdem Biyik, Suvir Mirchandani, Noah Goodman, and Dorsa Sadigh

This contains the source code for the NeurIPS 2022 paper "Assistive Teaching of Motor Control Tasks to Humans". In this work, we propose an AI-assisted teaching algorithm to break down any motor task into (i) teachable skills, (ii) construct novel drill sequences, and (iii) individualize curricula to students with different capabilities. We conduct an extensive mix of synthetic and user studies in two domains: parking a car with a joystick in a simulated environment, and learning to write Balinese characters with a computer mouse. 
Please contact megha@cs.stanford.edu with any questions! 

## Repository Overview

Code for extracting skills with CompILE is in the compile/ directory. Specifically:
* ```generate_data.py``` generates expert data rollouts for environments using an RL expert (e.g. Parking)
* ```train_compile.py``` trains a CompILE module
  * an example command for the parking environment is: ```python train_compile.py  --iterations=5 --rollouts_path_train=expert-rollouts/parking_train_1683450947.pkl --rollouts_path_eval=expert-rollouts/parking_eval_1683450947.pkl --latent_dist concrete --latent_dim 4 --num_segments 4 --cont_action_dim 2 --prior_rate 10 --mode state+action --run_name parking-concrete-4d-state1 --state_dim 12 --beta_s 1```
* ```parking_viz.py``` visualizes skills from a trained CompILE module over a dataset of rollouts
