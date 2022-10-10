""" Adapted from https://github.com/tkipf/compile
Example call: 
python train_compile.py  --iterations=5 --rollouts_path_train=expert-rollouts/parking_train_1683450947.pkl --rollouts_path_eval=expert-rollouts/parking_eval_1683450947.pkl --latent_dist concrete --latent_dim 4 --num_segments 4 --cont_action_dim 2 --prior_rate 10 --mode state+action --run_name parking-concrete-4d-state1 --state_dim 12 --beta_s 1
"""

import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

import utils
import modules
from dataset import TrajectoryDatasetWithRew, pad_collate
from arguments import args, device


#set random seeds
torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)  

model = modules.CompILE(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
dl_train = DataLoader(TrajectoryDatasetWithRew(args.rollouts_path_train, args), collate_fn=pad_collate, batch_size=args.batch_size)
dl_eval = DataLoader(TrajectoryDatasetWithRew(args.rollouts_path_eval, args), collate_fn=pad_collate, batch_size=args.batch_size)


# Train model.
print('Training model...')
print("Device: " + str(device))
best_valid_loss_same = 0
curr_valid_loss = float('inf')
for step in range(args.iterations):
    train_loss = 0
    batch_num = 0
    dl_iter_train = iter(dl_train)
    for batch in dl_iter_train:
        states, actions, rewards, lengths, seeds = batch

        # Run forward pass.
        model.train()
        outputs = model.forward(states, actions, lengths)
        loss, nll, kl_z, kl_b = utils.get_losses(states, actions, outputs, args)

        train_loss += nll.item() # This is just the NLL loss (without regularizers) - #TODO: Log all the terms
        batch_num += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= batch_num

    if step % args.log_interval == 0:
        # Run evaluation.
        model.eval()
        
        dl_iter_eval = iter(dl_eval)
        total_nll = 0.
        total_acc = 0.
        total_count = 0
        for batch in dl_iter_eval:
            states, actions, rewards, lengths, seeds = batch
            count = len(states)
            outputs = model.forward(states, actions, lengths)
            _, nll, _, _ = utils.get_losses(states, actions, outputs, args)
            acc, rec = utils.get_reconstruction_accuracy(states, actions, outputs, args)
            total_nll += nll.item() * count
            total_acc += acc.item() * count
            total_count += count

        # Accumulate metrics.
        eval_acc = total_acc / count
        eval_nll = total_nll / count
        if(eval_nll <= curr_valid_loss):
            curr_valid_loss =  eval_nll
            best_valid_loss_same = 0
        else:
            best_valid_loss_same += 1
    if step % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))

    if(best_valid_loss_same >  50):
        break
