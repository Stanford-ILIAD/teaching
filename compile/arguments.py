import torch
from tap import Tap
import time
import os
import shutil


class Arguments(Tap):
    iterations: int = 2000 # Number of training iterations
    learning_rate: float = 1e-3 # Learning rate
    hidden_dim: int = 64 # Number of hidden units
    latent_dim: int = 32 # Dimensionality of latent variables.
    latent_dist: str = 'gaussian' # Latent variable type -> "gaussian" or "concrete"
    batch_size: int = 100 # Mini-batch size (for averaging gradients)
    state_dim: int = 12 # Dimensionality of state
    num_actions: int = 5 # Number of distinct actions in data generation
    cont_action_dim: int = 2 # Dimensionality of the continuous action space (or 0 if discrete)
    num_segments: int = 4 # Number of segments in data generation
    prior_rate: int = 50  # Expected length of segments
    log_interval: int = 5 # Logging interval
    save_interval: int = 5 # Saving interval
    rollouts_path_train: str = None # Path to training rollouts
    rollouts_path_eval: str = None # Path to evaluation rollouts
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name: str = None # Name of run
    save_dir: str = None # Path to save (computed)
    beta_b: float = 0.1 # Weight on KL term for boundaries
    beta_z: float = 0.1 # Weight on KL term for latents
    beta_s: float = 0 # Weight on state reconstruction
    mode: str = "action" # What to embed/reconstruct -> action or state+action or statediff+action
    action_type: str = "continuous" # Action type -> discrete or continuous
    wb: bool = False # Record to wandb
    random_seed: int = 0 #Random seed

parser = Arguments()
args = parser.parse_args()

if args.run_name is None:
    args.run_name = "test_" + time.strftime("%H%M%S-%Y%m%d")

args.action_type = "continuous" if args.cont_action_dim > 0 else "discrete"

dir = "results/" + args.run_name
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
args.save_dir = dir

args.save(os.path.join(args.save_dir, 'config.json'))

device = args.device