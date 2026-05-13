import random
import torch
import numpy as np
import os
from src.utils.util import parse_arguments



random.seed(0)
torch.manual_seed(0)
opt = parse_arguments()
cuda = torch.cuda.is_available() and (not opt.use_CPU)
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
rng = np.random.default_rng(opt.random_seed)

noisy_data = opt.noisy_data


VALIDATION_DATA_SPLIT = 0.2
validation_data = None
if isinstance(noisy_data) == list:
    random.shuffle(noisy_data)
    training_data = noisy_data[:int(len(noisy_data)*(1-VALIDATION_DATA_SPLIT))]
    validation_data = noisy_data[int(len(noisy_data)*(1-VALIDATION_DATA_SPLIT)):]
else:
    training_data = noisy_data

os.makedirs(opt.results_dir + f"/saved_models/{opt.exp_name}", exist_ok=True)
os.makedirs(opt.results_dir + f"/logs/{opt.exp_name}", exist_ok=True)
