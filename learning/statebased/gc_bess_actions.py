import os
import numpy as np

from modules.simulation.individual.bess import BESS

# simulation
from .gc_bess_state import time_step, n_actions, model

####
# general hyperparameters
####

# training
cache_initialization_process_count = 24
training_generation_process_count = 8
evaluation_generation_process_count = 0
training_process_count = 2

batch_size = 1024 * 3
training_cache_size = batch_size * 1000
evaluation_cache_size = batch_size * 100
evaluation_batch_count = evaluation_cache_size / batch_size

# meta
meta_search_sample_count = 16
meta_search_fully_stored_count = 1
meta_search_parameter_space = {} # see below

####
# sample generation parameters and ANN loss
####
import torch
import torch.nn as nn

# determine torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from modules.neuralnetwork.samplegenerators.statebased import BatchProcessor

input_processor = BatchProcessor()
input_processor.normalize([0], [model.capacity]) # soc
input_processor.none(2) # soc_min, soc_max

output_processor = None # handled by sample generator
losses = [(nn.BCEWithLogitsLoss().to(device), n_actions)]

ann_output_processor = BatchProcessor()
ann_output_processor.sigmoid(n_actions)

sampling_parameters = {
    'soc_distribution': ([(0,0.25), (0.25,0.75), (0.75,1)], [3./8, 2./8, 3./8]),
    'infeasible_chance': 2./3,
}

####
# sample generator
####
from modules.neuralnetwork.samplegenerators.statebased import SampleGenerator

# determine torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sample_generator = SampleGenerator(model, input_processor, output_processor, sampling_parameters, True, False, False, ann_output_processor)
sample_input, sample_output = next(sample_generator)
sample_input_size = sample_input.size(1)
sample_output_size = sample_output.size(1)

####
# ANN
####
from modules.neuralnetwork.loss import MixedLoss, SigmoidGatedL1RegularizationLoss, L1RegularizationLoss
from modules.neuralnetwork.training import EarlyStoppingCallback

# this is not necessarily the same as sample_output_size, since dimension of target (sample output) and estimation (ANN output) can differ
output_width = sum([loss[1] for loss in losses])

max_layers = 10
meta_search_parameter_space = {
        'batch_size': [batch_size],
        'loss': [MixedLoss(losses, device=device)],
        'regularization': [
            L1RegularizationLoss(2e-7, device),
            L1RegularizationLoss(2e-8, device),
            L1RegularizationLoss(2e-9, device),
            ],
        'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
        'epoch_count': [1000],
        'batch_count': [training_cache_size / batch_size],
        'max_grad_norm': [1e6],

        'input_width': [sample_input_size],
        'output_width': [output_width],
        'output_activation': [None],

        'hidden_layer_count': np.arange(min(max_layers-1,4),max_layers), # 'min(.,.)' is a failsafe
        'width': [2**i for i in range(5,11)],
        'width_interpolation_steps_input': [0],
        'width_interpolation_steps_output': [0,3,4,5],
        'betas': ([0.5, 1, 2], max_layers),
        'batch_norms': ([0]*14+[1], max_layers),
        'dropout': ([0], max_layers), #([0]*28+[0.2]+[0.5], max_layers),
        'skips': ([0]*19+[1], (max_layers, max_layers)),

        'early_stopping_callback': [EarlyStoppingCallback({
        }, 100)],
        'lr_scheduler': [(torch.optim.lr_scheduler.StepLR,{'step_size':25,'gamma':0.75}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.95}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.99})]
    }


####

output_location = '{}/output/{{}}_gc_bess_actions'.format(os.path.dirname(__file__)) 