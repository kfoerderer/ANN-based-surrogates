import os
import numpy as np

from modules.simulation.individual.bess import BESS

####
# general hyperparameters
####
# simulation
time_step = 15 * 60 # seconds
n_actions = 201

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
# model parameters and model
####

allow_infeasible_actions = True
model = BESS(time_step, BESS.create_action_set(-4600, 4600, 201), 9*1000*60*60, 0.83, 1, 0.0075, correct_infeasible=allow_infeasible_actions)

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
input_processor.one_hot(model.actions) # action

output_processor = BatchProcessor()
output_processor.normalize([0], [model.capacity]) # soc
#output_processor.discretize_index(np.arange(1001)/1000.0*model.capacity) # soc
output_processor.none(2) # soc_min, soc_max
output_processor.discretize_index(-model.actions) # el. interaction [minus, since the interaction is the negative action (for a feasible action)]
output_processor.discretize_index([0]) # th. interaction

losses = []
losses.append((nn.MSELoss().to(device), 1)) # soc
losses.append((nn.MSELoss().to(device), 2)) # soc_min, soc_max
#losses.append((nn.CrossEntropyLoss().to(device), 1001))
losses.append((nn.CrossEntropyLoss().to(device), n_actions)) # el interaction
losses.append((nn.CrossEntropyLoss().to(device), 1)) # th interaction
loss_weights = [1e6,1,1,1]

ann_output_processor = BatchProcessor()
ann_output_processor.denormalize([0], [model.capacity]).clip(0, model.capacity) # soc
ann_output_processor.none(2) # soc_min, soc_max
ann_output_processor.mode_of_distribution(-model.actions) # el. interaction
ann_output_processor.mode_of_distribution([0]) # th. interaction
ann_output_processor.split_result([3])

sampling_parameters = {
    'soc_distribution': ([(0,0.25), (0.25,0.75), (0.75,1)], [3./8, 2./8, 3./8]),
    'infeasible_chance': 2./3,
}

####
# sample generator
####
from modules.neuralnetwork.samplegenerators.statebased import SampleGenerator

sample_generator = SampleGenerator(model, input_processor, output_processor, sampling_parameters, False, True, True, ann_output_processor)
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
        'loss': [MixedLoss(losses, loss_weights, device=device)],
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
        'skips': ([0]*14+[1], (max_layers, max_layers)),

        'early_stopping_callback': [EarlyStoppingCallback({
        }, 100)],
        #'lr_scheduler': [(torch.optim.lr_scheduler.StepLR,{'step_size':25,'gamma':0.75}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.99})]
        'lr_scheduler': [(torch.optim.lr_scheduler.StepLR,{'step_size':25,'gamma':0.5}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.975}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.99})]
    }


####

output_location = '{}/output/{{}}_gc_bess_state'.format(os.path.dirname(__file__)) 