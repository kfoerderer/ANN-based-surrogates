import os
import numpy as np

from modules.simulation.individual.chpp import CHPP
from modules.simulation.individual.hwt import HWT
from modules.simulation.individual.demand import Demand
from modules.simulation.integrated.chpp_hwt import CHPP_HWT

####
# general hyperparameters
####
# simulation
time_step = 15 * 60 # seconds
n_actions = 101

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
# read data
with open('data/crest_heat_demand.txt', 'r') as file:
    demand_series = np.loadtxt(file, delimiter='\t', skiprows=1) # read file, dismiss header
    demand_series = demand_series.transpose(1,0) # dim 0 identifies the series
    demand_series *= 1000 # kW -> W

allow_infeasible_actions = True
hwt_volume = 0.750
hwt_min_temp = 60.
hwt_max_temp = 80.
relative_loss = HWT.relative_loss_from_energy_label(12, 5.93, hwt_volume, 45)

state_matrix = [
    [(0,0)              , (-5500/2,-12500/2)],
    [(-5500/2,-12500/2) , (-5500,-12500)]
]
chpp = CHPP(time_step, CHPP.create_action_set(-5500, 0, n_actions), state_matrix, correct_infeasible=allow_infeasible_actions)
hwt = HWT(time_step, hwt_min_temp, hwt_max_temp, hwt_volume, 1, 1, relative_loss)
demand = Demand(time_step, demand_series)
model= CHPP_HWT(chpp, hwt, demand, 0.01) # train a safety margin of 1°C into the model (instead of adding a state variable)

####
# sample generation parameters and ANN loss
####
import torch
import torch.nn as nn

# determine torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from modules.neuralnetwork.samplegenerators.statebased import BatchProcessor

staying_times = [i * 900 for i in range(1,4*2+1)]

input_processor = BatchProcessor()
input_processor.normalize([0], [10000]) # demand
input_processor.one_hot([i for i in range(len(state_matrix[0]))]) # chpp mode
input_processor.one_hot(staying_times) # staying time
input_processor.one_hot(staying_times) # min off
input_processor.one_hot(staying_times) # min on
input_processor.normalize([hwt_min_temp, hwt.ambient_temperature], [(hwt_max_temp-hwt_min_temp), 1.]) # hwt temp, ambient temp
input_processor.one_hot(model.actions) # action

output_processor = BatchProcessor()
output_processor.discretize_index([0]) # demand
output_processor.discretize_index([i for i in range(len(state_matrix[0]))]) # chpp Mode
output_processor.discretize_index(staying_times) # staying time
output_processor.discretize_index(staying_times) # min off
output_processor.discretize_index(staying_times) # min on
output_processor.normalize([hwt_min_temp], [(hwt_max_temp-hwt_min_temp)]) # hwt temp
output_processor.discretize_index([0]) # ambient temp
output_processor.discretize_index(-model.actions) # el. interaction [minus, since the interaction is the negative action (for a feasible action)]
output_processor.discretize_index([0]) # th. interaction

losses = []
losses.append((nn.CrossEntropyLoss().to(device), 1)) # demand
losses.append((nn.CrossEntropyLoss().to(device), len(state_matrix[0]))) # chpp mode
losses.append((nn.CrossEntropyLoss().to(device), len(staying_times))) # staying time
losses.append((nn.CrossEntropyLoss().to(device), len(staying_times))) # min off
losses.append((nn.CrossEntropyLoss().to(device), len(staying_times))) # min on
losses.append((nn.MSELoss().to(device), 1)) # hwt_temp
losses.append((nn.CrossEntropyLoss().to(device), 1)) # ambient temp
losses.append((nn.CrossEntropyLoss().to(device), len(model.actions))) # el interaction
losses.append((nn.CrossEntropyLoss().to(device), 1)) # th interaction
loss_weights = [1,100,100,10,10,10000,1,1,1]

ann_output_processor = BatchProcessor()
ann_output_processor.mode_of_distribution([0]) #demand
ann_output_processor.mode_of_distribution([i for i in range(len(state_matrix[0]))]) # chpp Mode
ann_output_processor.mode_of_distribution(staying_times) # staying time
ann_output_processor.mode_of_distribution(staying_times) # min off
ann_output_processor.mode_of_distribution(staying_times) # min on
ann_output_processor.denormalize([hwt_min_temp], [(hwt_max_temp-hwt_min_temp)]) # hwt temp
ann_output_processor.mode_of_distribution([0]) # ambient temp
ann_output_processor.mode_of_distribution(-model.actions) # el. interaction
ann_output_processor.mode_of_distribution([0]) # th. interaction
ann_output_processor.split_result([7])

sampling_parameters = {
    'min_off_times': staying_times,
    'min_on_times' : staying_times,
    'dwell_times': staying_times,
    'temp_distribution': ([(20,60), (60,80), (80,90)], [3/20, 14/20, 3/20]),
    'infeasible_chance': 1./2,
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
        'lr_scheduler': [(torch.optim.lr_scheduler.StepLR,{'step_size':25,'gamma':0.5}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.95}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.99})]
    }


####

output_location = '{}/output/{{}}_chpp_hwt_state'.format(os.path.dirname(__file__)) 
