import os
import numpy as np

from modules.simulation.integrated.holl import HoLL

####
# general hyperparameters
####
# simulation
time_step = 15 * 60 # seconds

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

model = HoLL(time_step, HoLL.action_set_100w, 0.01, correct_infeasible=allow_infeasible_actions)

hwt_min_temp = model.hwt_gcb.hwt.soft_min_temp
hwt_max_temp = model.hwt_gcb.hwt.soft_max_temp

####
# sample generation parameters and ANN loss
####
import torch
import torch.nn as nn

# determine torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from modules.neuralnetwork.samplegenerators.statebased import BatchProcessor

"""
State:
- EVSE
    - capacity
    - SOC min
    - SOC max
    - SOC
    - remaining standing time
- BESS
    - SOC
    - SOC min
    - SOC max
- CHPP
    - mode
    - dwell time
    - min off time
    - min on time
- HWT_GCB
    - GCB
        - mode: index of mode (0 to n)
    - HWT
        - temperature ``float`` C
        - ambient_temperature ``float`` C   
- Demand (heat)
    - demand (forecast)
"""
possible_bev_standing_times = [_ * time_step for _ in range(97)]
chpp_staying_times = [i * 900 for i in range(1,4*2+1)]

input_processor = BatchProcessor()
# EVSE
input_processor.normalize([0], [100 * 1000 * 60 * 60]) # capacity
input_processor.none(3) # soc_min, soc_max, soc
input_processor.one_hot(possible_bev_standing_times) # remaining standing time
# BESS
input_processor.normalize([0], [model.bess.capacity]) # soc
input_processor.none(2) # soc_min, soc_max
# CHPP
input_processor.one_hot([i for i in range(len(model.chpp.state_matrix[0]))]) # mode
input_processor.one_hot(chpp_staying_times) # staying time
input_processor.one_hot(chpp_staying_times) # min off
input_processor.one_hot(chpp_staying_times) # min on
# HWT_GCB
input_processor.one_hot([i for i in range(len(model.hwt_gcb.state_matrix[0]))]) # mode
input_processor.normalize([hwt_min_temp, model.hwt_gcb.hwt.ambient_temperature], [(hwt_max_temp-hwt_min_temp), 1.]) # hwt temp, ambient temp
# demand
input_processor.normalize([0], [100000])
# action
input_processor.one_hot(model.actions)



output_processor = BatchProcessor()
# EVSE
output_processor.normalize([0], [100 * 1000 * 60 * 60]) # capacity
output_processor.none(3) # soc_min, soc_max, soc
output_processor.discretize_index(possible_bev_standing_times) # remaining standing time
# BESS
output_processor.normalize([0], [model.bess.capacity]) # soc
output_processor.none(2) # soc_min, soc_max
# CHPP
output_processor.discretize_index([i for i in range(len(model.chpp.state_matrix[0]))]) # mode
output_processor.discretize_index(chpp_staying_times) # staying time
output_processor.discretize_index(chpp_staying_times) # min off
output_processor.discretize_index(chpp_staying_times) # min on
# HWT_GCB
output_processor.discretize_index([i for i in range(len(model.hwt_gcb.state_matrix[0]))]) # mode
output_processor.normalize([hwt_min_temp], [(hwt_max_temp-hwt_min_temp)]) # hwt temp
output_processor.discretize_index([0]) # ambient temp
# Demand
output_processor.discretize_index([0])
# interaction
output_processor.discretize_index(-model.actions) # el. interaction [minus, since the interaction is the negative action (for a feasible action)]
output_processor.discretize_index([0]) # th. interaction


losses = []
# EVSE
losses.append((nn.MSELoss().to(device), 1)) # capacity
losses.append((nn.MSELoss().to(device), 2)) # soc_min, soc_max
losses.append((nn.MSELoss().to(device), 1)) # soc
losses.append((nn.CrossEntropyLoss().to(device), 97)) # remaining standing time
# BESS
losses.append((nn.MSELoss().to(device), 1)) # soc
losses.append((nn.MSELoss().to(device), 2)) # soc_min, soc_max
# CHPP
losses.append((nn.CrossEntropyLoss().to(device), len(model.chpp.state_matrix[0]))) # mode
losses.append((nn.CrossEntropyLoss().to(device), len(chpp_staying_times))) # staying time
losses.append((nn.CrossEntropyLoss().to(device), len(chpp_staying_times))) # min off
losses.append((nn.CrossEntropyLoss().to(device), len(chpp_staying_times))) # min on
# HWT_GCB
losses.append((nn.CrossEntropyLoss().to(device), len(model.hwt_gcb.state_matrix[0]))) # mode
losses.append((nn.MSELoss().to(device), 1)) # hwt_temp
losses.append((nn.CrossEntropyLoss().to(device), 1)) # ambient temp
# Demand
losses.append((nn.CrossEntropyLoss().to(device), 1))
# interaction
losses.append((nn.CrossEntropyLoss().to(device), len(model.actions))) # el interaction
losses.append((nn.CrossEntropyLoss().to(device), 1)) # th interaction


loss_weights = [1e1,1e1,1e6,1e1, # evse (cap, min&max, SOC, time)
                1e6,1e1, #bess
                1e3,1e3,1e1,1e1, #chpp
                1e3,1e5,1e0, # hwt_gcb
                1e0, # demand
                1e0,1e0]


ann_output_processor = BatchProcessor()
# EVSE
ann_output_processor.denormalize([0], [100 * 1000 * 60 * 60]) # capacity
ann_output_processor.none(3).clip(0,1,3) # soc_min, soc_max, soc
ann_output_processor.mode_of_distribution(possible_bev_standing_times) # remaining standing time
# BESS
ann_output_processor.denormalize([0], [model.bess.capacity]).clip(0, model.bess.capacity) # soc
ann_output_processor.none(2) # soc_min, soc_max
# CHPP
ann_output_processor.mode_of_distribution([i for i in range(len(model.chpp.state_matrix[0]))]) # mode
ann_output_processor.mode_of_distribution(chpp_staying_times) # staying time
ann_output_processor.mode_of_distribution(chpp_staying_times) # min off
ann_output_processor.mode_of_distribution(chpp_staying_times) # min on
# HWT_GCB
ann_output_processor.mode_of_distribution([i for i in range(len(model.hwt_gcb.state_matrix[0]))]) # mode
ann_output_processor.denormalize([hwt_min_temp], [(hwt_max_temp-hwt_min_temp)]) # hwt temp
ann_output_processor.mode_of_distribution([0]) # ambient temp
# Demand
ann_output_processor.mode_of_distribution([0])
# interaction
ann_output_processor.mode_of_distribution(-model.actions) # el. interaction
ann_output_processor.mode_of_distribution([0]) # th. interaction
ann_output_processor.split_result([16])

sampling_parameters = {
    'soc_distribution': ([(0,0.25), (0.25,0.75), (0.75,1)], [3./8, 2./8, 3./8]),
    'min_off_times': chpp_staying_times,
    'min_on_times' : chpp_staying_times,
    'dwell_times': chpp_staying_times,
    'temp_distribution': ([(20,40), (40,60), (60,90)], [3/20, 14/20, 3/20]),
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

min_layers = 6
max_layers = 12
meta_search_parameter_space = {
        'batch_size': [batch_size],
        'loss': [MixedLoss(losses, loss_weights, device=device)],
        'regularization': [
            L1RegularizationLoss(2e-6, device),
            L1RegularizationLoss(2e-7, device),
            L1RegularizationLoss(2e-8, device),
            ],
        'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
        'epoch_count': [1000],
        'batch_count': [training_cache_size / batch_size],
        'max_grad_norm': [1e6],

        'input_width': [sample_input_size],
        'output_width': [output_width],
        'output_activation': [None],

        'hidden_layer_count': np.arange(min(max_layers-1,min_layers),max_layers), # 'min(.,.)' is a failsafe
        'width': [2**i for i in range(8,11)],
        'width_interpolation_steps_input': [0,1,2],
        'width_interpolation_steps_output': [0,3,4,5],
        'betas': ([0.5, 1, 2], max_layers),
        'batch_norms': ([0]*14+[1], max_layers),
        'dropout': ([0], max_layers), #([0]*28+[0.2]+[0.5], max_layers),
        'skips': ([0]*19+[1], (max_layers, max_layers)),

        'early_stopping_callback': [EarlyStoppingCallback({
        }, 100)],
        #'lr_scheduler': [(torch.optim.lr_scheduler.StepLR,{'step_size':25,'gamma':0.75}), (torch.optim.lr_scheduler.StepLR,{'step_size':1,'gamma':0.99})]
        'lr_scheduler': [(torch.optim.lr_scheduler.StepLR,{'step_size':25,'gamma':0.5}), (torch.optim.lr_scheduler.StepLR,{'step_size':25,'gamma':0.75})]
    }


####

output_location = '{}/output/{{}}_holl_state'.format(os.path.dirname(__file__)) 
