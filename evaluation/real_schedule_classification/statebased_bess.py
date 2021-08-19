import torch.multiprocessing as mp
mp.set_start_method('spawn', True)

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

import copy
import datetime
import logging
import logging.config
from learning.logging_config import logging_config

import numpy as np
np.set_printoptions(suppress=True)

import pandas as pd

import torch
import torch.nn as nn

from modules.simulation.individual.bess import BESS
state_model_location = 'evaluation/schedule_generation/results/gc_bess/state'
action_model_location = 'evaluation/schedule_generation/results/gc_bess/actions'

# prepare data
file_names = ['data/gc_bess_1.csv', 'data/gc_bess_2.csv']
data = None
for file_name in file_names:
    with open(file_name, 'r') as file:
        # load series
        historic_data = np.loadtxt(file, delimiter=',', skiprows=1, 
            dtype = {
                'names': ('time', 'soc', 'p_el'),
                'formats': ('datetime64[s]', np.float, np.float)
            },
            converters = {
                0: lambda s: np.datetime64(s),
                1: lambda s: float(s.strip() or np.nan),
                2: lambda s: float(s.strip() or np.nan),
            }
        ) # read file, dismiss header
    if data is None:
        data = historic_data
    else:
        data = np.concatenate((data, historic_data), axis=0)

data = data.reshape(-1, 96) # get individual days
input_data = [] # stores the inputs for each day
for idx in range(data.shape[0]):
    if np.isnan(data[idx]['p_el']).any() or (((data[idx]['time'][1:] - data[idx]['time'][:-1]) / np.timedelta64(1, 's')) != 900).any():
        continue
    input_data.append((data[idx]['soc']/100, data[idx]['p_el']))

# load neural networks (including corresponding simulation model)

from modules.utils import load_neural_model
state_meta_data, state_nn_parameters, state_neural_network = load_neural_model(state_model_location)
action_meta_data, action_nn_parameters, action_neural_network = load_neural_model(action_model_location)
simulation_model = state_meta_data['model']
sampling_parameters = state_meta_data['sampling_parameters']
simulation_model.eval()

"""
#
# estimate efficiency
#
# bess 10
# 2018-04-05 to 2018-05-04: 30 days, avg 105.0 W, SOC changed from 19% to 51% => 72.72 kWh = 0.105 kW * 24h * 30 - (51%-19%)*9 kWh
# 2018-05-01 to 2018-05-30: 30 days, avg 109.5 W, SOC changed from 25% to 29% => 78.48 kWh
#
# bess 7
# 2018-01-01 to 2018-01-30: 30 days, avg 96.8 W, SOC changed from 51% to 60% => 68.89 kWh
# 2018-07-15 to 2018-08-13: 30 days, avg 83.1 W, SOC changed from 71% to 48% => 61.90 kWh
#
# => avg. 70.5 kWh per 30 days lost (due to efficiency and self discharge)
#
#
#
simulation_model = BESS(900, BESS.create_action_set(-4600, 4600, 201), 9*1000*60*60, 0.83, 1, 0.0075, True)
simulation_model.constraint_fuzziness = 0
simulation_model.eval()
"""

#
# parameters
# 
evaluation_periods = [96]
time_step_count = 96
feasibility_threshold = 0.5

simulation_model.constraint_fuzziness = 0.0

process_count = 24

logger = {}
if __name__ == '__main__':
    logging_config['handlers']['file']['filename'] = '{}/{}-log.txt'.format(os.path.dirname(__file__), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging_config['formatters']['standard']['format'] = '%(message)s'
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('')

    logger.info(state_model_location)
    logger.info(action_model_location)


# create neural model
from modules.neuralnetwork.models.statebased import NeuralModel

actions = simulation_model.actions


neural_model = NeuralModel(state_meta_data['dt'], actions, 
                    state_neural_network, state_meta_data['input_processor'], state_meta_data['ann_output_processor'], 
                    action_neural_network, action_meta_data['input_processor'], action_meta_data['ann_output_processor'], 
                    feasibility_threshold=feasibility_threshold)

#
# test classification
#
def reproduce_load_profile(neural_model, simulation_model: BESS, input_data, logger):
    """
    Tries to follow a real load profile
    """
    # make sure the random seeds are different in each process
    #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    soc, powers = input_data
    time_step_count = powers.shape[0]

    # determine the initial state
    simulation_model.eval() # sample with eval() setting

    simulation_model.state_of_charge = soc[0]
    
    neural_model.load_state(simulation_model.state)
    simulation_model.train() # strict constraints (which the ANN should have learned)

    # do a forecast in order to predetermine the external input and the mask required to update inputs
    sampling_parameters = {}
    forecast, forecast_mask = simulation_model.forecast(time_step_count, **sampling_parameters)

    # save initial states to restore them later
    result = {}

    result['infeasible_at'] = time_step_count
    result['classified_infeasible_at'] = time_step_count

    delta_soc_ann = []
    delta_soc_sim = []
    capacity = simulation_model.capacity
    
    for step, power in enumerate(powers):
        ann_feasible = neural_model.feasible_actions
        sim_feasible = simulation_model.feasible_actions

        delta_soc_ann.append(neural_model.state[0] / capacity - soc[step])
        delta_soc_sim.append(simulation_model.state[0] / capacity - soc[step])

        # identify the correct action to follow
        action_choice = actions[np.argmin(np.abs(actions - (power)))]

        if not np.isin(action_choice, sim_feasible) and result['infeasible_at'] >= time_step_count:
            # infeasible action and therefore an infeasible load profile
            # an entry smaller than time_step_count means it has already been detected as infeasible
            result['infeasible_at'] = step

        if not np.isin(action_choice, ann_feasible) and result['classified_infeasible_at'] >= time_step_count:
            # action deemed infeasible
            # an entry smaller than time_step_count means it has already been detected as infeasible
            result['classified_infeasible_at'] = step
            # keep going to see whether the simulation model can reproduce the schedule or not
        
        # while a not detected infeasibility is actually an error at this moment, 
        # the remaining load schedule could still provide further indications that it is actually infeasible
        # (proceeding like this is also required for comparability with Bremer2015)

        state, interaction = neural_model.transition(action_choice)
        simulation_model.transition(action_choice)
        
        if step + 1 < time_step_count:
            # post processing to incorporate forecasts
            neural_model.state = state * (1-forecast_mask[step+1]) + forecast_mask[step+1] * forecast[step+1]
        #else:
            # reached final step without stopping due to a detected infeasibility

    result['delta_soc'] = delta_soc_ann
    result['[delta_soc]'] = delta_soc_sim

    return result

if __name__ == '__main__':
    logger.info('----------------------------------------------------------------')
    logger.info('Schedule classification')
    logger.info('----------------------------------------------------------------')
    logger.info('Feasibility threshold {}'.format(feasibility_threshold))
    logger.info('Constraint fuzziness {}'.format(simulation_model.constraint_fuzziness))

    logger.info('Testing {} real world load profiles'.format(len(input_data)))
    with mp.Pool(processes=process_count) as pool:
        results = pool.starmap(reproduce_load_profile, 
                            [(neural_model, simulation_model, data, logger) for data in input_data])
    
    for evaluation_period in evaluation_periods:

        feasible = [1 if entry['infeasible_at'] >= evaluation_period else 0 for entry in results]
        classified_feasible = [1 if entry['classified_infeasible_at'] >= evaluation_period else 0 for entry in results]

        correctly_classified = sum([truth == result for truth, result in zip(feasible, classified_feasible)])

        true_positive = sum([truth == result for truth, result in zip(feasible, classified_feasible) if truth == 1])
        false_positive = sum([truth != result for truth, result in zip(feasible, classified_feasible) if result == 1])
        true_negative = sum([truth == result for truth, result in zip(feasible, classified_feasible) if truth == 0])
        false_negative = sum([truth != result for truth, result in zip(feasible, classified_feasible) if result == 0])

        logger.info('--- {} time steps ---'.format(evaluation_period))
        logger.info('{} of {} schedules are correctly reproduced by the simulation model'.format(sum(feasible), len(input_data)))
        logger.info('Feasible schedule(s) \t{}'.format(sum(feasible)))
        logger.info('Infeasible schedule(s) \t{}'.format(len(feasible) - sum(feasible)))

        logger.info('{} of {} schedules are classified accordingly by the surrogate model '.format(correctly_classified, len(feasible)))
        logger.info('True positives \t\t{}'.format(true_positive))
        logger.info('False positives \t{}'.format(false_positive))
        logger.info('True negatives \t\t{}'.format(true_negative))
        logger.info('False negatives \t{}'.format(false_negative))

        logger.info('Average SOC delta (ANN) {} (reference is only an estimate)'.format(np.average(np.concatenate([entry['delta_soc'] for entry in results]))))
        logger.info('Average SOC delta (Sim) {} (reference is only an estimate)'.format(np.average(np.concatenate([entry['[delta_soc]'] for entry in results]))))
    logger.info('---')