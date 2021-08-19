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

from modules.simulation.integrated.chpp_hwt import CHPP_HWT
state_model_location = 'learning/statebased/output/' + '2020-08-20_18-18-40_chpp_hwt_state'
action_model_location = 'learning/statebased/output/' + '2020-08-17_11-55-57_chpp_hwt_actions'

# load historic data
data = pd.read_csv('data/eshl_data.csv', sep=',')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp')

input_data = []
for group in data.groupby(data.index.date):
    df = group[1]
    input_data.append((np.array(df['temp_sensor1'].tolist()), -np.array((1000*df['chpp']).tolist()), np.array((1000*df['demand']).tolist())))

#"""
# load neural networks (including corresponding simulation model)
from modules.utils import load_neural_model
state_meta_data, state_nn_parameters, state_neural_network = load_neural_model(state_model_location)
action_meta_data, action_nn_parameters, action_neural_network = load_neural_model(action_model_location)
simulation_model = state_meta_data['model']
sampling_parameters = state_meta_data['sampling_parameters']
simulation_model.eval()
"""

from modules.simulation.individual.chpp import CHPP
from modules.simulation.individual.hwt import HWT
from modules.simulation.individual.demand import Demand
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

chpp = CHPP(900, CHPP.create_action_set(-5500, 0, 101), state_matrix, correct_infeasible=allow_infeasible_actions)
hwt = HWT(900, hwt_min_temp, hwt_max_temp, hwt_volume, 1, 1, relative_loss)
demand = Demand(900, demand_series)
simulation_model= CHPP_HWT(chpp, hwt, demand, 0.01) # train a safety margin of 1°C into the model (instead of adding a state variable)
#"""

#
# parameters
# 

evaluation_periods = [96]
time_step_count = 96
feasibility_threshold = 0.5

#simulation_model.constraint_fuzziness = 0.0 # do NOT change this, the model has been trained with a buffer

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
#"""
neural_model = NeuralModel(state_meta_data['dt'], actions, 
                    state_neural_network, state_meta_data['input_processor'], state_meta_data['ann_output_processor'], 
                    action_neural_network, action_meta_data['input_processor'], action_meta_data['ann_output_processor'], 
                    feasibility_threshold=feasibility_threshold)
"""
neural_model = simulation_model
#"""

#
# test classification
#
def reproduce_load_profile(neural_model, simulation_model: CHPP_HWT, input_data, logger):
    """
    Tries to follow a real load profile
    """
    # make sure the random seeds are different in each process
    #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    temperature, powers, heat_demand = input_data
    time_step_count = powers.shape[0]

    # save initial states to restore them later
    result = {}
    result['temp_offset'] = max(-min(temperature) + 60, 0)
    temperature += result['temp_offset']  

    # determine the initial state
    simulation_model.eval() # sample with eval() setting

    simulation_model.chpp.mode = 0 if powers[0] > -3000 else 1
    simulation_model.chpp.min_off_time = 900
    simulation_model.chpp.min_on_time = 900
    simulation_model.chpp.dwell_time = 900

    simulation_model.hwt.temperature = temperature[0]

    simulation_model.demand.demand = heat_demand[0]
    simulation_model.demand.forecast_series = heat_demand[1:].reshape(-1,1)
    
    neural_model.load_state(simulation_model.state)
    simulation_model.train() # strict constraints (which the ANN should have learned)

    # do a forecast in order to predetermine the external input and the mask required to update inputs
    sampling_parameters = {}
    forecast, forecast_mask = simulation_model.forecast(time_step_count, **sampling_parameters)

    result['infeasible_at'] = time_step_count
    result['classified_infeasible_at'] = time_step_count
    
    delta_temp_ann = []
    delta_temp_sim = []
    for step, power in enumerate(powers):
        ann_feasible = neural_model.feasible_actions
        sim_feasible = simulation_model.feasible_actions

        delta_temp_ann.append(neural_model.state[-2] - temperature[step])
        delta_temp_sim.append(simulation_model.state[-2] - temperature[step])

        # identify the correct action to follow
        if power > -3000: # off
            action_choice = simulation_model.chpp.state_matrix[simulation_model.chpp.mode][0][0]
        else: # on
            action_choice = simulation_model.chpp.state_matrix[simulation_model.chpp.mode][1][0]
    
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

    result['delta_temp'] = delta_temp_ann
    result['[delta_temp]'] = delta_temp_sim

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
        logger.info('Added a temperature offset {} times'.format(sum([1 if entry['temp_offset'] > 0 else 0 for entry in results])))
        logger.info('Feasible schedule(s) \t{}, added offset {} times'.format(sum(feasible), 
            sum([1 if entry['temp_offset'] > 0 and entry['infeasible_at'] >= evaluation_period else 0 for entry in results])))
        logger.info('Infeasible schedule(s) \t{}, added offset {} times'.format(len(feasible) - sum(feasible),
            sum([1 if entry['temp_offset'] > 0 and entry['infeasible_at'] < evaluation_period else 0 for entry in results])))
        
        logger.info('{} of {} schedules are classified accordingly by the surrogate model '.format(correctly_classified, len(feasible)))
        logger.info('True positives \t\t{}'.format(true_positive))
        logger.info('False positives \t{}'.format(false_positive))
        logger.info('True negatives \t\t{}'.format(true_negative))
        logger.info('False negatives \t{}'.format(false_negative))

        logger.info('Average temp. delta (ANN) {} (reference is only an estimate)'.format(np.average(np.concatenate([entry['delta_temp'] for entry in results]))))
        logger.info('Average temp. delta (Sim) {} (reference is only an estimate)'.format(np.average(np.concatenate([entry['[delta_temp]'] for entry in results]))))
    logger.info('---')
