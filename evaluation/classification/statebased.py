from typing import Tuple, Dict

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

import torch
import torch.nn as nn

state_model_location = 'learning/statebased/output/' + '2020-08-20_18-18-40_chpp_hwt_state'
action_model_location = 'learning/statebased/output/' + '2020-08-17_11-55-57_chpp_hwt_actions'

# load neural networks (including corresponding simulation model)
from modules.utils import load_neural_model
state_meta_data, state_nn_parameters, state_neural_network = load_neural_model(state_model_location)
action_meta_data, action_nn_parameters, action_neural_network = load_neural_model(action_model_location)
simulation_model = state_meta_data['model']
sampling_parameters = state_meta_data['sampling_parameters']
simulation_model.eval()

#""""
# HOTFIX for CHPP_HWT and DFH:
#
# This block is only required if the original experiment is repeated with the CHPP_HWT or DFH models provided in the repository.
# The heat demand data loaded from crest_heat_demand.txt during the training and saved into the simulation model was erroneous. 
# This error did not affect the training process, but does influence the evaluation results, making them better as they should be.
# Newly trained models do not require this step.
#
with open('data/crest_heat_demand.txt', 'r') as file: # read data
    demand_series = np.loadtxt(file, delimiter='\t', skiprows=1) # read file, dismiss header
    demand_series = demand_series.transpose(1,0) # dim 0 identifies the series
    demand_series *= 1000 # kW -> W
simulation_model.demand.demand_series = demand_series
# HOTFIX END
#""""

#
# parameters
# 

evaluation_periods = [96,32,24,4]
time_step_count = 96
feasibility_threshold = 0.5

simulation_model.constraint_fuzziness = 0.0

process_count = 24
schedule_count_random = 25000
schedule_count_reference = 25000
schedule_count_arbitrary = 50000

# CHPP systems
from modules.simulation.integrated.holl import HoLL
if type(simulation_model) is HoLL:
    sampling_parameters['temp_distribution'] = ([(35,40), (40,60), (60,65)], [1/20, 18/20, 1/20])
else:
    sampling_parameters['temp_distribution'] = ([(55,60), (60,80), (80,85)], [1/20, 18/20, 1/20])
# BESS systems
sampling_parameters['soc_distribution'] = ([(simulation_model.constraint_fuzziness, 1-simulation_model.constraint_fuzziness)], [1])
# EVSE systems
sampling_parameters['possible_capacities'] = [17.6, 27.2, 36.8, 37.9, 52, 70, 85, 100]

logger = None
if __name__ == '__main__':
    logging_config['handlers']['file']['filename'] = '{}/{}-log.txt'.format(os.path.dirname(__file__), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging_config['formatters']['standard']['format'] = '%(message)s'
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('')

    logger.info(state_model_location)
    logger.info(action_model_location)


# create neural model
from modules.neuralnetwork.models.statebased import NeuralModel

def classify_by_interaction(model: NeuralModel, feasibility_threshold):
    # experimental
    actions = model.actions
    state, interaction, ratings = model.batch_transition(actions)
    ratings = torch.abs(torch.Tensor(actions) - interaction[:,0])
    return ratings/max(ratings), actions[ratings < max(-min(actions), max(actions)) * feasibility_threshold]

actions = simulation_model.actions
neural_model = NeuralModel(state_meta_data['dt'], actions, 
                    state_neural_network, state_meta_data['input_processor'], state_meta_data['ann_output_processor'], 
                    #classify_by_interaction,
                    action_neural_network, action_meta_data['input_processor'], action_meta_data['ann_output_processor'], 
                    feasibility_threshold=feasibility_threshold)

#
# test classification
#
from modules.simulation.simulationmodel import SimulationModel

def choose_arbitrary_action(logger, step, model: SimulationModel, filter_list=[], **kwargs) -> Tuple[int, bool, Dict]:
    return np.random.choice(model.actions), False, {}

def choose_action_randomly(logger, step, model: SimulationModel
,  filter_list=[], **kwargs) -> Tuple[int, bool, Dict]:
    feasible_actions = np.setdiff1d(model.feasible_actions, filter_list)
    if len(feasible_actions) == 0:
        logger.info('Action selection fallback for state {}'.format(np.round(model.state, 4).tolist()))
        return actions[np.argmax(model.ratings)], True, {}
    return np.random.choice(feasible_actions), False, {}

def choose_action_using_reference(  logger, step, model: SimulationModel, filter_list=[], 
                                    reference_schedule=None, reference_schedule_length=24, reference_schedule_step_length=4, 
                                    reference_intervals=None, **kwargs) -> Tuple[int, bool, Dict]:
    if reference_intervals is None:
        actions = model.actions
        reference_intervals = np.linspace(min(actions), max(actions), 5)

    if reference_schedule is None:
        reference_schedule = np.random.choice(len(reference_intervals)-1, reference_schedule_length+1).repeat(reference_schedule_step_length)
        #reference_schedule = np.roll(reference_schedule, np.random.choice(reference_schedule_step_length))
    
    next_kwargs = {
        'reference_schedule': reference_schedule,
        'reference_intervals': reference_intervals,
        'reference_schedule_length': reference_schedule_length,
        'reference_schedule_step_length': reference_schedule_step_length
    }

    feasible_actions = np.setdiff1d(model.feasible_actions, filter_list)
    if len(feasible_actions) == 0:
        logger.info('Action selection fallback for state {}'.format(np.round(model.state, 4).tolist()))
        return model.actions[np.argmax(model.ratings)], True, next_kwargs

    mask = []
    for i in range(len(reference_intervals)):
        mask = (feasible_actions >= reference_intervals[max(0,reference_schedule[step]-i)])
        mask &= (feasible_actions <= reference_intervals[min(len(reference_intervals)-1, reference_schedule[step]+1+i)])
        if sum(mask) > 0:
            break
    return np.random.choice(feasible_actions[mask]), False, next_kwargs


# Generator Code
#

def classify_load_profile(neural_model, simulation_model, choose_action, time_step_count, sampling_parameters, logger):
    """
    Generates a load schedule from a neural model and evaluates it with the simulation model
    """
    # make sure the random seeds are different in each process
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    # determine an initial state
    simulation_model.eval() # sample with eval() setting
    simulation_model.sample_state(**sampling_parameters)
    neural_model.load_state(simulation_model.state)
    simulation_model.train() # strict constraints (which the ANN should have learned)

    # do a forecast in order to predetermine the external input and the mask required to update inputs
    forecast, forecast_mask = simulation_model.forecast(time_step_count)

    # save initial states to restore them later
    result = {}

    kwargs = {}
    result['infeasible_at'] = time_step_count
    result['classified_infeasible_at'] = time_step_count
    for step in range(time_step_count):
        #result['_'] = step # DEBUG / Testing

        ann_feasible = neural_model.feasible_actions
        sim_feasible = simulation_model.feasible_actions

        # choose an action
        action_choice, fallback_action, kwargs = choose_action(logger, step, simulation_model, [], **kwargs)
        
        if not np.isin(action_choice, sim_feasible) and result['infeasible_at'] >= time_step_count:
            # infeasible action and therefore an infeasible load profile
            # an entry smaller than time_step_count means it has already been detected as infeasible
            result['infeasible_at'] = step

        if not np.isin(action_choice, ann_feasible):
            # action deemed infeasible
            # there is nothing left to do
            result['classified_infeasible_at'] = step
            break
        
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
            
    return result

if __name__ == '__main__':
    logger.info('----------------------------------------------------------------')
    logger.info('Schedule classification')
    logger.info('----------------------------------------------------------------')
    logger.info('Feasibility threshold {}'.format(feasibility_threshold))
    logger.info('Constraint fuzziness {}'.format(simulation_model.constraint_fuzziness))
    
    import json

    logger.info('Testing {} arbitrary schedules'.format(schedule_count_arbitrary))
    with mp.Pool(processes=process_count) as pool:
        results_arbitrary = pool.starmap(classify_load_profile, 
                            [(neural_model, simulation_model, choose_arbitrary_action, time_step_count, sampling_parameters, logger) for i in range(schedule_count_arbitrary)])
    
    logger.info('Testing {} random strategy schedules'.format(schedule_count_random))
    with mp.Pool(processes=process_count) as pool:
        results_random = pool.starmap(classify_load_profile, 
                            [(neural_model, simulation_model, choose_action_randomly, time_step_count, sampling_parameters, logger) for i in range(schedule_count_random)])

    logger.info('Testing {} reference strategy schedules'.format(schedule_count_reference))
    with mp.Pool(processes=process_count) as pool:
        results_reference = pool.starmap(classify_load_profile, 
                            [(neural_model, simulation_model, choose_action_using_reference, time_step_count, sampling_parameters, logger) for i in range(schedule_count_reference)])
    
    for evaluation_period in evaluation_periods:

        feasible = [1 if entry['infeasible_at'] >= evaluation_period else 0 for array in [results_arbitrary, results_random, results_reference] for entry in array]
        classified_feasible = [1 if entry['classified_infeasible_at'] >= evaluation_period else 0 for array in [results_arbitrary, results_random, results_reference] for entry in array]

        correctly_classified = sum([truth == result for truth, result in zip(feasible, classified_feasible)])

        true_positive = sum([truth == result for truth, result in zip(feasible, classified_feasible) if truth == 1])
        false_positive = sum([truth != result for truth, result in zip(feasible, classified_feasible) if result == 1])
        true_negative = sum([truth == result for truth, result in zip(feasible, classified_feasible) if truth == 0])
        false_negative = sum([truth != result for truth, result in zip(feasible, classified_feasible) if result == 0])

        logger.info('--- {} time steps ---'.format(evaluation_period))
        logger.info('{} of {} schedules classified correctly'.format(correctly_classified, len(feasible)))
        logger.info('Feasible schedule(s) \t{}'.format(sum(feasible)))
        logger.info('Infeasible schedule(s) \t{}'.format(len(feasible) - sum(feasible)))

        logger.info('True positives \t\t{}'.format(true_positive))
        logger.info('False positives \t{}'.format(false_positive))
        logger.info('True negatives \t\t{}'.format(true_negative))
        logger.info('False negatives \t{}'.format(false_negative))

    logger.info('---')
