from copy import deepcopy
from typing import Tuple, List

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

state_model_location = 'learning/statebased/output/' + '2020-09-02_19-02-11_dfh_state'
action_model_location = 'learning/statebased/output/' + '2020-08-31_16-11-58_dfh_actions'

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

evaluation_periods = [96,24,4]
time_step_count = 96
feasibility_threshold = 0.5
repeat_action_choice_threshold = 1 # >= 1 => means actions are not re-drawn

simulation_model.constraint_fuzziness = 0.02

process_count = 24
testset_sample_count    = 0 #int(1e5) # 100k
generation_sample_count = int(1e4) # 10k
validate_with_milp      = True # set this to True only for aggregated systems, individual DERs do not need this step
opt_time_limit = 20

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

logger = {}
if __name__ == '__main__':
    logging_config['handlers']['file']['filename'] = '{}/{}-log.txt'.format(os.path.dirname(__file__), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging_config['formatters']['standard']['format'] = '%(message)s'
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('')

    logger.info(state_model_location)
    logger.info(action_model_location)

    if validate_with_milp == True and os.environ.get('GUROBI_HOME') == None:
        logger.info('$GUROBI_HOME not specified')
        sys.exit()

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
# test accuracry
#
def test_sample(neural_model, simulation_model, sampling_parameters, logger):
    actions = simulation_model.actions
    simulation_model.sample_state(**sampling_parameters)
    neural_model.load_state(simulation_model.state)
    
    feasible_actions = simulation_model.feasible_actions
    feasible_actions_ann = neural_model.feasible_actions
    infeasible_actions = np.setdiff1d(actions, feasible_actions)
    
    action = np.random.choice(actions)
    
    #    +   -
    #^+  TP  FP
    #^-  FN  TN
    isin = np.isin(feasible_actions_ann, feasible_actions)
    true_positive = sum(isin)
    false_positive = len(feasible_actions_ann) - true_positive
    false_negative = len(feasible_actions) - true_positive
    true_negative = len(infeasible_actions) - false_positive
    classifier_result = [true_positive, false_positive, true_negative, false_negative]

    #old_state = np.copy(simulation_model.state)
    new_state_nn, interaction_nn = neural_model.transition(action)
    new_state_sim, interaction_sim = simulation_model.transition(action)

    errors_state = (new_state_sim - new_state_nn)
    errors_interaction = (interaction_sim - interaction_nn)

    return classifier_result, errors_state, errors_interaction

if __name__ == '__main__' and testset_sample_count > 0:
    logger.info('----------------------------------------------------------------')
    logger.info('Dataset evaluation')
    logger.info('----------------------------------------------------------------')
    logger.info('Generating {} samples'.format(testset_sample_count))

    simulation_model.train()

    with mp.Pool(processes=process_count) as pool:
        results = pool.starmap(test_sample, 
                            [(neural_model, simulation_model, sampling_parameters, logger) for i in range(testset_sample_count)])

    classifier_results = np.array([e[0] for e in results])
    errors_state = np.array([e[1] for e in results])
    errors_interaction = np.array([e[2] for e in results])

    logger.info('classifier')
    tp, fp, tn, fn = np.sum(classifier_results, axis=0) / classifier_results.shape[0] / len(simulation_model.actions)
    logger.info('[true positive] ={}'.format(tp))
    logger.info('[false positive]={}'.format(fp))
    logger.info('[true negative] ={}'.format(tn))
    logger.info('[false negative]={}'.format(fn))
    logger.info('state')
    for step in range(errors_state.shape[1]):
        logger.info('[{}]={}'.format(step,np.quantile(errors_state[:,step], [0,0.25,0.5,0.75,1])))
    logger.info('interaction')
    for step in range(errors_interaction.shape[1]):
        logger.info('[{}]={}'.format(step,np.quantile(errors_interaction[:,step], [0,0.25,0.5,0.75,1])))
    logger.info('')


#
# test generation
#
def compare_log(log, idx):
    return np.array([np.array(e['state'])[idx] - np.array(e['[state]'])[idx] for e in log])

# Custom code
#
def choose_action_randomly(logger, step, model: NeuralModel,  filter_list=[], **kwargs) -> Tuple[int, bool, dict]:
    feasible_actions = np.setdiff1d(model.feasible_actions, filter_list)
    if len(feasible_actions) == 0:
        logger.info('Action selection fallback for state {}'.format(np.round(model.state, 4).tolist()))
        return actions[np.argmax(model.ratings)], True, {}
    return np.random.choice(feasible_actions), False, {}


def choose_action_using_reference(  logger, step, model, filter_list=[], 
                                    reference_schedule=None, reference_schedule_length=24, reference_schedule_step_length=4, 
                                    reference_intervals=None, **kwargs) -> Tuple[int, bool, dict]:
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

#
# Generator Configuration
#

choose_action = choose_action_using_reference
#choose_action = choose_action_randomly

action_evaluation_tolerance = (max(actions) - min(actions)) / (len(actions)-1) / 2

# Generator Code
#

def generate_load_schedule(neural_model, simulation_model, choose_action, time_step_count, sampling_parameters, logger):
    """
    Generates a load schedule from a neural model and evaluates it with the simulation model
    """
    # make sure the random seeds are different in each process
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    # create an array for storing the results
    generator_log = []

    # determine an initial state
    simulation_model.eval() # sample with eval() setting
    simulation_model.sample_state(**sampling_parameters)
    neural_model.load_state(simulation_model.state)

    # do a forecast in order to predetermine the external input and the mask required to update inputs
    forecast, forecast_mask = simulation_model.forecast(time_step_count)

    # save initial states to restore them later
    result = {}
    #simulation_model.constraint_fuzziness = 0.0
    result['initial_state'] = np.copy(simulation_model.state)
    #simulation_model.constraint_fuzziness = 0.02
    result['initial_hidden_state'] = copy.deepcopy(simulation_model.hidden_state)

    infeasibility_step = -1

    debug_model = copy.deepcopy(simulation_model)
    debug_model.constraint_fuzziness = 0
    _ = debug_model.state
    _[1] = 0.0
    _[2] = 1
    debug_model.state = _

    action_range = np.abs(max(simulation_model.actions)-min(simulation_model.actions))
    kwargs = {}
    for step in range(time_step_count):
        # holds the data of a single iteration
        iteration_data = {}

        filter_list = []

        feasible_actions = neural_model.feasible_actions
        neural_model.push_state() # store the current state
        simulation_model.train() # strict constraints

        sim_feasible = simulation_model.feasible_actions

        # do the transition
        iteration_data['state'] = np.round(neural_model.state, 4).tolist()
        iteration_data['[state]'] = simulation_model.state.tolist()
        iteration_data['feasible'] = neural_model.feasible_actions.tolist()
        iteration_data['[feasible]'] = sim_feasible.tolist()

        iteration_data['_ds'] = debug_model.state.tolist()
        iteration_data['_df'] = debug_model.feasible_actions.tolist()
        while True:
            # choose an action
            action_choice, fallback_action, kwargs = choose_action(logger, step, neural_model, filter_list, **kwargs)
            iteration_data['action'] = action_choice

            # only switch to evaluation mode (i.e., relaxed constraints) if necessary 
            if not np.isin(action_choice, sim_feasible) and simulation_model.constraint_fuzziness > 0:
                simulation_model.eval() # relaxed constraints

                # temporary fix
                simulation_model._feasible_actions = None

                sim_feasible = simulation_model.feasible_actions
                iteration_data['[[feasible]]'] = sim_feasible.tolist()
            else:
                iteration_data['[[feasible]]'] = []

            state, interaction = neural_model.transition(action_choice)
            iteration_data['interaction'] = np.round(interaction, 4).tolist()

            #fallback_action = True
            if not fallback_action and \
                (np.abs(action_choice + interaction[0]) > repeat_action_choice_threshold * action_range and len(feasible_actions) - len(filter_list) > 0):
                # more than X % deviation, and there are alternatives
                filter_list.append(action_choice)
                neural_model.pop_state() # undo the transition
                continue

            sim_state, sim_interaction = simulation_model.transition(action_choice)
            iteration_data['[interaction]'] = sim_interaction.tolist()

            debug_state, debug_interaction = debug_model.transition(action_choice)
            
            # post processing to incorporate forecasts
            if step + 1 < time_step_count:
                neural_model.state = state * (1-forecast_mask[step+1]) + forecast_mask[step+1] * forecast[step+1]

            # was the action feasible?  
            iteration_data['[valid]'] = np.isin(action_choice, sim_feasible).tolist()
            if infeasibility_step < 0 and iteration_data['[valid]'] == False:
                infeasibility_step = step
            
            iteration_data['delta_load'] = (sim_interaction[0] + action_choice).tolist()
            iteration_data['delta_interaction'] = iteration_data['interaction'][0] - iteration_data['[interaction]'][0]

            iteration_data['_dl'] = (debug_interaction[0] + action_choice).tolist()
            
            """"
            #DEBUG
            if iteration_data['[valid]'] == False and step <= 5:
            #if iteration_data['delta_load'] < -20000:
                dummy = '_'
            """

            break
        generator_log.append(iteration_data)

    # compile result    
    result['schedule'] = [e['action'] for e in generator_log]
    result['valid'] = np.all([e['[valid]'] for e in generator_log])
    result['infeasibility_step'] = infeasibility_step
    result['delta_load'] = np.array([e['delta_load'] for e in generator_log])
    result['delta_interaction'] = np.quantile([e['delta_interaction'] for e in generator_log], [0,0.25,0.5,0.75,1])
    
    result['_dl'] = np.array([e['_dl'] for e in generator_log])
    """
    # DEBUG
    #result['l'] = np.array(generator_log)

    if result['valid'] == False:
        validity = np.array([_['[valid]'] for _ in generator_log])
        invalids = np.arange(validity.shape[0])[validity == False]
        dummy = '_'
 
        if infeasibility_step <= 1:
            dummy = '_'
    # /DEBUG
    #"""
    return result

if __name__ == '__main__' and generation_sample_count > 0:
    logger.info('----------------------------------------------------------------')
    logger.info('Schedule generation')
    logger.info('----------------------------------------------------------------')
    logger.info('Using {}() to choose actions'.format(choose_action.__name__))
    logger.info('Constraint fuzziness {}'.format(simulation_model.constraint_fuzziness))
    logger.info('Generating {} load schedules'.format(generation_sample_count))
    simulation_model.eval()

    import json

    with mp.Pool(processes=process_count) as pool:
        results = pool.starmap(generate_load_schedule, 
                            [(neural_model, simulation_model, choose_action, time_step_count, sampling_parameters, logger) for i in range(generation_sample_count)])
    
    
    infeasible_results = [result for result in results if result['valid'] == False]
    feasible_count = sum([e['valid'] for e in results])

    logger.info('{} of {} schedules feasible'.format(feasible_count, len(results)))

    logger.info('')
    logger.info('Interaction: min delta={}, max delta={}'.format(min([e['delta_interaction'][0] for e in results]), max([e['delta_interaction'][4] for e in results])))
    logger.info('')

    # (this does not depend on the evaluation period, as it is exactly determined how actions are achieved)
    logger.info('Feasible schedules by step')
    feasibility_by_step = np.array([generation_sample_count] * (time_step_count + 1))
    for result in results:
        if result['infeasibility_step'] >= 0:
            feasibility_by_step[result['infeasibility_step']+1:] -= 1
    logger.info(json.dumps(feasibility_by_step.tolist()))
    logger.info('')

    evaluation_periods = sorted(evaluation_periods, reverse=True) # begin with longest period, to reduce amount required computations
    if len(infeasible_results) > 0:
        for evaluation_period in evaluation_periods:
            logger.info('---- {} time steps -- simulation ----'.format(evaluation_period))
            filtered_infeasible_results = [e for e in infeasible_results if e['infeasibility_step'] < evaluation_period]
            logger.info('{} of {} feasible'.format(len(results) - len(filtered_infeasible_results), len(results)))
            if len(filtered_infeasible_results) == 0:
                continue
            all_errors = np.concatenate([e['delta_load'][:evaluation_period] for e in filtered_infeasible_results])
            quantiles = [0,0.01,0.025,0.05,0.25,0.5,0.75,0.95,0.975,0.99,1]
            logger.info('Number of errors (quartiles): {}'.format(np.quantile([sum(np.logical_not(np.isclose(e['delta_load'][:evaluation_period], 0))) 
                                                                        for e in filtered_infeasible_results], [0,0.25,0.5,0.75,1]).tolist()))
            logger.info('MAE of infeasibles: {}'.format(np.average(np.abs(all_errors))))
            logger.info('MSE of infeasibles: {}'.format(np.average(all_errors**2)))
            logger.info('{}-Quantiles'.format(quantiles))
            logger.info('All errors: {}'.format(np.quantile(all_errors, quantiles).tolist()))
            logger.info('Only errors: {}'.format(np.quantile(all_errors[np.logical_not(np.isclose(all_errors,0))], quantiles).tolist()))
            logger.info('')

    ##
    # MILP evaluation

    if validate_with_milp:
        logger.info('----------------------------------------------------------------')
        logger.info('Validation of {} "infeasible" [sim.] schedules with MILP (timelimit={})'.format(len(infeasible_results), opt_time_limit))
        logger.info('----------------------------------------------------------------')

        from modules.optimization.targetdeviation import TargetDeviationMILP

        for evaluation_period in evaluation_periods:
            logger.info('--------------------------------')
            logger.info('---- {} time steps ----'.format(evaluation_period))

            milp_opt_values = []
            milp_load_schedules = []
            milp_delta_load = []

            # filter feasible schedules, only validate infeasibles
            if evaluation_period < time_step_count:
                infeasible_results = [e for e in infeasible_results if e['infeasibility_step'] < evaluation_period]
            # DEBUG test all
            #infeasible_results = results 
            # /DEBUG
            
            filtered_infeasible_results = [] # stores those results that can not be reproduced
            for idx, result in enumerate(infeasible_results):
                # DEBUG only test feasibles
                #if sum(np.logical_not(np.isclose(result['delta_load'][:evaluation_period], 0))) != 0: continue
                # /DEBUG
                try:
                    print('Solving MILP {} of {}'.format(idx+1, len(infeasible_results)), end='\r')

                    milp = TargetDeviationMILP(simulation_model.dt, evaluation_period)
                    simulation_model.load_state(result['initial_state'], result['initial_hidden_state'])
                    milp.add_constraints(simulation_model)
                    milp.create_objective(result['schedule'][:evaluation_period])
                    solution = milp.solve(timelimit=opt_time_limit, verbose=False)

                    objective_value = milp.model.obj.expr()
                    milp_opt_values.append(objective_value)
                    if not np.isclose(objective_value, 0):
                        # probably truly infeasible
                        filtered_infeasible_results.append(result)
                    # DEBUG check if conflicting simulation (either abort or error)
                    #elif result['infeasibility_step'] < evaluation_period:
                    #    logger.info('Simulation / MILP conflict')
                    # /DEBUG

                    schedule = np.array([milp.model.P_total[i].value for i in milp.model.t])
                    milp_load_schedules.append(schedule)
                    milp_delta_load.append(result['schedule'][:evaluation_period] - schedule)
                except (KeyboardInterrupt, SystemExit):
                    logging.warning('Premature exit')
                    sys.exit(0)
                except:
                    logger.exception('Exception: {}'.format(sys.exc_info()[0]))
                    filtered_infeasible_results.append(result)

            reproducable_count = sum(np.isclose(milp_opt_values, 0))
            logger.info('{} of {} schedules exactly reproducable'.format(reproducable_count, len(infeasible_results)))
            if reproducable_count < len(infeasible_results):
                # simulation, but this time only those schedules not reproducible
                logger.info('---- {} time steps -- filtered simulation results ----'.format(evaluation_period))
                all_errors = np.concatenate([e['delta_load'][:evaluation_period] for e in filtered_infeasible_results])
                quantiles = [0,0.01,0.025,0.05,0.25,0.5,0.75,0.95,0.975,0.99,1]
                logger.info('Number of errors (quartiles): {}'.format(np.quantile([sum(np.logical_not(np.isclose(e['delta_load'][:evaluation_period], 0))) 
                                                                            for e in filtered_infeasible_results], [0,0.25,0.5,0.75,1]).tolist()))
                logger.info('MAE of infeasibles: {}'.format(np.average(np.abs(all_errors))))
                logger.info('MSE of infeasibles: {}'.format(np.average(all_errors**2)))
                logger.info('{}-Quantiles'.format(quantiles))
                logger.info('All errors: {}'.format(np.quantile(all_errors, quantiles).tolist()))
                logger.info('Only errors: {}'.format(np.quantile(all_errors[np.logical_not(np.isclose(all_errors,0))], quantiles).tolist()))
                logger.info('')

                logger.info('---- {} time steps -- MILP results ----'.format(evaluation_period))
                # MILP results (which may actually be better, since optimization time limit is very short)
                filtered_milp_delta_load = [delta_load for opt_value, delta_load in zip(milp_opt_values, milp_delta_load) if not np.isclose(opt_value, 0)]
                all_errors_milp = np.concatenate(filtered_milp_delta_load)
                logger.info('Number of errors (quartiles): {}'.format(np.quantile([sum(np.logical_not(np.isclose(e,0))) 
                                                                            for e in filtered_milp_delta_load], [0,0.25,0.5,0.75,1]).tolist()))
                logger.info('MAE of infeasibles: {}'.format(np.average(np.abs(all_errors_milp))))
                logger.info('MSE of infeasibles: {}'.format(np.average(all_errors_milp**2)))
                logger.info('{}-Quantiles'.format(quantiles))
                logger.info('All errors: {}'.format(np.quantile(all_errors_milp, quantiles).tolist()))
                logger.info('Only errors: {}'.format(np.quantile(all_errors_milp[np.logical_not(np.isclose(all_errors_milp,0))], quantiles).tolist()))
                logger.info('')
                
            infeasible_results = filtered_infeasible_results

            # further analysis
            logger.info('Feasible schedules by step')
            feasibility_by_step = np.array([generation_sample_count] * (evaluation_period + 1))
            for result in infeasible_results:
                feasibility_by_step[result['infeasibility_step']+1:] -= 1
            logger.info(json.dumps(feasibility_by_step.tolist()))
            logger.info('')

    logger.info('---')
