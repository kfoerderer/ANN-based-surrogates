################
# Parameters
# 
evaluation_periods = [8,32]
feasibility_threshold = 0.5

constraint_fuzziness = 0.02

# how many instances of the same device/model should be considered?
# this is a proof of concept and only a first evaluation, future experiments should combine different models
ensemble_size = 10

max_iterations = 100 # max restarts from root
exploration_steps = 5 # how man paths should be explored starting from the current node

process_count = 8
target_schedule_count = 100#0 # how often should the experiment be executed?

configuration = 'bess'
state_model_location = 'evaluation/schedule_generation/results/' + configuration + '/state'
action_model_location = 'evaluation/schedule_generation/results/' + configuration + '/actions'

#
# /Parameters
################

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

from anytree import AnyNode

import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn as nn

from modules.utils import load_neural_model
from modules.neuralnetwork.models.statebased import BatchedNeuralModel

from modules.simulation.integrated.holl import HoLL

def create_target_schedule(simulation_model, ensemble_size, time_step_count, hidden_state, sampling_parameters):
    # make sure the random seeds are different in each process
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    simulation_model.eval()
    initial_states = [copy.deepcopy(simulation_model.sample_state()) for i in range(ensemble_size)]

    load_schedule = [0] * time_step_count

    # simply drawing random actions would yield uncorrelated loads
    # in reality, it is likely that the systems' behaviour would show some correlation
    # for instance, due to the photovoltaic production
    # --> increase likelihood of similar actions by using a reference schedule
    actions = simulation_model.actions
    reference_schedule_step_length = 4
    reference_intervals = np.linspace(min(actions), max(actions), 9)#5)
    reference_schedule = np.random.choice(len(reference_intervals)-1, int(time_step_count / reference_schedule_step_length)+1).repeat(reference_schedule_step_length)

    forecasts = []
    masks = []

    # create $ensemble_size schedules and aggregate them to create a feasible target
    for initial_state in initial_states:
        simulation_model.sample_state(**sampling_parameters)
        simulation_model.load_state(initial_state, hidden_state) # load initial state
        forecast, forecast_mask = simulation_model.forecast(time_step_count)
        forecasts.append(forecast)
        masks.append(forecast_mask)
        
        simulation_model.train()
        for step in range(time_step_count):
            feasible_actions = simulation_model.feasible_actions
            
            mask = []
            # select action based on reference intervals
            for i in range(len(reference_intervals)):
                # in iteration 1 (i=0) the reference interval given by the reference schedule is checked
                # in each subsequent iteration the lower and upper bounds are shifted outwards by $i interval lengths
                mask = (feasible_actions >= reference_intervals[max(0,reference_schedule[step]-i)])
                mask &= (feasible_actions <= reference_intervals[min(len(reference_intervals)-1, reference_schedule[step]+1+i)])
                if sum(mask) > 0:
                    break
            action = np.random.choice(feasible_actions[mask])

            load_schedule[step] += action
            simulation_model.transition(action)
        simulation_model.eval()
    
    return (np.array(initial_states), np.array(forecasts), np.array(masks), load_schedule)

def monte_carlo_tree_search(batched_neural_model, initial_states, forecasts, masks, target_schedule, max_iterations, exploration_steps):
    # make sure the random seeds are different in each process
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    actions = batched_neural_model.actions
    preferred_action_radius = (np.max(actions) - np.min(actions))*0.1 / 2

    batched_neural_model.state = np.array(initial_states)
    ensemble_size = len(initial_states)
    time_step_count = len(target_schedule)

    root = AnyNode(
        t=0,
        prior_reward = 0,
        current_reward = 0,
        subsequent_reward_sum = 0,
        explored_number = 0,
        action = None,
        state = initial_states)

    leaves = []
    best_reward = float('-inf')
    best_leaf = None
    #random_share = 0.75
    for i in range(max_iterations):
        # selection, begin at root
        current_node = root
        for time_step in range(time_step_count):
            for exploration_step in range(exploration_steps):
                #print('starting exploration at step {}'.format(current_node.t))
                #current_node = best_node
                # explore via simulation
                simulation_node = current_node
                batched_neural_model.load_state(simulation_node.state)

                # simulation: search the tree until a leaf is reached
                for simulation_time_step in range(simulation_node.t, time_step_count):

                    feasible_actions = batched_neural_model.feasible_actions

                    # select an action:
                    # computing all possible combinations is impossible for larger ensembles
                    # it is therefore generally impossible to test or rate all actions!

                    # make random selection, but switch up the order of the devices
                    # get a random order
                    system_order = np.random.choice(range(ensemble_size), ensemble_size, replace=False)

                    selected_actions = [0] * ensemble_size
                    delta_target = target_schedule[simulation_time_step]

                    #random_action_number = int(ensemble_size * random_share)
                    for j, system_id in enumerate(system_order):#[:random_action_number]:
                        _feasible_actions = feasible_actions[system_id]
                        len_feasibles = len(_feasible_actions)
                        
                        if len_feasibles > 1:
                            # build a weight vector 
                            weights = np.ones(len(_feasible_actions))
                            # myopic perspective: 'good' choices should be more likely than 'bad' choices

                            individual_target = delta_target / (ensemble_size-j)
                            idxs = (_feasible_actions > individual_target - preferred_action_radius) * (_feasible_actions < individual_target + preferred_action_radius)
                            len_idxs = np.sum(idxs)
                            if len_idxs > 0 and len_idxs < len_feasibles:
                                weights[idxs] = ((len_feasibles - len_idxs) / 0.75 - (len_feasibles - len_idxs))/len_idxs

                            idx = np.argmin(np.abs(_feasible_actions - delta_target / (ensemble_size-j)))
                            weights[idx] = len_feasibles*(1+j) # the last few systems should try to match the power
                            
                            action = np.random.choice(np.repeat(_feasible_actions, weights.astype(int)), 1)
                        else:
                            # choose the action most likely to be feasible
                            action = np.array([batched_neural_model.actions[np.argmax(batched_neural_model.ratings[system_id])]])
                        delta_target -= action
                        selected_actions[system_id] = action                    

                    state, interaction = batched_neural_model.transition(selected_actions)

                    if simulation_time_step + 1 < time_step_count:
                        # inject forecasts
                        batched_neural_model.state =    batched_neural_model.state * (1-masks[:,simulation_time_step + 1]) + \
                                                        masks[:,simulation_time_step + 1] * forecasts[:,simulation_time_step + 1]
                        
                    node = AnyNode(
                        t = simulation_time_step + 1,
                        prior_reward = simulation_node.prior_reward + simulation_node.current_reward,
                        current_reward = -np.abs(delta_target),
                        subsequent_reward_sum = 0,
                        explored_number = 0,
                        action = selected_actions,
                        state = batched_neural_model.state,
                        parent = simulation_node)

                    if node.t == time_step_count:
                        leaf = node
                        leaves.append(node)
                        # reached a leaf
                        # back-propagate information
                        reward_sum = 0
                        while node is not root:
                            parent = node.parent
                            reward_sum += node.current_reward
                            parent.explored_number += 1
                            parent.subsequent_reward_sum += reward_sum
                            node = parent
                        
                        if leaf.prior_reward + leaf.current_reward > best_reward:
                            best_reward = leaf.prior_reward + leaf.current_reward
                            best_leaf = leaf
                            if best_reward == 0: break # optimum found
                    else:
                        simulation_node = node

                if best_reward == 0: break # optimum found
            
            if best_reward == 0: break # optimum found
            
            # Selection
            best_child_reward = float('-inf')
            best_child_node = None
            for child in current_node.children:
                if best_child_reward < child.current_reward + (child.subsequent_reward_sum / max(1,child.explored_number)):
                    best_child_reward = child.current_reward + (child.subsequent_reward_sum / max(1,child.explored_number))
                    best_child_node = child
            current_node = best_child_node

        if best_reward == 0: break # optimum found

    # extract actions
    node = best_leaf
    neural_model_schedules = []
    while node is not root:
        neural_model_schedules.append(node.action)
        node = node.parent
    # reverse array
    neural_model_schedules = neural_model_schedules[::-1]
    
    return (initial_states, forecasts, masks, np.array(neural_model_schedules)[:,:,0], best_reward, target_schedule, None) # DEBUG: swap None and best_leaf


def evaluate_result(simulation_model, initial_states, forecasts, masks, neural_model_schedules, target_schedule, hidden_state, debug_neural_model, debug_beast_leaf):
    resulting_schedules = []
    ensemble_size = len(initial_states)
    simulation_model.eval()

    for i in range(ensemble_size):
        schedule = []
        simulation_model.sample_state() # clears hidden_state
        simulation_model.load_state(initial_states[i], hidden_state)
        """
        # DEBUG
        debug_neural_model.state = [initial_states[i]]
        #"""

        for step, action in enumerate(neural_model_schedules[:,i]):
            """
            # DEBUG
            debug_neural_model.state = debug_neural_model.state * (1-masks[i,step]) + masks[i,step] * forecasts[i,step] # inject forecast
            _ann_feasible = debug_neural_model.feasible_actions
            if action not in _ann_feasible[0]:
                print('!{}'.format(action))
            debug_neural_model.transition([[action]])
            #"""

            simulation_model.state = simulation_model.state * (1-masks[i,step]) + masks[i,step] * forecasts[i,step] # inject forecast
            _sim_feasible = simulation_model.determine_feasible_actions()
            state, interaction = simulation_model.transition(action)
            schedule.append(-interaction[0]) # interaction equals the negated actual action
            """
            # DEBUG
            if interaction[0] != -action:
                print('#')
            #"""
        resulting_schedules.append(schedule)

    return (target_schedule, neural_model_schedules, np.swapaxes(np.array(resulting_schedules),0,1), initial_states, forecasts, masks)


if __name__ == '__main__':
    output_directory = os.path.dirname(__file__)

    # logger setup
    logging_config['handlers']['file']['filename'] = '{}/{}-log.txt'.format(output_directory, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging_config['formatters']['standard']['format'] = '%(message)s'
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('')

    logger.info(state_model_location)
    logger.info(action_model_location)
    # load neural networks (including corresponding simulation model)
    state_meta_data, state_nn_parameters, state_neural_network = load_neural_model(state_model_location)
    action_meta_data, action_nn_parameters, action_neural_network = load_neural_model(action_model_location)
    simulation_model = state_meta_data['model']
    sampling_parameters = state_meta_data['sampling_parameters']
    simulation_model.eval()
    simulation_model.constraint_fuzziness = constraint_fuzziness
    hidden_state = copy.deepcopy(simulation_model.hidden_state)
    state_neural_network.eval()
    action_neural_network.eval()

    #""""
    # HOTFIX for CHPP_HWT and DFH:
    #
    # This block is only required if the original experiment is repeated with the CHPP_HWT or DFH models provided in the repository.
    # The heat demand data loaded from crest_heat_demand.txt during the training and saved into the simulation model was erroneous. 
    # This error did not affect the training process, but does influence the evaluation results, making them better as they should be.
    # Newly trained models do not require this step.
    #
    if configuration == 'chpp' or configuration == 'dfh':
        with open('data/crest_heat_demand.txt', 'r') as file: # read data
            demand_series = np.loadtxt(file, delimiter='\t', skiprows=1) # read file, dismiss header
            demand_series = demand_series.transpose(1,0) # dim 0 identifies the series
            demand_series *= 1000 # kW -> W
        simulation_model.demand.demand_series = demand_series
    # HOTFIX END
    #""""

    # Update sampling parameters
    # CHPP systems
    if type(simulation_model) is HoLL:
        sampling_parameters['temp_distribution'] = ([(35,40), (40,60), (60,65)], [1/20, 18/20, 1/20])
    else:
        sampling_parameters['temp_distribution'] = ([(55,60), (60,80), (80,85)], [1/20, 18/20, 1/20])
    # BESS systems
    sampling_parameters['soc_distribution'] = ([(simulation_model.constraint_fuzziness, 1-simulation_model.constraint_fuzziness)], [1])
    # EVSE systems
    sampling_parameters['possible_capacities'] = [17.6, 27.2, 36.8, 37.9, 52, 70, 85, 100]

    # create neural model
    actions = simulation_model.actions
    neural_model = BatchedNeuralModel(state_meta_data['dt'], actions, 
                        state_neural_network, state_meta_data['input_processor'], state_meta_data['ann_output_processor'], 
                        #classify_by_interaction,
                        action_neural_network, action_meta_data['input_processor'], action_meta_data['ann_output_processor'], 
                        feasibility_threshold=feasibility_threshold)
    
    neural_model.cuda()    

    # conduct experiments
    logger.info('----------------------------------------------------------------')
    logger.info('MCTS for optimizing an ensemble')
    logger.info('----------------------------------------------------------------')
    logger.info('Ensemble size {}'.format(ensemble_size))
    logger.info('Constraint fuzziness {}'.format(simulation_model.constraint_fuzziness))
    logger.info('Max MCTS iterations {}'.format(max_iterations))
    logger.info('Exploration steps {}'.format(exploration_steps))
    for time_step_count in evaluation_periods:
        logger.info('--- Schedule length: {} steps ---'.format(time_step_count))

        target_schedule_file = "{}/{}-{}_{}-target-schedules-of-{}-steps.npy".format(output_directory, ensemble_size, type(simulation_model).__name__, 
                                    target_schedule_count, time_step_count)
        evaluation_results_file = "{}/{}_{}-{}_{}-evaluation-results-for-{}-steps.npy".format(output_directory, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 
                                    ensemble_size, type(simulation_model).__name__, target_schedule_count, time_step_count)

        # load schedules if available in order to save time
        if os.path.isfile(target_schedule_file):
            # schedules have been found, load them
            target_schedule_data = np.load(target_schedule_file, allow_pickle=True)
            logger.info('Read {} schedules from file {}'.format(target_schedule_data.shape[0], target_schedule_file))
        else:
            # no schedules found, generate new ones and save them
            logger.info('Generating {} target schedules'.format(target_schedule_count))
            with mp.Pool(processes=process_count) as pool:
                target_schedule_data = pool.starmap(create_target_schedule, 
                            [(simulation_model, ensemble_size, time_step_count, hidden_state, sampling_parameters) for i in range(target_schedule_count)])
            np.save(target_schedule_file, target_schedule_data)
            logger.info('Saved target schedules to file {}'.format(target_schedule_file))
            
        # target_schedule_data = [(initial_states, forecasts, masks, target_schedule)]

        # try to match the target schedule
        # no schedules found, generate new ones and save them
        logger.info('Conducting MCTS')
        with mp.Pool(processes=process_count) as pool:
            optimization_results = pool.starmap(monte_carlo_tree_search, 
                        [(neural_model, initial_states, forecasts, masks, target_schedule, max_iterations, exploration_steps) for 
                        (initial_states, forecasts, masks, target_schedule) in target_schedule_data])
        # optimization_results = [(initial_states, forecasts, masks, neural_model_schedules, best_reward, target_schedule, debug_best_leaf)]
        rewards = [best_reward for (initial_states, forecasts, masks, neural_model_schedules, best_reward, target_schedule, debug_best_leaf) in optimization_results]
        logger.info('Finished MCTS')
        logger.info('Average expected reward {}'.format(np.average(rewards)))

        # evaluate the result
        logger.info('Conducting evaluation of found solutions')
        with mp.Pool(processes=process_count) as pool:
            evaluation_results = pool.starmap(evaluate_result, 
                        [(simulation_model, initial_states, forecasts, masks, neural_model_schedules, target_schedule, hidden_state, None, debug_beast_leaf) for
                        (initial_states, forecasts, masks, neural_model_schedules, best_reward, target_schedule, debug_beast_leaf) in optimization_results])
        # evaluation_results = [(target_schedule, neural_model_schedules, resulting_schedules, initial_states, forecasts, masks)]
        np.save(evaluation_results_file, evaluation_results)
        logger.info('Finished evaluation')
        logger.info('Saved evaluation results to file {}'.format(target_schedule_file))

        # compute some basic metrics
        resulting_rewards = []
        abs_delta_energy = []
        rel_delta_energy = []
        max_delta_power = []
        for (target_schedule, neural_model_schedules, resulting_schedules, initial_states, forecasts, masks) in evaluation_results:
            total_resulting_schedule = np.sum(resulting_schedules, axis=1)
            
            abs_diff = np.abs(total_resulting_schedule - target_schedule)
            resulting_rewards.append(-np.sum(abs_diff))
            abs_delta_energy.append(np.sum(abs_diff) / 4)
            rel_delta_energy.append(np.sum(abs_diff) / (np.sum(np.abs(target_schedule)))) # (1/4)/(1/4) cancels out
            max_delta_power.append(np.max(abs_diff))
            
        logger.info('Results:')
        logger.info('Average resulting reward {}'.format(np.average(resulting_rewards)))
        logger.info('Average absolute delta energy {}'.format(np.average(abs_delta_energy)))
        logger.info('Average relative delta energy {}'.format(np.average(rel_delta_energy)))
        logger.info('Average absolute deviation {}'.format(-np.average(np.array(resulting_rewards)/time_step_count)))
        logger.info('Max delta power {}'.format(np.max(max_delta_power)))

    logger.info('---')
