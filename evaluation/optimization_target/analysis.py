################
# Parameters
# 

# files to read (relative to the directory of this script)
result_files = [
    #'2021-02-26_10-13-13_10-BESS_100-evaluation-results-for-8-steps.npy',
    #'2021-02-26_10-21-07_10-BESS_100-evaluation-results-for-32-steps.npy',
    #'2021-02-26_00-02-25_10-EVSE_100-evaluation-results-for-8-steps.npy',
    #'2021-02-26_00-18-11_10-EVSE_100-evaluation-results-for-32-steps.npy',
    '2021-02-25_15-20-45_10-CHPP_HWT_100-evaluation-results-for-8-steps.npy',
    #'2021-02-25_15-40-56_10-CHPP_HWT_100-evaluation-results-for-32-steps.npy'
    ]

configuration = 'chpp'
state_model_location = 'evaluation/schedule_generation/results/' + configuration + '/state'
action_model_location = 'evaluation/schedule_generation/results/' + configuration + '/actions'

feasibility_threshold = 0.5
constraint_fuzziness = 0.02

#
# /Parameters
################ 

import sys
import os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

import logging
import logging.config
from learning.logging_config import logging_config

import datetime
import copy

import numpy as np

from modules.utils import load_neural_model
from modules.neuralnetwork.models.statebased import BatchedNeuralModel

# load neural networks (including corresponding simulation model)
state_meta_data, state_nn_parameters, state_neural_network = load_neural_model(state_model_location)
action_meta_data, action_nn_parameters, action_neural_network = load_neural_model(action_model_location)
simulation_model = state_meta_data['model']
sampling_parameters = state_meta_data['sampling_parameters']
simulation_model.eval()
simulation_model.constraint_fuzziness = constraint_fuzziness
dummy_hidden_state = copy.deepcopy(simulation_model.hidden_state)
state_neural_network.eval()
action_neural_network.eval()

actions = simulation_model.actions
neural_model = BatchedNeuralModel(state_meta_data['dt'], actions, 
                    state_neural_network, state_meta_data['input_processor'], state_meta_data['ann_output_processor'], 
                    #classify_by_interaction,
                    action_neural_network, action_meta_data['input_processor'], action_meta_data['ann_output_processor'], 
                    feasibility_threshold=feasibility_threshold)

def evaluate_result(simulation_model, initial_states, forecasts, masks, neural_model_schedules, target_schedule, hidden_state, neural_model, logger: logging.Logger):
    resulting_schedules = []
    ensemble_size = len(initial_states)
    simulation_model.eval()

    def log_header(initial_state, schedule, printed_header):
        if printed_header == False: 
            logger.info('Initial state {}'.format(repr(initial_state).replace('\n','')))
            logger.info('Schedule {}'.format(repr(schedule).replace('\n','')))
        return True

    def log_footer():
        logger.info('---')

    for i in range(ensemble_size):
        schedule = []
        simulation_model.sample_state() # clears hidden_state
        simulation_model.load_state(initial_states[i], hidden_state)
        
        neural_model.state = [initial_states[i]]

        printed_header = False
        for step, action in enumerate(neural_model_schedules[:,i]):

            neural_model.state = neural_model.state * (1-masks[i,step]) + masks[i,step] * forecasts[i,step] # inject forecast
            neural_state = np.copy(neural_model.state)
            _ann_feasible = neural_model.feasible_actions
            if action not in _ann_feasible[0]:
                printed_header = log_header(initial_states[i], neural_model_schedules[:,i], printed_header)
                logger.info('- step {} -'.format(step))
                logger.error('! neural model inconsistence; action {} at state {}'.format(action, repr(neural_state).replace('\n','')))
            neural_model.transition([[action]])

            simulation_model.state = simulation_model.state * (1-masks[i,step]) + masks[i,step] * forecasts[i,step] # inject forecast
            simulation_state = np.copy(simulation_model.state)
            _sim_feasible = simulation_model.determine_feasible_actions()
            state, interaction = simulation_model.transition(action)
            schedule.append(-interaction[0]) # interaction equals the negated actual action

            if interaction[0] != -action:
                printed_header = log_header(initial_states[i], neural_model_schedules[:,i], printed_header)
                logger.info('- step {} -'.format(step))
                logger.info('requested {} at state {},\n but got {} at state {}'.format(action, repr(neural_state).replace('\n',''), -interaction[0], repr(simulation_state).replace('\n','')))

        if printed_header == True:
            log_footer()
        resulting_schedules.append(schedule)

    return (target_schedule, neural_model_schedules, np.swapaxes(np.array(resulting_schedules),0,1), initial_states, forecasts, masks)


if len(result_files) > 0:
    np.set_printoptions(suppress=True)
    # logger setup
    script_directory = os.path.dirname(__file__)
    logging_config['handlers']['file']['filename'] = '{}/{}-analysis-log.txt'.format(script_directory, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging_config['formatters']['standard']['format'] = '%(message)s'
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('')

    logger.info('----------------------------------------------------------------')
    logger.info('Analysis of MCTS results')
    logger.info('----------------------------------------------------------------')
    logger.info('Files: {}'.format(result_files))

    for file_name in result_files:
        full_file_name = os.path.join(script_directory, file_name)
        if os.path.isfile(full_file_name):
            logger.info('---')
            logger.info(file_name)

            # load results from file
            results = np.load(full_file_name, allow_pickle=True)

            for (target_schedule, neural_model_schedules, resulting_schedules, initial_states, forecasts, masks) in results:
                
                evaluate_result(simulation_model, initial_states, forecasts, masks, neural_model_schedules, target_schedule, dummy_hidden_state, neural_model, logger)

logger.info('---')