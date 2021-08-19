################
# Parameters
# 

# files to read (relative to the directory of this script)
result_files = [
    '2021-02-26_10-13-13_10-BESS_100-evaluation-results-for-8-steps.npy',
    '2021-02-26_10-21-07_10-BESS_100-evaluation-results-for-32-steps.npy',
    '2021-02-26_00-02-25_10-EVSE_100-evaluation-results-for-8-steps.npy',
    '2021-02-26_00-18-11_10-EVSE_100-evaluation-results-for-32-steps.npy',
    '2021-02-25_15-20-45_10-CHPP_HWT_100-evaluation-results-for-8-steps.npy',
    '2021-02-25_15-40-56_10-CHPP_HWT_100-evaluation-results-for-32-steps.npy'
    ]

#
# /Parameters
################ 

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

import logging
import logging.config
from learning.logging_config import logging_config

import datetime

import numpy as np

# logger setup
script_directory = os.path.dirname(__file__)
logging_config['handlers']['file']['filename'] = '{}/{}-log.txt'.format(script_directory, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
logging_config['formatters']['standard']['format'] = '%(message)s'
logging.config.dictConfig(logging_config)
logger = logging.getLogger('')

if len(result_files) > 0:
    logger.info('----------------------------------------------------------------')
    logger.info('Evaluating MCTS results')
    logger.info('----------------------------------------------------------------')
    logger.info('Files: {}'.format(result_files))

    for file_name in result_files:
        full_file_name = os.path.join(script_directory, file_name)
        if os.path.isfile(full_file_name):
            logger.info('---')
            logger.info(file_name)

            # load results from file
            results = np.load(full_file_name, allow_pickle=True)

            # per device metrics
            feasible_count = []
            total_schedule_count = []
            mae_power = []
            mse_power = []

            # ensemble metric
            abs_delta_total_energy = []
            rel_delta_total_energy = []
            mae_total_power = []
            expected_mae_total_power = []
            max_delta_total_power = []
            for (target_schedule, neural_model_schedules, resulting_schedules, initial_states, forecasts, masks) in results:
                # determine resulting schedule
                total_resulting_schedule = np.sum(resulting_schedules, axis=1)

                # per device metrics
                feasible_count.append(np.sum(np.all(neural_model_schedules == resulting_schedules, axis=0))) # on equality a schedule is feasible
                total_schedule_count.append(neural_model_schedules.shape[1])

                deltas = (neural_model_schedules - resulting_schedules).transpose()[np.any(neural_model_schedules != resulting_schedules, axis=0)] # select only infeasible schedules
                if len(deltas) > 0:
                    mae_power.append(np.average(np.abs(deltas)))
                    mse_power.append(np.average(deltas**2))

                # ensemble metrics
                abs_diff = np.abs(total_resulting_schedule - target_schedule)
                abs_delta_total_energy.append(np.sum(abs_diff) / 4)
                rel_delta_total_energy.append(np.sum(abs_diff) / (np.sum(np.abs(target_schedule)))) # (1/4)/(1/4) cancels out
                mae_total_power.append(np.average(abs_diff))
                expected_mae_total_power.append(np.average(np.abs(np.sum(neural_model_schedules, axis=1) - target_schedule)))
                max_delta_total_power.append(np.max(abs_diff))
       
            logger.info('Per device statistics:')
            logger.info('Feasible {}'.format(np.sum(feasible_count)/np.sum(total_schedule_count)))
            if len(mae_power) > 0:
                logger.info('MAE of infeasibles {}'.format(np.average(mae_power)))
            if len(mse_power) > 0:
                logger.info('MSE of infeasibles {}'.format(np.average(mse_power)))

            logger.info('Ensemble statistics:')
            logger.info('Average absolute error (energy) {}'.format(np.average(abs_delta_total_energy)))
            logger.info('Average relative error (energy) {}'.format(np.average(rel_delta_total_energy)))
            logger.info('Mean absolute error (power) {}'.format(np.average(mae_total_power)))
            logger.info('Expected mean absolute error (power) by MCTS output {}'.format(np.average(expected_mae_total_power)))
            logger.info('Max delta power distribution {} (q=[0,0.25,0.5,0.75,0.95,1])'.format(np.quantile(max_delta_total_power, [0,0.25,0.5,0.75,0.95,1])))
            
logger.info('---')