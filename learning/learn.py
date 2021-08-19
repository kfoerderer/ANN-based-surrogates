import sys, os, shutil
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from modules.utils import process_log_records, register_kill_hook, register_debug_hook

# torch
import torch

# multiprocessing
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
import platform
if platform.system() == 'Linux':
    mp.set_start_method('spawn', True)

# logging
import logging
import logging.config
import threading

# numpy
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=150)

# plotting
#get_ipython().run_line_magic('pylab', 'inline')
#import matplotlib.pyplot as plt

# others
import datetime


from modules.neuralnetwork.samples import SampleCache
"""
class MManager(BaseManager): pass
MManager.register('SampleCache', SampleCache,
    exposed=('__getitem__', '__len__', 'get_cache_size', 'is_full', 'put_batch', 'put', 'get_batch', 'get_data', 'get_meta_data', 'get_renewal_count'))
"""


register_kill_hook()

if __name__ == '__main__':    
    print('PID: {}'.format(os.getpid()))
    ####
    # config
    ####

    from learning.statebased.gc_bess_state import \
                        output_location, model, device, sample_generator, sample_input_size, sample_output_size, \
                        cache_initialization_process_count, training_generation_process_count, evaluation_generation_process_count, training_process_count, \
                        training_cache_size, evaluation_cache_size, evaluation_batch_count, \
                        meta_search_sample_count, meta_search_fully_stored_count, meta_search_parameter_space

    model.train()

    assert model.constraint_fuzziness <= 0.1

    """
    # debug
    while True:
        inp, outp = next(sample_generator)
        dummy = '_'
    """

    ####
    # /config
    ####

    output_location = output_location.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    from learning.logging_config import logging_config
    logging_config['handlers']['file']['filename'] = output_location + '/log.txt'

    ####
    # prepare operations
    ####
    print('Current working directory is {}'.format(os.getcwd()))
    print('Writing output to {}'.format(output_location))
    os.mkdir(output_location)

    logging.config.dictConfig(logging_config)
    log_queue = mp.Queue()
    logging_thread = threading.Thread(target=process_log_records, args=(log_queue,))
    logging_thread.start()

    logger = logging.getLogger('')

    # start manager
    #anager = MManager()
    #manager.start()
    
    logger.info('Torch Version \'{}\' on \'{}\''.format(torch.__version__, device))

    ####
    # debug
    ####

    """ Test
    debug_cache = manager.SampleCache(device, 2, sample_input_size, sample_output_size, sample_generator.meta_data)
    for i in range(7):
        sample_input, sample_output = next(sample_generator)
        logger.info(sample_input)
        logger.info(sample_output)
        debug_cache.put(sample_input, sample_output)
        batch = debug_cache.get_batch(min(i+1, 10))
        logger.info(batch)
        logger.info(debug_cache.get_renewal_count())
    sys.exit(0)
    #"""

    try:
        ####
        # create caches
        ####
        training_cache = SampleCache(training_cache_size, sample_input_size, sample_output_size, sample_generator.meta_data)
        evaluation_cache = SampleCache(evaluation_cache_size, sample_input_size, sample_output_size, sample_generator.meta_data)

        ####
        # training
        ####
        from modules.neuralnetwork.samples import SampleGenerator
        from modules.neuralnetwork.training import ANNTrainer
        from modules.hyperparameters.randomsearch import random_search

        generation_processes = []

        task_queue = mp.Queue()
        task_results_queue = mp.Queue()

        training_processes = []

        # fill training cache
        for i in range(cache_initialization_process_count):
            process = SampleGenerator(iter(sample_generator), training_cache, training_cache_size, int(training_cache_size/cache_initialization_process_count), log_queue)
            generation_processes.append(process)
            process.start()

        # wait for cache to fill up
        for process in generation_processes:
            process.join()
        generation_processes = []

        # fill evaluation cache
        for i in range(cache_initialization_process_count):
            process = SampleGenerator(iter(sample_generator), evaluation_cache, evaluation_cache_size, int(evaluation_cache_size/cache_initialization_process_count), log_queue)
            generation_processes.append(process)
            process.start()

        # wait for cache to fill up
        for process in generation_processes:
            process.join()
        generation_processes = []
        
        logger.info('Caches have been initialized.')

        # further sample generation
        for i in range(training_generation_process_count):
            process = SampleGenerator(iter(sample_generator), training_cache, -1, int(training_cache_size/cache_initialization_process_count), log_queue) # write more frequently to cache
            generation_processes.append(process)
            process.daemon = True
            process.start()
        
        for i in range(evaluation_generation_process_count):
            process = SampleGenerator(iter(sample_generator), evaluation_cache, -1, int(evaluation_cache_size/cache_initialization_process_count), log_queue) # write more frequently to cache
            generation_processes.append(process)
            process.daemon = True
            process.start()

        # training
        if device.type == 'cuda':
            training_cache_cuda = training_cache.write_to_cuda()
            evaluation_cache_cuda = evaluation_cache.write_to_cuda()
        else:
            training_cache_cuda = None
            evaluation_cache_cuda = None

        for i in range(training_process_count):
            process = ANNTrainer(training_cache, training_cache_cuda, evaluation_cache, evaluation_cache_cuda, 
                evaluation_batch_count, task_queue, task_results_queue, log_queue, 
                print_cache_renewals=(len(generation_processes)>0))
            training_processes.append(process)
            process.daemon = True
            process.start()

        logger.info('Performing random search by sampling {:d} parameter combinations'.format(meta_search_sample_count))

        search_results = random_search(device,
            meta_search_sample_count,
            meta_search_parameter_space,  
            task_queue, task_results_queue, 
            output_location,
            meta_search_fully_stored_count)

        scores = np.array([[identifier, entry['score'], entry['NaN'], entry['early_stop']] for identifier, entry in search_results.items()], dtype=object)
        logger.info('Results:')
        logger.info(scores)
        min_idx = np.argmin(scores[:,1])
        logger.info('Best result: {}'.format(scores[min_idx]))
        logger.info(str(search_results[scores[min_idx,0]]['data']))

        # create a copy of the best scoring ann
        shutil.copyfile(output_location + '/{}/_meta.pt'.format(scores[min_idx,0]), output_location + '/_meta.pt')
        shutil.copyfile(output_location + '/{}/_nn.pt'.format(scores[min_idx,0]), output_location + '/_nn.pt')
        shutil.copyfile(output_location + '/{}/_parameters.pt'.format(scores[min_idx,0]), output_location + '/_parameters.pt')

        ####
        # debug
        ####

        """ Test
        evaluation_samples = evaluation_cache.get_data()
        actions = torch.sum(evaluation_samples[:,len(model.state):sample_input_size] * torch.Tensor(model.actions/1000), dim=1)
        interactions = evaluation_samples[:,-2]
        print('infeasible actions {}'.format(torch.sum(actions != interactions)))
        print('feasible actions {}'.format(torch.sum(actions == interactions)))
        """
        #"""
    except (KeyboardInterrupt, SystemExit):
        logging.warning('Premature exit')

    ####
    # shutdown code
    ####
    for process in generation_processes:
        process.terminate()
        process.join()

    for process in training_processes:
        process.terminate()
        process.join()

    #manager.shutdown()

    log_queue.put(None)
    logging_thread.join()
    logger.info('Exit')