import logging

from collections import OrderedDict
import queue

import os
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from ..neuralnetwork.activation import Swish, MixedActivation
from ..neuralnetwork.layer import SkipConnection, DebugLayer

class ParameterSampler:

    def __init__(self, parameters: {}):
        self.parameters = {}

        # parse parameters
        for key, value in parameters.items():
            # tuple: choose repeatedly
            if type(value) is tuple:
                if len(value) == 2:
                    self.parameters[key] = (np.array(value[0]), value[1], True)
                elif len(value) == 3:
                    self.parameters[key] = (np.array(value[0]), value[1], value[2])
                else:
                    raise ValueError('Expected 2 or 3 values in tuple (value_pool, shape, [replace=True]) for parameter %s'%key)
            # list: single choice
            elif type(value) is list:
                self.parameters[key] = np.array(value)
            elif type(value) is np.ndarray:
                self.parameters[key] = value
            # not implemented
            elif type(value) is dict:
                raise NotImplementedError()
            else:
                raise ValueError('Expected list, tuple or numpy array')
        
    def sample(self) -> {}:
        """
        - None: None
        - tuple: np.random.choice(set, count, replace)
        - list: value[np.random.choice(len(value))]
        """
        params = {}
        for key, value in self.parameters.items():
            choice = None
            if value is None:
                choice = None
            elif type(value) is tuple:
                choice = np.random.choice(value[0], size=value[1], replace=value[2])
            elif type(value) is list or type(value) is np.ndarray:
                choice = value[np.random.choice(len(value))] # only consider first dimension for selection
            else:
                raise ValueError()
            params[key] = choice
        return params


def generate_ann(parameters) -> nn.Module:
    """
    Generates an ANN based on the genome.

    input(input_width) -> 
        for i from 0 to hidden_layer_count:
            [ skip_to(j>i) -> 
            skip_from(0<=j<i) -> 
            batch_norm -> 
            lin(.,width) ]
        -> output(output_width) -> [output_activation]

    #### Arguments
    - parameters: Model parameters specifying topology

        {
            'input_width': [1],
            'output_width': [1],
            'output_activation': [[]],

            'hidden_layer_count': np.arange(4,16),
            'width': [2**i for i in range(5,10)],
            'width_interpolation_steps_input': [0,1,2],
            'width_interpolation_steps_output': [0,1,2],
            'betas': ([0, 0.5, 1, 2, 4, 128], 16),
            'batch_norms': ([0]*14+[1], 16),
            'skips': ([0]*9+[1], (15,15))
        }

        - hidden_layer_count: ``int`` number of hidden layers
        - width: ``int`` output width of each hidden layer
        - betas: ``[float]`` swish activation beta of each hidden layer's activation
        - batch_norms: ``[boolean]`` whether or not a batch norm should be applied before the linear module
        - skips: ``[[boolean]]`` determines whether there is a skip or not. Entry (i,j) for skip from before hidden layer i to before hidden layer j+1.

    
    """
    modules = OrderedDict()        

    # parameters
    input_width = parameters['input_width']
    output_width = parameters['output_width']
    output_activation = parameters['output_activation']
    layer_count = parameters['hidden_layer_count']
    width = parameters['width']
    width_interpolation_steps_input = parameters['width_interpolation_steps_input']
    width_interpolation_steps_output = parameters['width_interpolation_steps_output']
    betas = parameters['betas'][-layer_count:]
    batch_norms = parameters['batch_norms'][:layer_count+1]
    dropout = parameters['dropout']
    skips = np.triu(parameters['skips'][:layer_count, -layer_count:])

    assert(len(betas) == layer_count)
    assert(len(batch_norms) == layer_count+1)

    if type(dropout) is dict:
        probabilities = []
        for key, value in dropout.items():
            probabilities += [float(key)] * int(value)
        probabilities += [0] * (layer_count+1-len(probabilities))
        dropout = probabilities
        np.random.shuffle(dropout) # in place
    else:
        dropout = dropout[:layer_count+1]
    assert(len(dropout) == layer_count+1)

    # dictionary holding dynamically generated modules
    skip_modules = {}

    # widths = [input_width] + [width] * layer_count + [output_width]
    widths = [input_width]
    for i in range(layer_count):
        if i >= layer_count - width_interpolation_steps_output:
            widths.append(int(np.ceil((layer_count-i)*(width-output_width)/(width_interpolation_steps_output+1)+output_width)))
        elif i < width_interpolation_steps_input:
            widths.append(int(np.ceil((i+1)*(width-input_width)/(width_interpolation_steps_input+1)+input_width)))
        else:
            widths.append(width)

    widths.append(output_width)

    # input layer    
    previous_dim = input_width
    for layer, width in enumerate(widths[1:]):
        # skips from (before) this layer to other (subsequent) layers
        for dest in range(skips.shape[1]):
            if layer <= dest and skips[layer,dest] == 1:
                # layer is a source
                skip_modules[(layer, dest)] = SkipConnection()
                modules['%d_skip_to_%d(%d)'%(layer, dest+1, previous_dim)] = skip_modules[(layer, dest)]

        # skips to (before) hidden layer
        for src in range(skips.shape[0]):
            if layer == 0 or src > layer-1:
                break
            if skips[src,layer-1] == 1:
                # layer is a target  
                modules['%d_skip_from_%d(%d)'%(layer, src, widths[src])] = skip_modules[(src, layer-1)]
                # adapt dimension
                previous_dim += widths[src]

        # debug
        #modules['%d_debug'%(layer)] = DebugLayer(layer)

        # batch norm (before) layer
        if batch_norms[layer] == 1:
            # add a batch normalization
            modules['%d_batch_norm(%d)'%(layer, previous_dim)] = nn.BatchNorm1d(previous_dim)
        
        # dropout layer (before)
        if dropout[layer] > 0:
            # add dropout layer
            modules['{}_dropout({})'.format(layer, previous_dim).replace('.', '\'')] = nn.Dropout(dropout[layer])
        
        # linear
        modules['%d_linear(%d,%d)'%(layer, previous_dim, width)] = nn.Linear(previous_dim, width)
        previous_dim = width

        if layer < layer_count:
            # swish
            modules['%d_swish(%s)'%(layer, ('%.2f'%betas[layer]).replace('.', '\''))] = Swish(betas[layer])

    # output activation
    if output_activation is not None:
        modules['%d_activation'%(layer_count)] = MixedActivation(output_activation)
        
    return nn.Sequential(modules)


def random_search(device, sample_count: int, parameters: {}, evaluation_queue: mp.Queue, evaluation_results_queue: mp.Queue, output_dir=None, fully_stored_limit=-1):
    """
    """   
    logger = logging.getLogger('')

    search_log = {}
    fully_stored = {}

    def check_result_queue(block=True):
        try:
            result = evaluation_results_queue.get(block)
            if result is None:
                logger.warning('Received \'None\' as result.')
            else:
                # new result, update corresponding entry
                identifier = result['identifier']
                search_log[identifier]['score'] = result['score']
                search_log[identifier]['NaN'] = result['NaN']
                search_log[identifier]['early_stop'] = result['early_stop']

                if output_dir is not None and fully_stored_limit > 0:
                    # there is a limit for fully stored samples, keep track
                    fully_stored[identifier] = result['score']
                    if len(fully_stored) > fully_stored_limit:
                        # determine and remove worst sample
                        max_key = max(fully_stored, key=(lambda key: fully_stored[key]))
                        logger.info('Removing ANN snapshots for worst sample: {}'.format(max_key))
                        fully_stored.pop(max_key)
                        
                        files = glob.glob(search_log[max_key]['output_dir'] + '/nn*.pt')
                        for file in files:
                            try:
                                os.remove(file)
                            except:
                                logger.warning('Could not delete file {}'.format(file))
            return True
        except queue.Empty:
            pass
        return False

    parameter_sampler = ParameterSampler(parameters)
    n_open_jobs = 0

    for i in range(sample_count):

        # schedule a new job
        try:
            task_id = "sample-%d"%i
            task_data = parameter_sampler.sample()
            task_ann = generate_ann(task_data).to(device)
        except Exception as e:
            logger.error('Job creation failed')
            logger.error(e)
            continue

        if output_dir is not None:
            # create output directory
            task_output_dir = output_dir + '/' + task_id
            try:
                if not os.path.exists(task_output_dir):
                    os.mkdir(task_output_dir)
            except:
                logger.error('Creating directory \'{}\' failed'.format(task_output_dir))
                continue
        else:
            task_output_dir = None

        search_log[task_id] = {
            'data': task_data, 
            'score': float('inf'), 
            'NaN': False, 
            'early_stop': False, 
            'output_dir': task_output_dir
        }
        evaluation_queue.put({
            'identifier': task_id, 
            'data': task_data, 
            'ann': task_ann, 
            'output_dir': task_output_dir
        })
        n_open_jobs += 1

        # check if there is a new result
        n_open_jobs -= check_result_queue(False)

    # wait for all jobs to finish
    while n_open_jobs > 0:
        n_open_jobs -= check_result_queue(True)

    if output_dir is not None:
        np.save(output_dir + '/search_log.npy', search_log)

    return search_log
