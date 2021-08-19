from collections import OrderedDict

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from ..neuralnetwork.activation import Swish, MixedActivation
from ..neuralnetwork.layer import SkipConnection, DebugLayer

class Individual:
    """

    Network genes:
    0. 'width': Width of each layer. len(width) = depth.

        [!] The first entry / first layer represents the input. There are no functions applied in this layer.

        The first and final entries will never change in order to guarantee a given input and output dimension.

    1. 'betas': Betas of the swish activation functions after each hidden layer. len(betas) = depth-2
    2. 'skips': Matrix of {0,1} with 1 indicating a connection from row to column (assumed to be upper triangle matrix, lower triangle is ignored). skip.size(0) = skip.size(1) = depth-1
        Skips start and end after the referenced layer, starting with the input (layer) itself.
    3. 'batch_norms': Vector of {0,1} indicating whether a batch normalization is applied after the layer. len(batch_norms) = depth-1

    """

    def __init__(self, training_genes, network_genes, boundaries_depth, boundaries_width, identifier="unknown"):

        parameters = np.array([])

        #batch_size = np.random.randint(4, max_batch_size)
        learning_rate = training_genes.get('learning_rate', 1e-4)
        epoch_count = training_genes.get('epoch_count', 1)
        batch_count = training_genes.get('batch_count', 1)
        reg_loss_input_shift = training_genes.get('reg_loss_input_shift', 10000)
        reg_loss_input_scale = training_genes.get('reg_loss_input_scale', 100)
        grad_norm_clip = training_genes.get('grad_norm_clip', 1)

        self.training_genes = np.array([learning_rate, epoch_count, batch_count, reg_loss_input_shift, reg_loss_input_scale, grad_norm_clip])

        width = network_genes.get('width', np.array([1,1])) # [1,1] => (1) -> Layer -> (1)
        betas = network_genes.get('betas', np.array([])) # no standard activation for last layer
        skips = network_genes.get('skips', np.zeros((1,1))) # [0]
        batch_norms = network_genes.get('batch_norms', np.array([0]))
        # residual ? -> skip may be enough ?
        # gates?

        self.network_genes = np.array([width, betas, skips, batch_norms])   

        self.boundaries_depth = boundaries_depth
        self.boundaries_width = boundaries_width

        self._identifier = identifier

        self.aggregated_score = 0
        self.score_count = 0
    
    def __str__(self):
        return '(score[%f], \ntraining[%s], \nnetwork[%s])'%(self.score, self.training_genes, self.network_genes)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        equiv = np.array_equal(self.training_genes, other.training_genes)
        if equiv == True:
            for i, element in enumerate(self.network_genes):
                equiv = equiv and np.array_equal(element, other.network_genes[i])
        return  equiv

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier

    def add_score(self, score):
        self.aggregated_score += score
        self.score_count += 1
    
    @property
    def score(self):
        if self.score_count == 0:
            return float('inf')
        return self.aggregated_score / self.score_count

    # Training

    @property
    def learning_rate(self):
        return self.training_genes[0]
    
    @property
    def epoch_count(self):
        return np.rint(self.training_genes[1]).astype(np.int)
    
    @property
    def batch_count(self):
        return np.rint(self.training_genes[2]).astype(np.int)

    @property
    def reg_loss_input_shift(self):
        return self.training_genes[3]

    @property
    def reg_loss_input_scale(self):
        return self.training_genes[4]

    @property
    def max_grad_norm(self):
        return self.training_genes[5]

    # Neural Network

    @property
    def width(self):
        return self.network_genes[0]

    @property
    def betas(self):
        return self.network_genes[1]

    @property
    def skips(self):
        return self.network_genes[2]

    @property
    def batch_norms(self):
        return self.network_genes[3]

    def generate_ann(self, output_activation: {}={}) -> nn.Module:
        """
        Generates an ANN based on the genome.
        
        Parameters:
        - output_activation: mapping from activation functions to the number of elements the function is applied to, e.g., {nn.ReLU(): 2}
        """
        modules = OrderedDict()        
        
        skips = self.skips
        skip_modules = {} # dictionary holding dynamically generated modules

        # checking dimensions
        assert len(self.betas) == len(self.width) - 2
        assert len(self.batch_norms) == len(self.width) - 1
        assert skips.shape == (len(self.width)-1, len(self.width)-1)

        # input 'layer'
        previous_dim = self.width[0]

        for layer, width in enumerate(self.width[1:]):
            # skips from (before) layer
            for dest in range(skips.shape[1]):
                if layer < dest and skips[layer,dest] == 1:
                    # layer is a source
                    skip_modules[(layer, dest)] = SkipConnection()
                    modules['%d_skip_to_%d(%d)'%(layer, dest, previous_dim)] = skip_modules[(layer, dest)]
            
            # skips to (before) layer
            for src in range(skips.shape[0]):
                if src >= layer:
                    break
                if skips[src,layer] == 1:
                    # layer is a target  
                    modules['%d_skip_from_%d(%d)'%(layer, src, self.width[src])] = skip_modules[(src, layer)]
                    # adapt dimension
                    previous_dim += self.width[src]

            # batch norm (before) layer
            if self.batch_norms[layer] == 1:
                # add a batch normalization
                modules['%d_batch_norm(%d)'%(layer, previous_dim)] = nn.BatchNorm1d(previous_dim)
            
            # debug
            #modules['%d_debug'%(layer)] = DebugLayer(layer)

            # linear
            modules['%d_linear(%d,%d)'%(layer, previous_dim, width)] = nn.Linear(previous_dim, width)
            previous_dim = width

            # swish
            if layer < len(self.betas):    
                modules['%d_swish(%s)'%(layer, ('%.2f'%self.betas[layer]).replace('.', '_'))] = Swish(self.betas[layer])

        # last layer
        if len(output_activation) > 0:
            modules['%d_activation'%layer] = MixedActivation(output_activation)
            
        return nn.Sequential(modules)

        
    def cross(self, other: 'Individual', probabilities: {str: float} = {}) -> 'Individual':       
        # Note: self defines the shape of the network
        
        # percentage of genes taken from a
        prob_pass_own_genes = probabilities.get('pass_own_genes', 0.75)
        
        # cross training genes
        training_genes = np.copy(self.training_genes)
        mask_self = np.random.rand(*training_genes.shape) < prob_pass_own_genes
        mask_other = np.ones(training_genes.shape) - mask_self
        training_genes = training_genes * mask_self + other.training_genes * mask_other
        
        # cross network genes
        network_genes = []
        for array_self, array_other in zip(self.network_genes, other.network_genes):
            shape_max = max(array_self.shape, array_other.shape) # tuples required
            shape_self = np.array(array_self.shape, dtype=int) # np array for computing delta
            shape_other = np.array(array_other.shape, dtype=int) # np array for computing delta       
            
            mask_self = (np.random.rand(*shape_max) < prob_pass_own_genes).astype(int)
            mask_other = np.ones(shape_max, dtype=int) - mask_self
                                
            # pad the arrays to ensure identical shapes
            padding = np.transpose(np.vstack([np.zeros(len(shape_max), dtype=int), shape_max - shape_self]))
            array_self = np.pad(array_self, padding, mode='constant', constant_values=0)
            
            padding = np.transpose(np.vstack([np.zeros(len(shape_max), dtype=int), shape_max - shape_other]))
            array_other = np.pad(array_other, padding, mode='constant', constant_values=0)
            
            # now reduce the size again to match the shape of array a
            network_genes.append((array_self * mask_self + array_other * mask_other)[tuple(map(slice, shape_self))])
        
        # create a new individual and assign the new genome
        child = Individual({}, {}, self.boundaries_depth, self.boundaries_width)
        child.training_genes = training_genes
        child.network_genes = np.array(network_genes)
        child.network_genes[0][0] = self.width[0] # restore width for the input layer    
        child.network_genes[0][-1] = self.width[-1] # restore width for the output layer    
        return child
    

    def mutate(self, probabilities: {str: float} = {}) -> 'Individual':
    
        # parameters
        prob_adding_layer = probabilities.get('adding_layer', 0.1)
        prob_removing_layer = probabilities.get('removing_layer', 0.1)
        prob_scale_layer = probabilities.get('scale_layer', 0.5)
        prob_change_betas = probabilities.get('change_betas', 0.5)
        prob_change_batch_norms = probabilities.get('change_batch_norms', 0.25)
        prob_reset_batch_norms = probabilities.get('reset_batch_norms', 0.1)
        prob_change_skip = probabilities.get('change_skip', 0.4)
        prob_reset_skip = probabilities.get('reset_skip', 0.1)
        
        training_genes = np.copy(self.training_genes)
        network_genes = []
        for genes in self.network_genes:
            network_genes.append(np.copy(genes))
        network_genes = np.array(network_genes)
        
        # remove a layer
        if len(self.width) > self.boundaries_depth[0] and np.random.random() < prob_removing_layer:        
            mask = np.ones(len(self.width), dtype=bool)
            mask[np.random.randint(1, len(mask)-1)] = False
            network_genes[0] = network_genes[0][mask]
            
        # add a layer
        if len(self.width) < self.boundaries_depth[1] and np.random.random() < prob_adding_layer:        
            new_layer_width = np.random.randint(*self.boundaries_width)
            new_layer_location = np.random.randint(1, len(network_genes[0]))
            network_genes[0] = np.insert(network_genes[0], new_layer_location, new_layer_width)
            
        # change layer width
        if np.random.random() < prob_scale_layer:
            scale = np.random.uniform(0.5, 2, len(network_genes[0]))
            scale[0] = scale[-1] = 1
            network_genes[0] = (network_genes[0] * scale).clip(*self.boundaries_width).astype(int)
        
        # adapt other genes accordingly
        
        # betas
        delta = len(network_genes[0]) - 2 - len(network_genes[1])
        if delta > 0:
            # add
            network_genes[1] = np.concatenate((network_genes[1], np.ones(delta)))
        elif delta < 0:
            # remove
            network_genes[1] = network_genes[1][:delta]
        
        # batch_norms
        delta = len(network_genes[0]) - 1 - len(network_genes[3])
        if delta > 0:
            # add
            network_genes[3] = np.concatenate((network_genes[3], np.zeros(delta)))
        elif delta < 0:
            # remove
            network_genes[3] = network_genes[3][:delta]
            
        # skip
        delta = len(network_genes[0]) - 1 - network_genes[2].shape[0]
        if delta > 0:
            padding = np.array([[0, delta], [0, delta]], dtype=int)
            network_genes[2] = np.pad(network_genes[2], padding, mode='constant', constant_values=0)
        elif delta < 0:
            network_genes[2] = network_genes[2][:delta, :delta]
            
        # mutate other genes
        
        # betas
        if np.random.random() < prob_change_betas:
            network_genes[1] += np.random.uniform(-1, 1, len(network_genes[1]))
        
        # batch_norms
        if np.random.random() < prob_change_batch_norms:
            network_genes[3] += np.random.randint(-1, 2, len(network_genes[3])).clip(0, 1).astype(int)
            
        if np.random.random() < prob_reset_batch_norms:
            network_genes[3] = np.zeros(len(network_genes[3]), dtype=int)
            
        # skip
        if np.random.random() < prob_change_skip:
            network_genes[2] += np.random.randint(-1, 2, network_genes[2].shape)
            network_genes[2] = np.triu(network_genes[2].clip(0, 1).astype(int), 1)        
            
        if np.random.random() < prob_reset_skip:
            network_genes[2] = np.zeros(network_genes[2].shape, dtype=int)
                
        mutant = Individual({}, {}, self.boundaries_depth, self.boundaries_width)
        mutant.training_genes = training_genes
        mutant.network_genes = np.array(network_genes)
        mutant.network_genes[0][0] = self.width[0] # restore width for the input layer    
        mutant.network_genes[0][-1] = self.width[-1] # restore width for the output layer
        return mutant


def generate_individual(identifier, boundaries, probabilities):
    """
    Randomly generates an individual within the given boundaries.

    Parameters:
    - boundaries:
        - 'learning_rate': x in (min, max) -> 10**x
        - 'epoch_count': x in (min, max) -> 10**x
        - 'batch_count': x in (min, max) -> 10**x
        - 'reg_loss_input_shift': x in (min, max) -> 10**x
        - 'reg_loss_input_scale': x in (min, max) -> 10**x
        - 'grad_norm_clip': x in (min, max) -> 10**x

        - 'input_dim': x in (min, max)
        - 'output_dim': x in (min, max)
        - 'depth': x in (min, max)
        - 'width': x in (min, max) -> 2**x
        - 'betas': x in (min, max) -> 10**x

    - probabilities:
        - 'skips':
        - 'batch_norms':
    """
    
    training_genes = {
        'learning_rate': 10**np.random.uniform(*boundaries['learning_rate']),
        'epoch_count': 10**np.random.uniform(*boundaries['epoch_count']),
        'batch_count': 10**np.random.uniform(*boundaries['batch_count']),
        'reg_loss_input_shift': 10**np.random.uniform(*boundaries['reg_loss_input_shift']),
        'reg_loss_input_scale': 10**np.random.uniform(*boundaries['reg_loss_input_scale']),
        'grad_norm_clip': 10**np.random.uniform(*boundaries['grad_norm_clip']),
    }

    depth = np.random.randint(*boundaries['depth'])
        
    width = [2**np.random.randint(*boundaries['width'])]*depth
    width[0] = boundaries['input_dim']
    width[-1] = boundaries['output_dim']

    skips = (np.random.random((depth-1, depth-1)) < probabilities['skips']) * 1
    skips = np.triu(skips, 1)    

    network_genes = {
        'width': np.array(width, dtype=int),
        'betas': np.array([10**np.random.uniform(*boundaries['betas'])]*(depth-2)),
        'skips': skips,
        'batch_norms': (np.random.random(depth-1) < probabilities['batch_norms']) * 1,
    }

    return Individual(training_genes, network_genes, boundaries['depth'], boundaries['width'], identifier=identifier)


def search_best_individual(population: np.ndarray, 
                        target_population_size: int, 
                        generation_count: int, 
                        probabilities: {}, 
                        evaluation_repetition_count: int, 
                        evaluation_queue: mp.Queue, 
                        evaluation_results_queue: mp.Queue,
                        output_dir=None):
    """

    Performs an optimization using an evolutionary algorithm to search for a good model.

    Parameters:

    - population:
    - generation_count:
    - evaluation_func: Function for evaluating an individual, e.g., training and evaluating the performance of an ANN generated from an individual.
    
    - probabilities:

        probabilities = {
            'mating': {
                '_': 0.5,
                ...
            },
            'mutating': {
                '_': 0.5,
                ...
            },
        }
    """   
    logger = logging.getLogger('')

    def evaluate(individuals):
        n_jobs = 0
        for i in range(evaluation_repetition_count):
            for individual in individuals:
                evaluation_queue.put(individual)
                n_jobs += 1

        # wait for all evaluation jobs to finish
        while n_jobs > 0:
            result = evaluation_results_queue.get()
            if result is None:
                logger.warning('Received \'None\' as result.')
            else:
                # new result, find corresponding individual
                for individual in population:
                    if result[0] == individual:
                        # add to old results
                        individual.add_score(*result[1:])
            n_jobs -= 1

    logger.info('[EA] Evaluating generation 0')
    evaluate(population)

    if output_dir is not None:
        np.save(output_dir + '/population0.npy', population)

    # compute average scores
    scores = np.array([individual.score for individual in population])

    for generation_counter in range(1, generation_count):
        logger.info('[EA] Generating generation %d'%generation_counter)
        individual_counter = 0
        new_individuals = np.array([])

        # randomly determine individuals and perform crossover
        mating_pool = np.arange(len(population))[np.random.rand(len(population)) > probabilities['mating']['_']]
        while len(mating_pool) > 1:
            idx_a, idx_b = np.random.choice(mating_pool, 2, replace=False)
            mating_pool = np.setdiff1d(mating_pool, idx_b)
            offspring = population[idx_a].cross(population[idx_b], probabilities['mating'])
            if offspring not in population:
                offspring.identifier = 'g%di%d.c.%s+%s'%(generation_counter, individual_counter, population[idx_a].identifier.split('.')[0], population[idx_b].identifier.split('.')[0])
                individual_counter += 1
                new_individuals = np.append(new_individuals, offspring)
        
        # randomly mutate individuals
        mutant_pool = np.arange(len(population))[np.random.rand(len(population)) > probabilities['mutating']['_']]
        for idx in mutant_pool:
            mutant = population[idx].mutate(probabilities['mutating'])
            if mutant not in population:
                mutant.identifier = 'g%di%d.m.%s]'%(generation_counter, individual_counter, population[idx].identifier.split('.')[0])
                individual_counter += 1
                new_individuals = np.append(new_individuals, mutant)
                
        # add new individuals to population
        population = np.append(population, new_individuals)

        logger.info('[EA] Evaluating generation %d'%generation_counter)
        evaluate(new_individuals)

        # compute average scores
        scores = np.array([individual.score for individual in population])

        # reduce population size
        # Note: there may be multiple entries with a score equal to min_score
        # in this case the population size is larger than target_population_size
        min_score = np.sort(scores)[-target_population_size]
        population = np.array([individual for individual in population if individual.score >= min_score])
        scores = scores[scores >= min_score]

        if output_dir is not None:
            np.save(output_dir + '/population' + str(generation_counter) + '.npy', population)


    individual_counter = np.argmax(scores)  
    return population[individual_counter]