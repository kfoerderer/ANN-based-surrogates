import traceback
import sys

import logging
import logging.handlers
import os
import shutil
import queue
import pprint

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.multiprocessing as mp

from ..neuralnetwork.samples import SampleCache

class TrainingCallback:
     
     def __call__(self, epoch, **kwargs):
         pass


class EarlyStoppingCallback(TrainingCallback):
    """
    Callback for premature training stopping, when the results are bad or do not improve

    Arguments
    - stopping_scores: {epoch: max_median_score}
    - improvement_window: number of epochs to wait before stopping for failing to improve
    """
    def __init__(self, stopping_scores: {int: float}, improvement_window: int=100):
        super().__init__()
        self.stopping_scores = stopping_scores
        self.improvement_window = improvement_window

    def __str__(self):
        return 'EarlyStoppingCallback(stopping_scores={}, improvement_window={})'.format(self.stopping_scores, self.improvement_window)

    def __repr__(self):
        return self.__str__()

    def __call__(self, epoch, evaluation_loss_statistics, **kwargs):
        max_score = self.stopping_scores.get(epoch, float('inf'))
        if np.average(evaluation_loss_statistics[epoch]) > max_score:
            # result is too bad
            return True
        best_score_epoch = np.argmin(np.average(evaluation_loss_statistics[:epoch+1], axis=1))
        if best_score_epoch + self.improvement_window < epoch:
            # it has been too long since the last improvement
            return True
        return False
         
class ANNTrainer(mp.Process):
    """
    Process for training ANNs

    Input Queue: {identifier: str, data: {}, ann: nn.Module, output_dir}
    Output Queue: {identifier: str, score: float, NaN: boolean, early_stop: boolean}

    """

    def __init__(self, 
                training_cache: SampleCache,
                training_cache_cuda: torch.Tensor,
                evaluation_cache: SampleCache,
                evaluation_cache_cuda: torch.Tensor,
                evaluation_batch_count: int,
                input_queue: mp.Queue, 
                output_queue: mp.Queue, 
                log_queue=None,
                print_cache_renewals=False):
        super().__init__()
        
        # assign the shared cuda Tensors (and load the torch kernels into VRAM)
        if training_cache_cuda is not None:
            training_cache.write_to_cuda(training_cache_cuda)
        if evaluation_cache_cuda is not None:
            evaluation_cache.write_to_cuda(evaluation_cache_cuda)

        self.training_cache = training_cache
        self.evaluation_cache = evaluation_cache
        self.evaluation_batch_count = int(evaluation_batch_count)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.log_queue = log_queue
        self.print_cache_renewals = print_cache_renewals
        
    def run(self):
        #cache_data = self.training_cache.get_data() # debug
        try:
            debug_switch = True

            if self.log_queue is not None:
                logger = logging.getLogger('')
                logger.addHandler(logging.handlers.QueueHandler(self.log_queue))
                logger.setLevel(logging.INFO)

            logger = logging.getLogger('')
            logger.info('Process {} starts (PID={})'.format(self.name, os.getpid()))

            # frequently needed variables/objects
            training_cache = self.training_cache
            evaluation_cache = self.evaluation_cache
            evaluation_batch_count = self.evaluation_batch_count  
            print_cache_renewals = self.print_cache_renewals

            training_logger = logging.getLogger('{}'.format(self.name))
            file_handler = None

            pp = pprint.PrettyPrinter(indent=4)

            #torch.autograd.set_detect_anomaly(True)

            while self.input_queue is not None:
                result = None
                try:
                    # remove old handler if existing
                    if file_handler is not None:
                        training_logger.removeHandler(file_handler)

                    # get next task
                    task = self.input_queue.get(True)
                    task_id = task['identifier']
                    task_data = task['data']
                    task_ann = task['ann']
                    task_output_dir = task['output_dir']

                    result = {
                        'identifier': task_id, 
                        'score': float('inf'), 
                        'NaN': False, 
                        'early_stop': False
                    }
                    best_result_epoch = -1
                    
                    # logging (if results are stored)
                    if task_output_dir is not None:
                        file_handler = logging.FileHandler(task_output_dir + '/' + '_log.txt')
                        file_handler.setFormatter(logger.handlers[0].formatter)
                        training_logger.addHandler(file_handler)

                        # persist data in torch format (it may contain torch Modules)
                        meta_data = training_cache.get_meta_data()
                        torch.save(meta_data, task_output_dir + '/_meta.pt')
                        torch.save(task_data, task_output_dir + '/_parameters.pt')

                        # write human readable data
                        with open(task_output_dir + '/_summary.txt', 'x') as file:
                            file.write('meta data:\n{}\n---\n'.format(pp.pformat(meta_data)))
                            file.write('parameters:\n{}\n---\n'.format(pp.pformat(task_data)))
                            file.write('neural network:\n{}\n---\n'.format(str(task_ann)))
                            param_count = sum(p.numel() for p in task_ann.parameters() if p.requires_grad)
                            file.write('parameter count:\n{}\n---\n'.format(param_count))

                    # learning parameters
                    loss_function = task_data['loss']
                    batch_size = task_data['batch_size']
                    reg_loss_function = task_data['regularization']

                    learning_rate = task_data['learning_rate']
                    epoch_count = int(task_data['epoch_count'])
                    batch_count = int(task_data['batch_count'])
                    max_grad_norm = task_data['max_grad_norm']
                    lr_scheduler = task_data.get('lr_scheduler', (None, None))

                    early_stopping_callback = task_data.get('early_stopping_callback', None)

                    epoch_print_digits = int(np.log(epoch_count)/np.log(10))

                    training_logger.info('Starting job {}'.format(task_id))

                    # create solver
                    solver = torch.optim.Adam(task_ann.parameters(), lr=learning_rate)

                    # create lr scheduler
                    if lr_scheduler[0] is None:
                        lr_scheduler = None
                    else:
                        # create an object of class [0] with keyword arguments [1]
                        lr_scheduler = lr_scheduler[0](solver, **lr_scheduler[1])

                    # data structures for collecting statistics
                    training_loss_statistics = np.zeros((epoch_count,5))
                    evaluation_loss_statistics = np.zeros((epoch_count,5))

                    for epoch in range(epoch_count):
                        # training
                        if training_cache.has_cuda_tensor():
                            training_cache.write_to_cuda()
                        task_ann.train()
                        batch_sequence = None
                        losses = torch.zeros((batch_count,3), device=next(task_ann.parameters()).device)
                        for batch in range(batch_count):
                            # get batch
                            batch_in, batch_target, batch_sequence = training_cache.get_batch(batch_size, batch_sequence)

                            # compute loss
                            batch_out = task_ann(batch_in)
                            loss = loss_function(batch_out, batch_target)

                            if reg_loss_function is not None:
                                reg_loss = reg_loss_function(task_ann)
                                loss += reg_loss
                                losses[batch][1] = reg_loss

                            if torch.isnan(loss).any():
                                logger.info('Model {}, training epoch {:d}/{:d}, stopping due to NaN'.format(task_id, epoch, epoch_count-1))
                                if (batch_in > 100).any():
                                    logger.warning('There are large inputs (max={}) passed to the ANN.'.format(float(torch.max(batch_in))))
                                result['NaN'] = True
                                raise ValueError('Got NaN in loss')

                            # compute gradients
                            solver.zero_grad()
                            loss.backward()

                            # regularization using norm clipping
                            losses[batch][2] = torch.nn.utils.clip_grad_norm_(task_ann.parameters(), max_grad_norm)

                            solver.step()

                            # store losses
                            losses[batch][0] = loss

                        training_loss_statistics[epoch] = np.quantile(losses[:,0].cpu().detach().numpy(), [0,0.25,0.5,0.75,1])

                        training_logger.info('Model {}, training epoch {:d}/{:d}, {}'.format(task_id, epoch, epoch_count-1, str(training_loss_statistics[epoch])))

                        # write model to disk
                        if task_output_dir is not None:
                            # [!] if the filename is changed, it must also be changed further below, where a copy of the best ANN is created
                            torch.save(task_ann.state_dict(), task_output_dir + '/nn' + '{:0{}d}'.format(epoch, epoch_print_digits) + '.pt')

                        # evaluation
                        if evaluation_cache.has_cuda_tensor():
                            evaluation_cache.write_to_cuda()
                        task_ann.eval()
                        batch_sequence = np.arange(evaluation_cache.current_size.value) # fixed sequence for better comparability (as long as batch_count <= cache size)
                        losses = torch.zeros((evaluation_batch_count), device=next(task_ann.parameters()).device)
                        with torch.no_grad():
                            for batch in range(evaluation_batch_count):
                                # get batch
                                batch_in, batch_target, batch_sequence = evaluation_cache.get_batch(batch_size, batch_sequence)
                                # compute loss
                                batch_out = task_ann(batch_in)
                                loss = loss_function(batch_out, batch_target)

                                if debug_switch and loss > 10000:
                                    debug_switch = False
                                    logger.warning(batch_in.cpu())
                                    logger.warning(batch_out.cpu())
                                    logger.warning(batch_target.cpu())

                                if reg_loss_function is not None:
                                    loss += reg_loss_function(task_ann)

                                # store losses
                                losses[batch] = loss

                        evaluation_loss_statistics[epoch] = np.quantile(losses.cpu().numpy(), [0,0.25,0.5,0.75,1])

                        score = np.average(evaluation_loss_statistics[epoch])  # average of quantiles [0, 0.25, 0.5, 0.75, 1]
                        if result['score'] > score:
                            result['score'] = score
                            best_result_epoch = epoch

                        training_logger.info('Model {}, evaluation epoch {:d}/{:d}, {}'.format(task_id, epoch, epoch_count-1, str(evaluation_loss_statistics[epoch])))

                        # stop early for bad scores
                        if early_stopping_callback is not None:
                            if(early_stopping_callback(epoch, evaluation_loss_statistics=evaluation_loss_statistics)):
                                training_logger.info('Early stop for model {} at epoch {}'.format(task_id, epoch))
                                result['early_stop'] = True
                                break
                            
                        # adapt learning rate
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            training_logger.info('Learning rate for next epoch is {:E}'.format(lr_scheduler.get_lr()[0]))

                        # info
                        if print_cache_renewals:
                            training_logger.info('Number of total cache renewals is {} and {}'.format(training_cache.get_renewal_count(), evaluation_cache.get_renewal_count()))

                    # store statistics
                    if task_output_dir is not None:
                        np.save(task_output_dir + '/_loss_statistics_training.npy', evaluation_loss_statistics)
                        np.save(task_output_dir + '/_loss_statistics_evaluation.npy', evaluation_loss_statistics)

                        # create a copy of the best scored ann
                        shutil.copyfile(task_output_dir + '/nn' + '{:0{}d}'.format(best_result_epoch, epoch_print_digits) + '.pt',
                                        task_output_dir + '/_nn.pt')

                    training_logger.info('Finished training of model {} with score {:f} in epoch {}'.format(task_id, result['score'], best_result_epoch))

                except queue.Empty:
                    pass
                except KeyboardInterrupt as err:
                    raise err # escalate error
                except ValueError as err:
                    training_logger.exception(err)
                    #traceback.print_tb(err.__traceback__)
                except Exception as err:
                    training_logger.exception(err)
                    #traceback.print_tb(err.__traceback__)
                finally:
                    self.output_queue.put(result)

            logger.info('Process {} exits'.format(self.name))
        except KeyboardInterrupt:
            traceback.print_exc(file=sys.stdout)
            logger.info('Received keyboard interrupt. Closing process {}.'.format(self.name))




