import os
from typing import Iterable
from typing import Tuple

import logging
import time
import queue

import torch
import torch.multiprocessing as mp
import torch.utils.data

import numpy as np

class SampleCache:
    """
    A cache for generated samples that can be shared between processes.

    (Using a BaseManager caused random lock ups of all processes, therefore this implementation can be used without a base manager)

    The cache is build and managed in RAM, but can be transfered to CUDA if necessary. 
    Activating cuda is required for each individual tensor, to avoid unnecessary loading of the pytorch kernels into the GPU memory.
    """

    def __init__(self, cache_size, input_dim, output_dim, meta_data):
        ## non shared
        # holds meta_data describing the cached samples
        self.meta_data = meta_data

        self.input_dim = input_dim
        self.output_dim = output_dim

        ## shared 
        # index of most recently added sample
        self.most_recent = mp.Value('i', -1)
        # number of samples in cache
        self.current_size = mp.Value('i', 0)
        # additional info
        self.renewal_count = mp.Value('i', -1)
        # tensor holding all samples
        self.data = torch.empty((cache_size, input_dim + output_dim))
        self.data.share_memory_() # move tensor into shared memory
        self.data_cuda = None
        # lock for access control
        self.lock = mp.Lock() # shared
        
    def __getitem__(self, key: int) -> (torch.Tensor, torch.Tensor):
        with self.lock:
            return self.data[key].split((self.input_dim, self.output_dim))
    
    def __len__(self):
        return self.current_size.value

    def size(self):
        return self.data.size(0)

    def is_full(self) -> bool:
        return self.current_size.value == self.data.size(0)

    def get_meta_data(self):
        # create a copy
        return dict(self.meta_data)

    def get_data(self) -> torch.Tensor:
        return self.data

    def has_cuda_tensor(self):
        return self.data_cuda is not None

    def write_to_cuda(self, cuda_tensor: torch.Tensor=None) -> torch.Tensor:
        """
        Copy the cached (or provided) data from RAM into VRAM. This method triggers the loading of torch specific code into VRAM.
        """
        with self.lock:
            if self.data_cuda is None:
                if cuda_tensor is None:
                    self.data_cuda = self.data.cuda()
                    return self.data_cuda
                self.data_cuda = cuda_tensor
            return self.data_cuda.copy_(self.data)

    def get_renewal_count(self):
        return self.renewal_count.value

    def put_batch(self, input_batch: torch.Tensor, output_batch: torch.Tensor):
        with self.lock:
            data = self.data
            cache_size = data.size(0)
            # increment index and start from 0 again if final element is reached        
            start_idx = (self.most_recent.value + 1) % cache_size
            
            # put data
            if start_idx + input_batch.size(0) > cache_size:
                end_idx = input_batch.size(0) - (cache_size - start_idx)
                combined = torch.cat((input_batch, output_batch), 1)
                data[start_idx:] = combined[:-end_idx]
                data[:end_idx] = combined[-end_idx:]
                
                self.renewal_count.value += 1
                self.size = cache_size
            else:
                end_idx = start_idx + input_batch.size(0)
                data[start_idx:end_idx] = torch.cat((input_batch, output_batch), 1)
                
                self.current_size.value = min(self.current_size.value + input_batch.size(0), cache_size)
                if end_idx == cache_size:
                    self.renewal_count.value += 1

            self.most_recent.value = end_idx

    def put(self, sample_input: torch.Tensor, sample_output: torch.Tensor):
        with self.lock:
            data = self.data
            # increment index and start from 0 again if final element is reached        
            most_recent.value = (self.most_recent.value + 1) % data.size(0)
            # put data    
            data[most_recent.value] = torch.cat((sample_input, sample_output), 0)

            self.most_recent.value = most_recent.value
            self.current_size.value = min(self.current_size.value + 1, data.size(0))

            if most_recent.value == 0 and self.size == data.size(0):
                self.renewal_count.value += 1

    def get_batch(self, batch_size: int, sequence: np.ndarray=None) -> (torch.Tensor, torch.Tensor, np.ndarray):
        # determine sequence of samples
        if sequence is None or len(sequence) < batch_size:
            # no pool specified, initialize a new pool
            sequence = np.arange(self.current_size.value)
            np.random.shuffle(sequence)
        elif max(sequence[:batch_size]) > self.data.size(0):
            # there are too few samples, wait for more and try again
            print('--delayed batch delivery--')
            time.sleep(1)
            return self.get_batch(batch_size, sequence)
        
        # get indices
        idxs = sequence[:batch_size]

        # remove chosen samples from sequence
        sequence = sequence[batch_size:]

        # return batched inputs and outputs
        if self.data_cuda is None:
            with self.lock:
                data = self.data[idxs]
        else:
            with self.lock:
                data = self.data_cuda[idxs]

        i, o = data.split((self.input_dim, self.output_dim), dim=1)
        return  i, o, sequence


class SampleGenerator(mp.Process):
    """
    Process for generating samples and writing them into a SampleCache instance

    #### Arguments
    - generator: 
    - cache:
    - target_len:
    - queued_cache_access (bool): If ``True`` data is not directly written into the cache. It is enqueued instead to avoid further torch operations potentially causing the generation of a CUDA context. Make sure to process the queue in some other process.
    """
    def __init__(self, generator: Iterable[Tuple[torch.Tensor, torch.Tensor]], cache: SampleCache, target_len: int = -1, buffer_size: int=1000, log_queue = None):
        super().__init__()
            
        self.generator = generator
        self.cache = cache
        self.target_len = target_len
        self.buffer_size = buffer_size

        self.log_queue = log_queue
        
    def run(self):
        try:
            # change random seed to make sure they are different for every process
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

            if self.log_queue is not None:
                logger = logging.getLogger('')
                logger.addHandler(logging.handlers.QueueHandler(self.log_queue))
                logger.setLevel(logging.INFO)

            logger = logging.getLogger('')
            logger.info('Process {} starts with target {} (PID={})'.format(self.name, self.target_len, os.getpid()))

            # initiate buffers
            buffer_size = self.buffer_size
            buffered_count = 0
            sample_input, sample_output = next(self.generator)
            input_batch = torch.zeros(sample_input.size()).expand(buffer_size, -1).clone()
            output_batch = torch.zeros(sample_output.size()).expand(buffer_size, -1).clone()

            while self.target_len < 0 or len(self.cache) < self.target_len:
                # generate sample
                sample_input, sample_output = next(self.generator)
                if torch.isnan(sample_input).any() or torch.isnan(sample_output).any():
                    logger.warning('Received invalid sample containing NaN. Ignoring sample.')

                # buffer it
                input_batch[buffered_count] = sample_input.squeeze(0)
                output_batch[buffered_count] = sample_output.squeeze(0)
                buffered_count += 1

                if buffered_count == buffer_size:
                    # buffer is full, pass it to cache
                    self.cache.put_batch(input_batch, output_batch)
                    input_batch = torch.zeros(input_batch.size())
                    output_batch = torch.zeros(output_batch.size())
                    buffered_count = 0

            logger.info('Process %s exits at %d cached samples'%(self.name, len(self.cache)))
        except KeyboardInterrupt:
            logger.info('Received keyboard interrupt. Closing process {}.'.format(self.name))