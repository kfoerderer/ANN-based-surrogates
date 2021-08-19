import torch

import numpy as np

from ...simulation.simulationmodel import SimulationModel

class BatchProcessor:
    """
    Pre- and post-processes the data to be given to and returned by an ANN.
    
    The processor is set up by calling the intended transformation functions in the correct order.
    In order to transform a vector or batch of data it is passed to the processor as an argument.
    """
    def __init__(self):
        self.transformations = [] # [(type, slice, post_processor, *args)]
        self.idx = 0
        self.split = []

    def __str__(self):
        return 'BatchProcessor({})'.format(self.transformations)

    def __repr__(self):
        return str(self)

    def none(self, element_count=1):
        """
        No transformation is applied
        
        elements: n to n
        """
        post_processor = BatchProcessor()
        self.transformations.append(('1', slice(self.idx, self.idx+element_count), post_processor))
        self.idx += element_count
        return post_processor

    def normalize(self, bias: [], scale: []):
        """
        Normalize one or multiple consecutive elements in the vector
        
        elements: n to n
        """
        assert len(bias) == len(scale)
        post_processor = BatchProcessor()
        self.transformations.append(('nmlz', slice(self.idx, self.idx+len(bias)), post_processor, torch.Tensor(bias), torch.Tensor(scale)))
        self.idx += len(bias)
        return post_processor

    def denormalize(self, bias: [], scale: []):
        """
        Undo the normalization of one or multiple consecutive elements in the vector
        elements: n to n
        """
        assert len(bias) == len(scale)
        post_processor = BatchProcessor()
        self.transformations.append(('dnml', slice(self.idx, self.idx+len(bias)), post_processor, torch.Tensor(bias), torch.Tensor(scale)))
        self.idx += len(bias)
        return post_processor

    def one_hot(self, alternatives: []):
        """
        Encode a single element with a one hot encoding
        
        elements: 1 to n
        """
        post_processor = BatchProcessor()
        self.transformations.append(('oneh', slice(self.idx, self.idx+1), post_processor, torch.Tensor(alternatives)))
        self.idx += 1
        return post_processor

    def mode_of_distribution(self, alternatives: []):
        """
        Determine the mode of an estimated distribution 
        
        elements: n to 1
        """
        post_processor = BatchProcessor()
        self.transformations.append(('mode', slice(self.idx, self.idx+len(alternatives)), post_processor, torch.Tensor(alternatives)))
        self.idx += len(alternatives)
        return post_processor

    def discretize(self, discrete_values: []):
        """
        Replace a value with the closest values in ``discrete_values``
        """
        post_processor = BatchProcessor()
        self.transformations.append(('disc', slice(self.idx, self.idx+1), post_processor, torch.Tensor(discrete_values)))
        self.idx += 1
        return post_processor

    def discretize_index(self, discrete_values: []):
        """
        Replace the value with the index of the closest discrete value given
        
        elements: 1 to 1
        """
        post_processor = BatchProcessor()
        self.transformations.append(('didx', slice(self.idx, self.idx+1), post_processor, torch.Tensor(discrete_values)))
        self.idx += 1
        return post_processor

    def sigmoid(self, element_count):
        """
        Apply sigmoid for the next ``element_count`` elements

        elements: n to n
        """
        post_processor = BatchProcessor()
        self.transformations.append(('sgmd', slice(self.idx, self.idx+element_count), post_processor))
        self.idx += element_count
        return post_processor

    def clip(self, min_value=-float('inf'), max_value=float('inf'), element_count: int=1):
        """
        Clips the values outside the given interval by moving them to the nearest border

        elements: n to n
        """
        post_processor = BatchProcessor()
        self.transformations.append(('clip', slice(self.idx, self.idx+element_count), post_processor, min_value, max_value))
        self.idx += element_count
        return post_processor

    def invert(self, element_count: int=1):
        """
        Inverts individual elements, i.e., transforms x to x^-1
        """
        post_processor = BatchProcessor()
        self.transformations.append(('x^-1', slice(self.idx, self.idx+element_count), post_processor))
        self.idx += element_count
        return post_processor


    def custom(self, func: callable, element_count: int=1):
        """
        Use a custom transformation ``func(batch: torch.Tensor)`` for ``element_count`` elements.
        
        elements: n to m
        """
        post_processor = BatchProcessor()
        self.transformations.append(('func', slice(self.idx, self.idx+element_count), post_processor, func))
        self.idx += element_count
        return post_processor

    def split_result(self, idxs: []):
        """
        Call this method to split the resulting batch in two or more pieces, separating at each index of ``ìdxs``. 
        """
        if len(idxs) == 0:
            # reset
            self.split = []
        # prepare slices
        idx = 0
        slices = []
        for entry in idxs:
            slices.append(slice(idx, idx+entry))
            idx += entry
        slices.append(slice(idx, None))
        self.split = slices

    def __call__(self, tensor: torch.Tensor):
        # if it is only a vector, make it a batch with a single simple
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        result = []
        for transformation in self.transformations:
            post_processor = None
            # none
            if transformation[0] == '1':
                _, elements, post_processor = transformation
                result.append(tensor[:,elements])

            # normalize
            elif transformation[0] == 'nmlz':
                _, elements, post_processor, bias, scale = transformation
                result.append((tensor[:,elements] - bias) / scale)

            # de-normalize
            elif transformation[0] == 'dnml':
                _, elements, post_processor, bias, scale = transformation
                result.append((tensor[:,elements] * scale) + bias)
            
            # one hot
            elif transformation[0] == 'oneh':
                _, elements, post_processor, alternatives = transformation
                result.append((tensor[:,elements] == alternatives)*1.)

            # mode of a distribution
            elif transformation[0] == 'mode':
                _, elements, post_processor, alternatives = transformation
                result.append(alternatives[torch.argmax(tensor[:,elements], dim=1)].unsqueeze(1))

            # discrete
            elif transformation[0] == 'disc':
                _, elements, post_processor, values = transformation
                result.append(values[torch.argmin(torch.abs(values-tensor[:,elements]), dim=1)].unsqueeze(1).float())

            # discrete index
            elif transformation[0] == 'didx':
                _, elements, post_processor, values = transformation
                result.append(torch.argmin(torch.abs(values-tensor[:,elements]), dim=1).unsqueeze(1).float())

            # sigmoid function
            elif transformation[0] == 'sgmd':
                _, elements, post_processor, = transformation
                result.append(torch.sigmoid(tensor[:,elements]))

            # clip values
            elif transformation[0] == 'clip':
                _, elements, post_processor, min_value, max_value = transformation
                result.append(torch.clamp(tensor[:,elements], min_value, max_value))

            # invert elements
            elif transformation[0] == 'x^-1':
                _, elements, post_processor = transformation
                result.append(1 / tensor[:,elements])

            # custom transformation (callable)
            elif transformation[0] == 'func':
                _, elements, post_processor, func = transformation
                result.append(func(tensor[:,elements]))

            else:
                raise ValueError('Transformation {} unknown'.format(transformation[0]))

            if post_processor is not None and len(post_processor.transformations) > 0:
                result[-1] = post_processor(result[-1])

        if len(self.split) == 0:
            return torch.cat(result, dim=1)
        
        result = torch.cat(result, dim=1)
        return (*[result[:,elements] for elements in self.split],)

class SampleGenerator:
    """
    Generator (iterator) for creating training and evaluation samples from a ``SimulationModel`` instance.

    #### Samples
    - Input: state (, action)
    - Output: (new state), (action feasibility), (interaction)
    """
    def __init__(self, 
                model: SimulationModel, 
                input_processor: BatchProcessor=None,
                output_processor: BatchProcessor=None, 
                sampling_parameters: ()=(),
                output_action_feasibility: bool=True,
                output_new_state: bool=True,
                output_interaction: bool=True,
                ann_output_processor=None):
        self.model = model
        self.sampling_parameters = sampling_parameters

        self.input_processor = input_processor
        self.output_processor = output_processor

        self.output_action_feasibility = output_action_feasibility
        self.output_new_state = output_new_state
        self.output_interaction = output_interaction

        # not needed for generating samples, but for later ANN use
        # -> add it to the meta data for simplifyed handling
        self.ann_output_processor = ann_output_processor 

    @property
    def meta_data(self):
        return {
            'model': self.model,
            'dt': self.model.dt,
            'sampling_parameters': self.sampling_parameters,

            'input_processor': self.input_processor,
            'output_processor': self.output_processor,

            'output_action_feasibility': self.output_action_feasibility,
            'output_new_state': self.output_new_state,
            'output_interaction': self.output_interaction,      

            'ann_output_processor': self.ann_output_processor,
        }
    
    def __iter__(self):
        return self

    def __next__(self):
        model = self.model     
        all_actions = model.actions
        
        # input
        sample_input = torch.Tensor(model.sample_state(**self.sampling_parameters)) # torch.Tensor(.) generates a copy [!]

        if self.output_new_state or self.output_interaction:
            action, feasible_actions = model.sample_action(**self.sampling_parameters)
            new_state, interaction = model.transition(action) # determine new state
            sample_input = torch.cat((sample_input, torch.Tensor([action])))

        elif self.output_action_feasibility:  
            # [else if] since it is also determined in the previous case
            feasible_actions =  model.feasible_actions

        if self.input_processor is not None:
            sample_input = self.input_processor(sample_input)

        # output
        if self.output_new_state:
            sample_output = torch.Tensor(new_state)
        else:
            sample_output = torch.Tensor([])

        if self.output_interaction:
            sample_output = torch.cat((sample_output, torch.Tensor(interaction)))

        if self.output_action_feasibility:
            # use binary representations.
            # to retrieve the actual actions do:
            # model.actions[(feasible_actions.data.numpy() == 1)]
            feasible_actions = np.isin(all_actions, feasible_actions)*1
            sample_output = torch.cat((sample_output, torch.Tensor(feasible_actions).unsqueeze(0)))

        if self.output_processor is not None:
            sample_output = self.output_processor(sample_output)

        return sample_input, sample_output