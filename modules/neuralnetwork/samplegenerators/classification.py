import logging
import torch
import numpy as np

from ...simulation.simulationmodel import SimulationModel

class ClassificationSampleGenerator:
    """Generator (iterator) for creating training and evaluation samples based on a ``SimulationModel`` instance

    #### Samples
    - Input: load profile (fragment) of length ``schedule_length``
    - Output: feasibility rating(s) in [0,1] XOR step of infeasibility

    #### Arguments
    - sampling_parameters: ``infeasibility_chance``, ``secondary_infeasibility_chance``, ``boundary_feasible_rate``, ``boundary_infeasible_rate``
    - stepwise_classification ``bool``: If ``True`` either individual ratings (``ouput_feasibility_step=False``) or the step of infeasibility is returned (``ouput_feasibility_step=True``). If ``False`` the output is the overall feasibility. 
    - output_infeasibility_step ``bool``: If ``True`` the sample output is the number of the step the schedule becomes infeasible. The first step is 0. If the schedule is feasible the value equals ``schedule_length``.
    
    """
    def __init__(self, 
                model: SimulationModel, 
                schedule_length: int,
                normalization: {str: (np.ndarray, np.ndarray)}={}, 
                sampling_parameters: {}={},
                stepwise_classification: bool=False,
                output_infeasibility_step: bool=False):
        self.model = model
        self.sampling_parameters = sampling_parameters

        if 'infeasibility_chance' in sampling_parameters:
            logger = logging.getLogger('')
            if sampling_parameters['infeasibility_chance'] < 0:
                logger.warn('Uniform sampling of actions may generate too few feasible schedules')
            elif sampling_parameters['infeasibility_chance'] == 0:
                logger.warn('Generator will only generate feasible load schedules')
        if 'secondary_infeasibility_chance' in sampling_parameters:
            if (1 - sampling_parameters['secondary_infeasibility_chance'])**schedule_length < 0.1:
                logger.warn('Generator will almost exclusively generate infeasible schedules')
        
        self.schedule_length = schedule_length
        self.stepwise_classification = stepwise_classification
        self.output_infeasibility_step = output_infeasibility_step

        state_dimension = len(model.sample_state(**self.sampling_parameters))
        self.input_state_normalization = normalization.get('input_state', (np.zeros(state_dimension), np.ones(state_dimension)))
        self.action_normalization = normalization.get('action', (np.zeros(1), np.ones(1)))

    @property
    def meta_data(self):
        return {
            'model': self.model,
            'dt': self.model.dt,
            'sampling_parameters': self.sampling_parameters,

            'schedule_length': self.schedule_length,

            'normalization': {
                'input_state': self.input_state_normalization,
                'action': self.action_normalization
            }            
        }
    
    def __iter__(self):
        return self

    def find_boundary_actions(self, all_actions, selectable_actions):
        """ 
        Helper function to find all elements from ``selectable_actions`` which are next to a non-selectable action from ``all_actions``
        """
        boundary_actions = []
        for i in range(all_actions.shape[0]):
            if all_actions[i] not in selectable_actions:
                # element is not in the set which is filtered -> next element
                continue

            # the element at position i is in selectable_actions
            if i+1<all_actions.shape[0] and all_actions[i+1] not in selectable_actions:
                # but the element at i+1 is not -> boundary
                boundary_actions.append(all_actions[i])
            elif i>0 and all_actions[i-1] not in selectable_actions:
                # but the previous element at i-1 is not -> boundary
                boundary_actions.append(all_actions[i])
        return np.array(boundary_actions)

    def __next__(self):
        
        model = self.model     
        all_actions = model.actions
        
        state = model.sample_state(**self.sampling_parameters) # get the state for index idx
        sample_input = torch.Tensor((state + self.input_state_normalization[0]) * self.input_state_normalization[1]) # copy

        schedule = torch.zeros(self.schedule_length)
        
        infeasibility_chance = self.sampling_parameters.get('infeasibility_chance', 0.5)
        boundary_feasible_rate = self.sampling_parameters.get('boundary_feasible_rate', 0.03)
        boundary_infeasible_rate = self.sampling_parameters.get('boundary_infeasible_rate', 0.1)
        impossible_action_rate = self.sampling_parameters.get('impossible_action_rate', 0.01)
        if np.random.random() < infeasibility_chance:
            # make it infeasible (not guaranteed but likely)
            
            # for schedules of length 96
            # 3% chance means: ~5% with 0, ~16% with 1, ~79% with > 1 infeasible actions
            # 10% chance means: (0 extremely unlikely), ~50% < 10, 36% > 10 
            infeasibility_chance = self.sampling_parameters.get('secondary_infeasibility_chance', np.random.choice([0.03] * 9 + [0.1])) 
        else:
            # make it feasible
            infeasibility_chance = 0

        force_infeasible = False
        infeasibility_step = self.schedule_length
        for step in range(self.schedule_length):
            feasible_actions =  model.feasible_actions # determine the feasible actions
            if force_infeasible or np.random.random() < infeasibility_chance:
                # choose infeasible if possible
                infeasible_actions = np.setdiff1d(all_actions, feasible_actions)
                if infeasible_actions.size == 0:
                    # only feasible actions
                    force_infeasible = True # try again next time step
                    action = np.random.choice(feasible_actions)
                else:
                    # there are infeasible options, choose one
                    force_infeasible = False

                    rnd = np.random.random()
                    if rnd < impossible_action_rate:
                        action = np.random.choice([max(all_actions), min(all_actions)]) * np.random.choice([-10, -5, -2, -1.5, 1.5, 2, 5, 10]) 
                    elif rnd < impossible_action_rate + boundary_infeasible_rate:
                        action = np.random.choice(self.find_boundary_actions(all_actions, infeasible_actions))
                    else:
                        action = np.random.choice(infeasible_actions)
            else:
                # choose a feasible action
                if feasible_actions.shape[0] < all_actions.shape[0] and np.random.random() < boundary_feasible_rate:
                    # not all actions are feasible, select one on the boundary
                    action = np.random.choice(self.find_boundary_actions(all_actions, feasible_actions))
                else:
                    # select any feasible action
                    action = np.random.choice(feasible_actions)                    

            new_state, interaction = model.transition(action) # determine new state
            schedule[step] = (torch.Tensor([action]) + self.action_normalization[0]) * self.action_normalization[1] # append to schedule
            
            if step < infeasibility_step and not (action in feasible_actions): # determine feasibility
                infeasibility_step = step
        
        sample_input = torch.cat((sample_input, schedule))

        if self.stepwise_classification:
            if self.output_infeasibility_step:
                # output the step of infeasibility (feasible if step==schedule_length)
                sample_output = torch.Tensor([infeasibility_step])
            else:
                # output feasibility for every step
                
                feasibility = torch.zeros(self.schedule_length)
                feasibility[:infeasibility_step] = 1
                sample_output = feasibility
        else:
            # output overall feasibility
            sample_output = torch.ones(1) * (infeasibility_step == self.schedule_length)
            
        return sample_input, sample_output