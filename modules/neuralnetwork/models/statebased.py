from typing import List, Tuple
from ...model import Model

import numpy as np

import torch
import torch.nn as nn

class NeuralModel(Model):
    """
    Model that has been learned as ANN.

    Note: especially for a combined model (state+action) it could be beneficial to compute a whole batch of actions at once
    """
    def __init__(self, dt: int, actions: np.ndarray, state_model: nn.Module, state_input_processor=None, state_output_processor=None, 
                classifier=None, classifier_input_processor=None, classifier_output_processor=None, 
                correct_infeasible = False, feasibility_threshold=0.5,
                **kwargs):
        """
        #### Arguments
        - dt ``int``: time step length the model has been trained with
        - actions ``np.ndarray``: array holding all possible actions
        - model ``torch.nn.Module``: the neural model itself
        - input_processor ``Callable(state, action, actions) -> torch.Tensor``: prepares the ann input
        - output_processor ``Callable(output) -> state, interaction, ratings``: interprets the ann output
        - classifier, either
            - ``torch.nn.Module``: if it is a classifier model or ``Callable(ratings) -> feasibility`` if the feasibility ratings are embedded into the state
            - ``callable(NeuralModel)``: a function determining ratings
        - correct_infeasible ``bool``: If an infeasible action is passed to the transition it is corrected to the closes feasible action, when this argument is ``True``.
        """
        super().__init__(dt, actions, correct_infeasible, **kwargs)

        self.state_model = state_model
        self.state_model.eval()
        self.state_input_processor = state_input_processor
        self.state_output_processor = state_output_processor

        self.classifier = classifier
        if isinstance(classifier, nn.Module):
            self.classifier.eval()
        self.classifier_input_processor = classifier_input_processor
        self.classifier_output_processor = classifier_output_processor
        self.feasibility_threshold = feasibility_threshold

        self.state = None
        self.ratings = None # holds the action ratings (potentially required for handling situations where all actions are infeasible)

    def determine_feasible_actions(self) -> np.ndarray:
        classifier = self.classifier

        if isinstance(classifier, nn.Module):
            # it is an ANN
            with torch.no_grad():
                if self.classifier_input_processor is None:
                    ann_input = torch.Tensor(self.state)
                else:
                    ann_input = self.classifier_input_processor(torch.Tensor(self.state))
                
                    ann_output = classifier(ann_input)
                if self.classifier_output_processor is not None:
                    ann_output = self.classifier_output_processor(ann_output)

                self.ratings = ann_output
                self._feasible_actions = self._actions[(ann_output > self.feasibility_threshold).reshape(-1)]
        
        elif callable(classifier):
            self.ratings, self._feasible_actions = classifier(self, self.feasibility_threshold)
        
        else: 
            raise ValueError('No classifier has been specified')
        return self._feasible_actions
        
    def transition(self, action, interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            ann_input = torch.cat([torch.Tensor(self.state), torch.Tensor([action])])
            if self.state_input_processor is not None:
                ann_input = self.state_input_processor(ann_input)
            
            ann_output = self.state_model(ann_input)
            if self.state_output_processor is not None:
                ann_output = self.state_output_processor(ann_output)

        if len(ann_output) == 1:
            state = ann_output[0]
            interaction, ratings = torch.Tensor([]), torch.Tensor([])
        elif len(ann_output) == 2:
            state, interaction = ann_output
            ratings = torch.Tensor([])
        elif len(ann_output) == 3:
            state, interaction, ratings = ann_output
        # remove extra dimension
        self.state = state.reshape(-1).numpy()
        self.ratings = ratings.reshape(-1).numpy()
        interaction = interaction.reshape(-1).numpy()
        
        self._feasible_actions = None # reset set of feasible actions
        return self.state, interaction

    def batch_transition(self, actions: List[int], interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the result of a transition for a whole batch of actions
        """
        with torch.no_grad():
            actions = torch.Tensor(actions)
            ann_input = torch.cat([torch.Tensor(self.state).repeat(actions.size(0), 1), actions.unsqueeze(1)], axis=1)
            if self.state_input_processor is not None:
                ann_input = self.state_input_processor(ann_input)
            
            ann_output = self.state_model(ann_input)
            if self.state_output_processor is not None:
                ann_output = self.state_output_processor(ann_output)

        if len(ann_output) == 1:
            state = ann_output[0]
            interaction, ratings = torch.Tensor([]), torch.Tensor([])
        elif len(ann_output) == 2:
            state, interaction = ann_output
            ratings = torch.Tensor([])
        elif len(ann_output) == 3:
            state, interaction, ratings = ann_output
        
        return state, interaction, ratings


class BatchedNeuralModel(Model):
    """
    Model that has been learned as ANN.

    Use this class to process an enitre ensemble, where each system has the identical ANN model.
    """
    def __init__(self, dt: int, actions: np.ndarray, state_model: nn.Module, state_input_processor=None, state_output_processor=None, 
                classifier=None, classifier_input_processor=None, classifier_output_processor=None, 
                correct_infeasible = False, feasibility_threshold=0.5,
                **kwargs):
        """
        #### Arguments
        - dt ``int``: time step length the model has been trained with
        - actions ``np.ndarray``: array holding all possible actions
        - model ``torch.nn.Module``: the neural model itself
        - input_processor ``Callable(state, action, actions) -> torch.Tensor``: prepares the ann input
        - output_processor ``Callable(output) -> state, interaction, ratings``: interprets the ann output
        - classifier, either
            - ``torch.nn.Module``: if it is a classifier model or ``Callable(ratings) -> feasibility`` if the feasibility ratings are embedded into the state
            - ``callable(NeuralModel)``: a function determining ratings
        - correct_infeasible ``bool``: If an infeasible action is passed to the transition it is corrected to the closes feasible action, when this argument is ``True``.
        """
        super().__init__(dt, actions, correct_infeasible, **kwargs)

        self.state_model = state_model
        self.state_model.eval()
        self.state_input_processor = state_input_processor
        self.state_output_processor = state_output_processor

        self.classifier = classifier
        if isinstance(classifier, nn.Module):
            self.classifier.eval()
        self.classifier_input_processor = classifier_input_processor
        self.classifier_output_processor = classifier_output_processor
        self.feasibility_threshold = feasibility_threshold

        self.state = None
        self.ratings = None # holds the action ratings (potentially required for handling situations where all actions are infeasible)

        self._cuda = False

    def cuda(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.cuda()
        self.state_model.cuda()
        self._cuda = True

    def cpu(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.cpu()
        self.state_model.cpu()
        self._cuda = False

    def determine_feasible_actions(self) -> np.ndarray:
        classifier = self.classifier
        _cuda = self._cuda

        if isinstance(classifier, nn.Module):
            # it is an ANN
            with torch.no_grad():
                if self.classifier_input_processor is None:
                    ann_input = torch.Tensor(self.state)
                else:
                    ann_input = self.classifier_input_processor(torch.Tensor(self.state))
                
                    if _cuda:
                        ann_input = ann_input.cuda()
                    ann_output = classifier(ann_input)
                    if _cuda:
                        ann_output = ann_output.cpu()
                if self.classifier_output_processor is not None:
                    ann_output = self.classifier_output_processor(ann_output)

                self.ratings = ann_output
                self._feasible_actions = np.array([self._actions[row] for row in (ann_output > self.feasibility_threshold)])
        
        elif callable(classifier):
            self.ratings, self._feasible_actions = classifier(self, self.feasibility_threshold)
        
        else: 
            raise ValueError('No classifier has been specified')
        return self._feasible_actions
        
    def transition(self, actions: List[int], interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        _cuda = self._cuda
        with torch.no_grad():
            ann_input = torch.cat([torch.Tensor(self.state), torch.Tensor(actions)], dim=1)
            if self.state_input_processor is not None:
                ann_input = self.state_input_processor(ann_input)
            
            if _cuda:
                ann_input = ann_input.cuda()
            ann_output = self.state_model(ann_input)
            if _cuda:
                ann_output = ann_output.cpu()
            if self.state_output_processor is not None:
                ann_output = self.state_output_processor(ann_output)

        if len(ann_output) == 1:
            state = ann_output[0]
            interaction, ratings = torch.Tensor([]), torch.Tensor([])
        elif len(ann_output) == 2:
            state, interaction = ann_output
            ratings = torch.Tensor([])
        elif len(ann_output) == 3:
            state, interaction, ratings = ann_output
        # remove extra dimension
        self.state = state.numpy()
        self.ratings = ratings.numpy()
        interaction = interaction.numpy()
        
        self._feasible_actions = None # reset set of feasible actions
        return self.state, interaction
    