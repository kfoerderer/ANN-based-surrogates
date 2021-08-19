"""

Base classes for implementing models of systems

"""

from ..model import Model
from typing import List, Tuple

import numpy as np

class SimulationModel(Model):
    """
    Base class for all systems.

    Each ``system``, e.g., a battery, chp plant or an aggregate of systems, periodically performs ``actions``, e.g., idling, charging with a given power or discharging.
    The subsequent ``state`` is determined based on the current state and performed action.

    Sign of energy flow is as follows:
        + system consumes power
        - system releases power
        
        input (+) -> system -> output (-)
    """

    def __init__(self, dt: int, actions: np.ndarray, correct_infeasible = False, **kwargs):
        """
        Arguments
        - dt ``int``: length of a time step in seconds
        - actions ``np.ndarray``: array holding all possible actions
        - correct_infeasible ``bool``: If an infeasible action is passed to the transition it is corrected to the closes feasible action, when this argument is ``True``.
        """
        super().__init__(dt, actions, correct_infeasible, **kwargs)
        self._training = False # training mode

    def train(self, mode: bool=True):
        """
        Training mode may be used to prohibit certain actions in the learned model. 
        This can be used to learn more restrictive constraints without having to deal with multiple different model instances.

        The set of feasible actions in training mode (self.training=True) must be a subset of the set of feasible actions in evaluation mode (self.training=False).

        Arguments
        - mode ``bool``: specifies whether the model is in training mode (True) or not (False)
        """
        self._training = mode
        self._feasible_actions = None

    def eval(self):
        """
        Activates evaluation mode, i.e., sets training to False.
        """
        self.train(False)
        
    def sample_state(self, **kwargs) -> np.ndarray:
        """
        [Optional] May be used to provide samples for system states. 
        
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.
        """
        # 1. sample a state
        print('System.sample_state() is not yet implemented')
        # 2. update the state to the newly sampled state
        self.state = None 
        # 3. update feasible actions
        self._feasible_actions = None # 'only' mark as outdated to avoid unneccesary (potentially costly) computations
        # 4. return the newly sampled state
        return self.state

    def sample_action(self, **kwargs) -> Tuple[float, np.ndarray]:
        """
        ### Notes
        - [Optional] May be used to provide samples for actions to perform.

        ### Standard implementation 
        Separates feasible and infeasible actions, 
        chooses one of either sets and then selects a random action.

        Arguments
        - infeasibility_chance ``float=0.5``: chance for drawing from the infeasible actions. Pass ``-1`` to randomly draw any action, feasible or not.

        Returns 
        - ``(float, np.ndarray)``: the sampled action and the set of feasible actions as tuple.
        """
        infeasibility_chance = kwargs.get('infeasibility_chance', 0.5)

        feasible_actions = self.feasible_actions
        if self.correct_infeasible == False or len(feasible_actions) == len(self.actions):
            # if only feasible actions can be processed OR
            # all actions are feasible, return a random feasible action
            return np.random.choice(feasible_actions), feasible_actions

        if infeasibility_chance < 0:
            # sample from all actions
            return np.random.choice(self.actions), feasible_actions

        if np.random.random() < infeasibility_chance:
            # infeasible actions do exist, since not all actions are feasible
            infeasible_actions = np.setdiff1d(self.actions, feasible_actions) 
            return np.random.choice(infeasible_actions), feasible_actions

        else:
            return np.random.choice(feasible_actions), feasible_actions

    def forecast(self, time_step_count: int=1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a forecast consisting of the current state and the states of the subsequent ``time_step_count-1`` time steps.
        This forecast should be stored in the hidden state and determine the future behaviour, guaranteeing the same result for each repeated run and for at least ``time_step_count`` time steps.
        Previous forecasts should be taken into account when determining future states or creating new forecasts.

        Arguments        
        - time_steps_count ``int``: number of time steps to forecast

        Returns
        - forecast ``np.ndarray``
        - injection_mask ``np.ndarray(dtype=bool)``
        """
        return np.repeat(self.state.reshape(1,-1), time_step_count, axis=0), np.zeros((time_step_count, self.state.shape[0]), dtype=bool) # best guess



