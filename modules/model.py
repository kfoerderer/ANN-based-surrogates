"""

Base classes for implementing models of systems

"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import copy

class Model(ABC):
    """
    Base class for all system models.

    Each ``system``, e.g., a battery, chp plant or an aggregate of systems, periodically performs ``actions``, e.g., idling, charging with a given power or discharging.
    The subsequent ``state`` is determined based on the current state and performed action.

    Sign of energy flow is as follows:
        + system consumes power
        - system releases power
        
        input (+) -> system -> output (-)

    """

    @staticmethod
    def create_action_set(min_power: int, max_power: int, n_actions: int) -> np.ndarray:
        """
        ### Parameters
        min_power ``int`` min power in W
        
        max_power ``int`` max power in W

        n_actions ``int`` number of actions ranging from min to max
        """
        return np.array(list({min_power + (max_power-min_power)*i/(n_actions-1) for i in range(n_actions)}))

    def __init__(self, dt: int, actions: np.ndarray, correct_infeasible = False, **kwargs):
        """
        ### Parameters
        dt ``int`` length of a time step in seconds

        actions ``np.ndarray`` array holding all possible actions

        correct_infeasible ``bool`` If an infeasible action is passed to the transition it is corrected to the closes feasible action, when this argument is ``True``.
        """
        self._dt = dt
        self._actions: np.ndarray = actions
        # make sure all actions are ordered which is not guaranteed due to using a dictionary
        if actions is not None:
            self._actions.sort() 

        self.state: np.ndarray = None  # use a np.ndarray to store all data in a single array. Acces this using properties
        self.hidden_state: np.ndarray = None # use a np.ndarray to store all data in a single array. Acces this using properties

        self._pushed_state: np.ndarray = None
        self._pushed_hidden_state: np.ndarray = None
        self._pushed_feasible_actions: np.ndarray = None

        self._feasible_actions: np.ndarray = None
        self.correct_infeasible = correct_infeasible
        self._infeasibility_error_message = 'Action is infeasible while correct_infeasible is False'

    def push_state(self):
        """
        Stores the current state for restoring it later
        """
        self._pushed_state = np.copy(self.state)
        self._pushed_hidden_state = copy.deepcopy(self.hidden_state)
        if self._feasible_actions is None:
            self._pushed_feasible_actions = None
        else:
            self._pushed_feasible_actions = np.copy(self._feasible_actions)

    def pop_state(self):
        """
        Restores the previously stored state
        """
        self.state = np.copy(self._pushed_state)
        self.hidden_state = copy.deepcopy(self._pushed_hidden_state)
        if self._pushed_feasible_actions is None:
            self._feasible_actions = None
        else:
            self._feasible_actions = np.copy(self._pushed_feasible_actions)

    def load_state(self, state: np.ndarray, hidden_state: np.ndarray = None, feasible_actions: np.ndarray = None):
        self.state = np.copy(state)
        self.hidden_state = copy.deepcopy(hidden_state)
        self._feasible_actions = feasible_actions # None triggers recomputation once the information needed

    @property
    def actions(self) -> np.ndarray:
        """
        Returns the set of all actions which is a super set of the set of feasible actions.
        """
        return self._actions

    @property
    def dt(self) -> int:
        return self._dt

    @property
    def feasible_actions(self) -> List[int]:
        """
        Returns a vector holding all feasible actions (neglecting possible interactions with other systems).

        Note: Use np.isin(actions, feasible_actions) to get a binary encoding
        """
        if self._feasible_actions is None:
            return self.determine_feasible_actions()
        return self._feasible_actions

    @abstractmethod
    def determine_feasible_actions(self) -> np.ndarray:
        """
        Computes the vector holding all feasible actions (neglecting possible interactions with other systems).

        The result is stored in self._feasible_actions and then returned.
        """
        self._feasible_actions = None
        return None

    @abstractmethod
    def transition(self, action, interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the systems state according to the provided parameters and returns the new state.
        Parameters:
        - action: action to take. May be ``None`` for passive systems.
        - interaction: interaction with other systems in W (el, th)
        """
        # 1. check feasibility
        # use self.correct_action()
        # 2. determine resulting interaction (for a single DER this usually is the (repaired) action)
        resulting_interaction = interaction
        # 3. update state based on action and original interaction
        #pass
        # 4. determine feasible actions
        self._feasible_actions = None # do not directly call the method to avoid unnecessary (potentially costly) calculations
        # 5. return the arrays
        return self.state, resulting_interaction

    def correct_action(self, action: float) -> float:
        """
        Returns a feasible action corresponding to the given (infeasible) action.

        If infeasible returns the closest feasible action (in terms of power).
        """
        feasible_actions = self.feasible_actions

        if action not in feasible_actions:
            # infeasible action
            if not self.correct_infeasible:
                raise ValueError(self._infeasibility_error_message)

            # correct the given action
            action = feasible_actions[np.abs(feasible_actions - action).argmin()]

        return action

    def interaction(self, action, interaction: np.ndarray=np.zeros(2)) -> np.ndarray:
        """
        Provides further information about the interaction of the system and its environment when performing the given action.
        """
        # backup state data
        state = np.copy(self.state)
        hidden_state = copy.deepcopy(self.hidden_state)
        # do a transition to determine the resulting interaction
        new_state, interaction = self.transition(action, interaction)
        # restore the previous state
        self.state = state
        self.hidden_state = hidden_state
        return interaction