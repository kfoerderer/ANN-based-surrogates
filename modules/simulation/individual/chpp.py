from modules.simulation.simulationmodel import SimulationModel
from typing import List, Tuple

import numpy as np

class CHPP(SimulationModel):
    """
    Combined Heat and Power Plant.

    State
    - mode: index of mode (0 to n)
    - dwell_time: dwell time in current mode in seconds
    - min_dwell_time_off: minimum dwell time in seconds
    - min_dwell_time_on: minimum dwell time in seconds

    Arguments
    - dt ``int``: length of a time step in seconds
    - actions ``np.ndarray``: array holding all possible actions
    - state_matrix ``[[(int,int)]]``: matrix defining the average power (el., th.) for each mode k (states[k][k]) and transition from j to k (states[j][k]).
        Mode 0 is assumed to be the 'off'-state
    """

    def __init__(self, dt: int, actions: np.ndarray, state_matrix: List[List[Tuple[int,int]]], correct_infeasible=False):
        super().__init__(dt, actions, correct_infeasible)

        # find closest actions for the entries of state_matrix
        sanitized_matrix = []
        for row in state_matrix:
            sanitized_row = []
            for element in row:
                sanitized_row.append((actions[np.argmin(np.abs(element[0]-actions))], element[1]))
            sanitized_matrix.append(sanitized_row)
                
        self.state_matrix = sanitized_matrix

        self.ignore_dwell_time_on = False
        self.ignore_dwell_time_off = False

        # initialize state array
        # 1. mode = current action
        # 2. dwell time
        # 3. min off
        # 4. min on
        self.state: np.ndarray = np.zeros(4, dtype=int)

    def __repr__(self):
        return ('CHPP(state={}, state_matrix={}, correct_infeasible={})').format(
            self.state, self.state_matrix, self.correct_infeasible)

    @property
    def mode(self) -> int:
        return np.rint(self.state[0]).astype(int)

    @mode.setter
    def mode(self, v: int):
        self.state[0] = v
        self._feasible_actions = None

    @property
    def dwell_time(self) -> int:
        return self.state[1]

    @dwell_time.setter
    def dwell_time(self, v: int):
        self.state[1] = v
        self._feasible_actions = None

    @property
    def min_off_time(self) -> int:
        return self.state[2]

    @min_off_time.setter
    def min_off_time(self, v: int):
        self.state[2] = v
        self._feasible_actions = None

    @property
    def min_on_time(self) -> int:
        return self.state[3]

    @min_on_time.setter
    def min_on_time(self, v: int):
        self.state[3] = v
        self._feasible_actions = None

    def determine_feasible_actions(self) -> np.ndarray:
        if (self.ignore_dwell_time_on and self.mode != 0) or (self.ignore_dwell_time_off and self.mode == 0):
            actions =  np.array([_[0] for _ in self.state_matrix[self.mode]])
        else:
            # mode may only be changed if min_dwell_time has passed
            if (self.mode == 0) and self.dwell_time < self.min_off_time:
                # off, but not long enough
                actions = np.array([self.state_matrix[0][0][0]]) # el. power for staying in mode zero
            elif (self.mode != 0) and self.dwell_time < self.min_on_time:
                # on, but not long enough
                actions = []
                for i, new_state in enumerate(self.state_matrix[self.mode]):
                    if i > 0: # mode 0 (='off') is not allowed
                        actions.append(new_state[0])
                actions = np.array(actions)
            else:
                # sufficiently long off or on
                actions = []
                for new_state in self.state_matrix[self.mode]:
                    actions.append(new_state[0])
                actions =  np.array(actions)

        self._feasible_actions  = actions
        return actions

    def sample_state(self, min_off_times=[0,900,1800,2700,3600], min_on_times=[0,900,1800,2700,3600], dwell_times=[], dwell_time_distribution: Tuple[List[Tuple[float,float]],List[float]]=([(0,24*60*60)],[1]), **kwargs) -> np.ndarray:
        """
        [Optional] May be used to provide a standard sampling algorithm for system states. 
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.

        Arguments
        - min_off_times
        - min_on_times
        - dwell_times ``[int]`` Array of possible dwell times.
        - dwell_time_distribution ``{float:(float, float)}`` Dictionary, mapping the cummulative probability to intervals [starting_point, end_point). Only used if ``dwell_times`` is empty.
        """
        
        # 1. sample a state & 2. update the state to the newly sampled state
        self.mode = np.random.choice(len(self.state_matrix))
        
        if len(dwell_times) > 0:
            # draw from given values
            self.dwell_time = np.random.choice(dwell_times)
        else:
            # use distribution
            (lower_bound, upper_bound) = dwell_time_distribution[0][np.random.choice(len(dwell_time_distribution[0]), 1, p=dwell_time_distribution[1])[0]]
            self.dwell_time = int(np.random.uniform(lower_bound, upper_bound))

        self.min_off_time = np.random.choice(min_off_times)
        self.min_on_time = np.random.choice(min_on_times)

        # 3. update feasible actions
        self._feasible_actions = None

        # 4. return the newly sampled state
        return self.state

    def transition(self, action, interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the systems state according to the provided parameters and returns the new state.
        
        Arguments
        - action: action to take. May be ``None`` for passive systems.
        - interaction: possible interactions
        """
        # for faster access
        dt = self._dt
        state_matrix = self.state_matrix

        action = self.correct_action(action)
        self.ignore_dwell_time_on = False
        self.ignore_dwell_time_off = False
        
        current_mode = self.mode
        # determine which mode is following
        new_mode = -1
        for i, new_state in enumerate(state_matrix[current_mode]):
            if new_state[0] == action:
                # found the relevant entry
                new_mode = i
                break
        if new_mode < 0:
            raise ValueError('Unable to determine the next mode. (Current mode: {}, action: {})'.format(current_mode, action))

        el_power = state_matrix[current_mode][new_mode][0]
        th_power = state_matrix[current_mode][new_mode][1]
        
        # update state
        if (self.mode == 0 and new_mode > 0) or (self.mode > 0 and new_mode == 0):
            self.dwell_time = 0
        self.mode = new_mode
        self.dwell_time += dt

        # update feasible actions
        self._feasible_actions = None
        
        return self.state, interaction - np.array([el_power, th_power])