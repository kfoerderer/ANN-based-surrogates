from modules.simulation.simulationmodel import SimulationModel
from typing import List, Tuple

import numpy as np

class BESS(SimulationModel):
    """
    Battery Energy Storage System.

    State
    - stored_energy ``int`` Ws

    Arguments
    - dt ``int`` length of a time step in seconds
    - actions ``np.ndarray`` array holding all possible actions
    - capacity ``int`` capacity of the battery in Ws
    - charging_efficiency ``float`` efficiency of charging
    - discharging_efficiency ``float`` efficiency of discharging
    - relative_loss ``float`` relative storage energy loss per hour
    """

    def __init__(self, dt: int, actions: np.ndarray, capacity: int, charging_efficiency: float, discharging_efficiency: float, relative_loss: float, 
                correct_infeasible: bool=False, constraint_fuzziness: float=0):
        super().__init__(dt, actions, correct_infeasible)

        self.capacity = capacity
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.relative_loss = relative_loss

        self.state: np.ndarray = np.array([0, 0., 1.])

        self.constraint_fuzziness = constraint_fuzziness

    def __repr__(self):
        return ('BESS(state={}, capacity={}, charging_efficiency={}, discharging_efficiency={}, '
                'relative_loss={}, correct_infeasible={}'
                ')').format(self.state, self.capacity, self.charging_efficiency, self.discharging_efficiency,
                            self.relative_loss, self.correct_infeasible)

    @property
    def stored_energy(self) -> int:
        return self.state[0]

    @stored_energy.setter
    def stored_energy(self, v: int):
        self.state[0] = v
        self._feasible_actions = None

    @property
    def soc_min(self) -> float:
        return self.state[1]

    @soc_min.setter
    def soc_min(self, v: float):
        self.state[1] = v
        self._feasible_actions = None

    @property
    def soc_max(self) -> float:
        return self.state[2]

    @soc_max.setter
    def soc_max(self, v: float):
        self.state[2] = v
        self._feasible_actions = None

    @property
    def state_of_charge(self) -> float:
        return self.state[0] / self.capacity

    @state_of_charge.setter
    def state_of_charge(self, v: float):
        self.state[0] = v * self.capacity
        self._feasible_actions = None

    def determine_feasible_actions(self) -> np.ndarray:
        dt = self._dt
        # determine these variables only once
        stored_energy = self.stored_energy
        relative_loss_term = self.relative_loss * (dt/60/60) / 2

        soc_min = self.soc_min
        soc_max = self.soc_max
        if self._training == False:
            # during simulation (= evaluation)
            soc_min = max(0, soc_min - self.constraint_fuzziness) # relax constraint
            soc_max = min(1, soc_max + self.constraint_fuzziness) # relax constraint

        # determine max and min power
        max_power = self.capacity * soc_max - stored_energy * (1 - relative_loss_term)/(1 + relative_loss_term)
        max_power *= 1 + relative_loss_term # Ws
        if max_power > 0: # charging
            max_power *= 1 / self.charging_efficiency / dt # W
        else: # discharging
            max_power *= self.discharging_efficiency / dt # W

        min_power = self.capacity * soc_min - stored_energy * (1 - relative_loss_term)/(1 + relative_loss_term)
        min_power *= 1 + relative_loss_term # Ws
        if min_power < 0: # discharging
            min_power *= self.discharging_efficiency / dt # W
        else: # charging
            min_power *= 1 / self.charging_efficiency / dt # W

        # determine feasible actions
        actions = np.copy(self._actions)
        actions = actions[actions <= round(max_power,1)] # filter actions that are too large
        if len(actions) == 0:
            # max_power < max discharging
            actions = np.array([min(self._actions)])
        actions = actions[actions >= round(min_power,1)] # filter actions that are too small
        if len(actions) == 0:
            # min_power > max charging
            actions = np.array([max(self._actions)])
                
        self._feasible_actions = actions
        return actions

    def transition(self, action, interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the systems state according to the provided parameters and returns the new state.
        
        Arguments
        - action: action to take. May be ``None`` for passive systems.
        - interaction: possible interactions
        """
        dt = self._dt
        action = self.correct_action(action)

        dE = 0
        if action > 0:
            # charging
            dE = action * dt * self.charging_efficiency 
        elif action < 0:
            # discharging
            dE = action * dt / self.discharging_efficiency

        relative_loss_term = self.relative_loss * (dt/60/60) / 2

        self.stored_energy = np.clip(self.stored_energy * (1 - relative_loss_term) / (1 + relative_loss_term) + dE / (1 + relative_loss_term), 0, self.capacity)

        # update feasible actions
        self._feasible_actions = None

        return self.state, interaction - np.array([action,0])

    def sample_state(self, soc_distribution: Tuple[List[Tuple[float,float]],List[float]]=([(0,1)],[1]), **kwargs) -> np.ndarray:
        """
        Provides samples for system states. 
        
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.

        Arguments
        - soc_distribution ``([(float,float)],[float])``: Tuple with intervals and probabilities. The result is then drawn uniformly from the randomly selected interval given.
        """        
        # 1. sample a state 
        (lower_bound, upper_bound) = soc_distribution[0][np.random.choice(len(soc_distribution[0]), 1, p=soc_distribution[1])[0]]
        soc = np.random.uniform(lower_bound, upper_bound)

        # 2. update the state to the newly sampled state
        
        if self._training:
            rnd = np.random.random()
            if rnd < 0.2:
                self.soc_min = 0
            elif rnd < 0.6:
                self.soc_min = np.random.uniform(0, 0.1)
            else:
                self.soc_min = np.random.uniform(0, min(0.9, soc))

            rnd = np.random.random()
            if rnd < 0.2:
                self.soc_max = 1
            elif rnd < 0.6:
                self.soc_max = np.random.uniform(0.9, 1)
            else:
                self.soc_max = np.random.uniform(max(0.1, soc), 1)

            self.stored_energy = np.round(self.capacity * soc)
        else:
            self.soc_min = 0 + self.constraint_fuzziness
            self.soc_max = 1 - self.constraint_fuzziness
            self.stored_energy = np.round(self.capacity * np.clip(soc, self.soc_min, self.soc_max))

        # 3. update feasible actions
        self._feasible_actions = None

        # 4. return the newly sampled state
        return self.state

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
        return np.repeat(self.state.reshape(1,-1), time_step_count, axis=0), np.array([[False, True, True]]).repeat(time_step_count, axis=0) # best guess