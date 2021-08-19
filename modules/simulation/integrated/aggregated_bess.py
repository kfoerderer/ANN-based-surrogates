from modules.simulation.simulationmodel import SimulationModel
from modules.simulation.individual.bess import BESS
import numpy as np
from typing import List, Tuple

class AggregatedBESS(SimulationModel):
    """
    An aggregate of n BESSs

    State
    - BESS (n times)
        - stored energy
    - soc min
    - soc max
    
    Arguments
    - dt ``int`` length of a time step in seconds
    - actions ``np.ndarray`` array holding all possible actions
    - capacities ``[int]`` capacity of each battery in Ws
    - max_charging_powers ``[int]'' maximum charging power of each battery
    - max_discharging_powers ``[int]'' maximum discharging power of each battery (negative sign)
    - charging_efficiencies ``[float]`` efficiency of charging of each battery
    - discharging_efficiencies ``[float]`` efficiency of discharging of each battery
    - relative_losses ``[float]`` relative storage energy loss per hour of each battery
    """

    def __init__(self, dt: int, actions: np.ndarray, capacities: List[int], max_charging_powers: List[int], max_discharging_powers: List[int],
                charging_efficiencies: List[float], discharging_efficiencies: List[float], relative_losses: List[float], 
                correct_infeasible: bool=False, constraint_fuzziness: float=0):
        super().__init__(dt, actions, correct_infeasible)

        assert len(capacities) == len(max_charging_powers)
        assert len(capacities) == len(max_discharging_powers)
        assert len(capacities) == len(charging_efficiencies)
        assert len(capacities) == len(discharging_efficiencies)
        assert len(capacities) == len(relative_losses)

        self.capacities = np.array(capacities)
        self.max_charging_powers = np.array(max_charging_powers)
        self.max_discharging_powers = np.array(max_discharging_powers)
        self.charging_efficiencies = np.array(charging_efficiencies)
        self.discharging_efficiencies = np.array(discharging_efficiencies)
        self.relative_losses = np.array(relative_losses)

        self.total_capacity = np.sum(capacities)

        self.state: np.ndarray = np.array([0] * len(capacities) + [0., 1.])

        self.constraint_fuzziness = constraint_fuzziness

    def __repr__(self):
        return ('AggregatedBESS(bess_count={}, capacity={}, aggregated_soc={}, correct_infeasible={}'
                ')').format(len(self.capacities), self.total_capacity, self.aggregated_state_of_charge, self.correct_infeasible)

    @property
    def stored_energy(self) -> int:
        return self.state[0:-2]

    @stored_energy.setter
    def stored_energy(self, v: np.ndarray):
        if v.shape[0] != self.capacities.shape[0]:
            raise ValueError('Vector has wrong dimension')
        self.state[0:-2] = v
        self._feasible_actions = None

    @property
    def soc_min(self) -> float:
        return self.state[-2]

    @soc_min.setter
    def soc_min(self, v: float):
        self.state[-2] = v
        self._feasible_actions = None

    @property
    def soc_max(self) -> float:
        return self.state[-1]

    @soc_max.setter
    def soc_max(self, v: float):
        self.state[-1] = v
        self._feasible_actions = None

    @property
    def state_of_charge(self) -> float:
        return self.state[0:-2] / self.capacities

    @property
    def aggregated_state_of_charge(self) -> float:
        return np.sum(self.state[0:-2]) / self.total_capacity

    def determine_feasible_actions(self) -> np.ndarray:
        dt = self._dt

        #
        # Please note:
        #   formulas are identical to the individual BESS formulas, but here the variables are vectors
        #

        # determine these variables only once
        stored_energy = self.stored_energy
        relative_loss_term = self.relative_losses * (dt/60/60) / 2

        soc_min = self.soc_min
        soc_max = self.soc_max
        if self._training == False:
            # during simulation (= evaluation)
            soc_min = max(0, soc_min - self.constraint_fuzziness) # relax constraint
            soc_max = min(1, soc_max + self.constraint_fuzziness) # relax constraint

        # determine max and min power
        max_powers = self.capacities * soc_max - stored_energy * (1 - relative_loss_term)/(1 + relative_loss_term)
        max_powers *= 1 + relative_loss_term # Ws
        max_powers *= 1 / self.charging_efficiencies / dt # W
        max_powers = np.minimum(max_powers, self.max_charging_powers)

        min_powers = self.capacities * soc_min - stored_energy * (1 - relative_loss_term)/(1 + relative_loss_term)
        min_powers *= 1 + relative_loss_term # Ws
        min_powers *= self.discharging_efficiencies / dt # W
        min_powers = np.maximum(min_powers, self.max_discharging_powers)

        self._max_powers = max_powers
        self._min_powers = min_powers

        total_max_power = np.sum(max_powers)
        total_min_power = np.sum(min_powers)

        # determine feasible actions
        actions = np.copy(self._actions)
        actions = actions[actions <= round(total_max_power,1)] # filter actions that are too large
        actions = actions[actions >= round(total_min_power,1)] # filter actions that are too small
                
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

        # never discharge a BESS when the total result should be charging and vice versa

        capacities = self.capacities
        powers = np.zeros(len(capacities))
        dE = np.zeros(len(capacities))
        min_powers = self._min_powers
        max_powers = self._max_powers
        if action > 0:
            powers = np.clip(min_powers, 0, float('inf')) # >= 0
            delta_action = action - np.sum(powers)
            max_powers = max_powers - powers

            # simple and naive charging strategy
            powers += max_powers * delta_action / np.sum(max_powers) # action can't be larger than max_power
            dE = powers * dt * self.charging_efficiencies

        elif action < 0:
            powers = np.clip(max_powers, -float('inf'), 0) # <= 0
            delta_action = action - np.sum(powers)
            min_powers = min_powers - powers

            # simple and naive charging strategy
            powers += min_powers * delta_action / np.sum(min_powers) # action can't be smaller than min_power
            dE = powers * dt / self.discharging_efficiencies

        relative_loss_term = self.relative_losses * (dt/60/60) / 2

        self.stored_energy = np.clip(self.stored_energy * (1 - relative_loss_term) / (1 + relative_loss_term) + dE / (1 + relative_loss_term), 0, self.capacities)

        # update feasible actions
        self._feasible_actions = None

        return self.state, interaction - np.array([np.sum(powers),0])

    def sample_state(self, soc_distribution: Tuple[List[Tuple[float,float]],List[float]]=([(0,1)],[1]), **kwargs) -> np.ndarray:
        """
        Provides samples for system states. 
        
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.

        Arguments
        - soc_distribution ``([(float,float)],[float])``: Tuple with intervals and probabilities. The result is then drawn uniformly from the randomly selected interval given.
        """        
        # 1. sample a state 
        if np.random.random() < 0.5:
            # all SOCs are in about the same range
            (lower_bound, upper_bound) = soc_distribution[0][np.random.choice(len(soc_distribution[0]), 1, p=soc_distribution[1])[0]]
            soc = np.random.uniform(lower_bound, upper_bound, len(self.capacities))
        else:
            # completely random SOCs
            soc = np.random.random(len(self.capacities))

        # 2. update the state to the newly sampled state        
        if self._training:
            rnd = np.random.random()
            if rnd < 0.2:
                self.soc_min = 0
            else:
                self.soc_min = np.random.uniform(0, 0.1)
            
            rnd = np.random.random()
            if rnd < 0.2:
                self.soc_max = 1
            else:
                self.soc_max = np.random.uniform(0.9, 1)

            self.stored_energy = np.round(self.capacities * soc)
        else:
            self.soc_min = 0 + self.constraint_fuzziness
            self.soc_max = 1 - self.constraint_fuzziness
            self.stored_energy = np.round(self.capacities * np.clip(soc, self.soc_min, self.soc_max))

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
        return np.repeat(self.state.reshape(1,-1), time_step_count, axis=0), np.array([[False] * len(self.capacities) + [True, True]]).repeat(time_step_count, axis=0) # best guess