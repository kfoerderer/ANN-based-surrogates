from modules.simulation.simulationmodel import SimulationModel
from typing import List, Tuple

import numpy as np

from modules.simulation.individual.hwt import HWT

class HWT_GCB(SimulationModel):
    """
    A hot water tank combined with a gas condensing boiler. The combined system is "passive" in the sense that the GCB operates autonomously on a hysteresis control.

    State
    - GCB
        - mode: index of mode (0 to n)
    - HWT
        - temperature ``float`` C
        - ambient_temperature ``float`` C   

    Arguments
    - dt ``int`` delta time in seconds
    - soft_min_temp ``int`` lower (soft) bound for the water temperature
    - soft_max_temp ``int`` upper (soft) bound for the water temperature
    - volume ``float`` in m^3
    - charging_efficiency ``float`` efficiency of charging
    - discharging_efficiency ``float`` efficiency of discharging
    - relative_loss ``float`` relative storage energy loss per hour
    - gcb_production_matrix ``[[(int,int)]]``: 2x2 matrix defining the average power (el., th.) for each mode k (states[k][k]) and transition from j to k (states[j][k]).
        Mode 0 is assumed to be the 'off'-state
    """

    def __init__(self, 
                dt: int,
                soft_min_temp: int, 
                soft_max_temp: int, 
                volume: float, 
                charging_efficiency: float, 
                discharging_efficiency: float, 
                relative_loss: float,
                gcb_production_matrix: List[List[Tuple[int,int]]],
                max_temp: int=90):
        super().__init__(dt, None, False)

        # hwt
        self.hwt = HWT(dt, soft_min_temp, soft_max_temp, volume, charging_efficiency, discharging_efficiency, relative_loss, max_temp)

        # gcb
        self.state_matrix = gcb_production_matrix

        # initialize state array
        # 1. mode = current action
        self.gcb_state: np.ndarray = np.zeros(1, dtype=int)

    def __repr__(self):
        return ('HWT_GCB(state={}, soft_mix_temp={}, soft_max_temp={}, '
                'volume={}, charging_efficiency={}, discharging_efficiency={}, '
                'relative_loss={}, state_matrix={}, max_temp={}'
                ')').format(self.state, self.hwt.soft_min_temp, self.hwt.soft_max_temp,
                            self.hwt.volume, self.hwt.charging_efficiency, self.hwt.discharging_efficiency,
                            self.hwt.relative_loss, self.state_matrix, self.hwt.max_temp)

    ### 

    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self.gcb_state, self.hwt.state))

    @state.setter
    def state(self, v: np.ndarray):
        if v is not None:
            self.gcb_state, self.hwt.state = np.split(v, [1])

    ### hwt

    @property
    def stored_energy(self) -> int:
        return self.hwt.stored_energy

    @stored_energy.setter
    def stored_energy(self, v: int):
        self.hwt.stored_energy = v

    @property
    def ambient_temperature(self) -> int:
        return self.hwt.ambient_temperature

    @ambient_temperature.setter
    def ambient_temperature(self, v: int):
        self.hwt.ambient_temperature = v

    @property
    def temperature(self):
        return self.hwt.temperature

    @temperature.setter
    def temperature(self, v: int):
        self.hwt.temperature = v

    @property
    def state_of_charge(self) -> float:
        return self.hwt.state_of_charge

    ### gcb

    @property
    def mode(self) -> int:
        return np.rint(self.gcb_state[0]).astype(int)

    @mode.setter
    def mode(self, v: int):
        self.gcb_state = np.array([v])

    ###

    def determine_feasible_actions(self) -> np.ndarray:
        return None # passive

    def sample_state(self, **kwargs) -> np.ndarray:
        """
        [Optional] May be used to provide a standard sampling algorithm for system states. 
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.

        """
        
        # 1. sample a state & 2. update the state to the newly sampled state
        self.mode = np.random.choice(len(self.state_matrix))
        self.hwt.sample_state(**kwargs)

        # 3. (passive system, not needed)

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
        temperature = self.hwt.temperature
        current_mode = self.mode
        new_mode = current_mode

        # if soft constraints are violated, use GCB to counteract
        if temperature < self.hwt.soft_min_temp:
            new_mode = 1 # turn on
        elif temperature > self.hwt.soft_max_temp:
            new_mode = 0 # turn off
        
        el_power = state_matrix[current_mode][new_mode][0]
        th_power = state_matrix[current_mode][new_mode][1]
        
        # update state
        hwt_state, interaction = self.hwt.transition(0, interaction - np.array([el_power, th_power]))
        self.mode = new_mode
        
        return self.state, interaction