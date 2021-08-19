from modules.simulation.simulationmodel import SimulationModel
from .hwt import HWT
from typing import List, Tuple

import numpy as np

class HP(SimulationModel):
    """
    Heat pump.

    State
    - cold_sink_temperature: temperature of cold sink in C (Note: COP is computed with temperatures in K!)
    - hot_sink_temperature: temperature of hot sink in C
    """

    def __init__(self, dt: int, actions: np.ndarray, efficiency: float=1, correct_infeasible=False):
        """
        ### Parameters
        dt ``int`` length of a time step in seconds

        actions ``np.ndarray`` array holding all possible actions

        efficiency ``float`` efficiency factor for COP computation
        """
        super().__init__(dt, actions, correct_infeasible)

        self.efficiency = efficiency

        self.available_modes = np.array([v for v in self.actions if v >= 0]) # the hp CONSUMES electricity

        # initialize state array
        # 1. hot sink temperature (in C)
        # 2. cold sink temperature (in C)
        self.state: np.ndarray = np.array([50,0])

    def specify_modes(self, available_modes: List[int]):
        self.available_modes = np.array(available_modes)

    @property
    def hot_sink_temperature(self) -> int:
        return self.state[0]

    @hot_sink_temperature.setter
    def hot_sink_temperature(self, v: int):
        self.state[0] = v

    @property
    def cold_sink_temperature(self) -> int:
        return self.state[1]

    @cold_sink_temperature.setter
    def cold_sink_temperature(self, v: int):
        self.state[1] = v

    @property
    def cop(self)-> float:
        cold_sink_temp = self.cold_sink_temperature + 273.15 # K
        hot_sink_temp = self.hot_sink_temperature + 273.15 # K
        return self.efficiency * (hot_sink_temp / (hot_sink_temp - cold_sink_temp))

    def determine_feasible_actions(self) -> np.ndarray:
        self._feasible_actions  = self.available_modes
        return self.available_modes

    def sample_state(self, cold_sink_temperatures=[0], **kwargs) -> np.ndarray:
        """
        [Optional] May be used to provide a standard sampling algorithm for system states. 
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.
        """        
        # 1. sample a state & 2. update the state to the newly sampled state
        self.cold_sink_temperature = np.random.randint(cold_sink_temperatures)
        # 3. update feasible actions
        self._feasible_actions = None
        # 4. return the newly sampled state
        return self.state

    def transition(self, action, interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the systems state according to the provided parameters and returns the new state.
        Parameters:
        - action: action to take. May be ``None`` for passive systems.
        - interaction: possible interactions
        """
        action = self.correct_action(action)

        # update feasible actions
        self._feasible_actions = None
        
        return self.state, interaction - np.array([action, -action * self.cop])