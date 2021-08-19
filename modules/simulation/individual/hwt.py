from modules.simulation.simulationmodel import SimulationModel
from typing import List, Tuple

import numpy as np

# constants
water_heat_capacity = 4190 # J/(kg*K)
water_density = 997 # kg/m^3
ws_per_j = 1. # Ws/J

class HWT(SimulationModel):
    """
    Hot Water Tank (passive, no actions).

    Please note: The HWT does NOT enforce the min and max temp boundaries.

    State
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
    """

    @staticmethod
    def relative_loss_from_energy_label(a: float, b: float, volume: float, norm_temp_delta: float=45.):
        """
        For the EU energy labels see CELEX_32013R0812

        Arguments
        - a ``float`` see EU energy labels. [(A+,A),(A,B),(B,C),(C,D)]: [5.5, 8.5, 12, 16.66]
        - b ``float`` see EU energy labels. [(A+,A),(A,B),(B,C),(C,D)]: [3.16, 4.25, 5.93, 8.33]
        - volume ``int`` in m^3
        - norm_temp_delta ``float`` difference between tank and ambient temperature used to determine the tank loss (=45 K for multiple DIN norms)

        Returns
        - Relative loss per hour of a hot water tank with the given parameters.
        """
        # EU formula needs V in litres (1000 l/m^3)
        p = (a + b * (1000 * volume)**0.4) # W
        # relative loss
        # W / (K * (J/(kg*K)) (kg/m^3*m^3)) = W / (Ws) * s/h = 1/h
        return p / (norm_temp_delta * water_heat_capacity * (water_density * volume)) * 60 * 60

    def __init__(self, 
                dt: int,
                soft_min_temp: int, 
                soft_max_temp: int, 
                volume: float, 
                charging_efficiency: float, 
                discharging_efficiency: float, 
                relative_loss: float,
                max_temp: int=90):
        super().__init__(dt, None, False)

        self.volume = volume
        self.soft_min_temp = soft_min_temp 
        self.soft_max_temp = soft_max_temp
        self.max_temp = max_temp

        # (m^3 * kg/m^3) * (J/(kg*K)*Ws/J) = Ws/K
        self.tank_ws_per_k = (self.volume * water_density) * (water_heat_capacity * ws_per_j)
        # (K) * Ws/K = Ws
        self.capacity = (soft_max_temp - soft_min_temp) * self.tank_ws_per_k
        
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.relative_loss = relative_loss

        self.state: np.ndarray = np.array([20.,20.])

    def __repr__(self):
        return ('HWT(state={}, soft_mix_temp={}, soft_max_temp={}, '
                'volume={}, charging_efficiency={}, discharging_efficiency={}, '
                'relative_loss={}, max_temp={}'
                ')').format(self.state, self.soft_min_temp, self.soft_max_temp,
                            self.volume, self.charging_efficiency, self.discharging_efficiency,
                            self.relative_loss, self.max_temp)

    @property
    def stored_energy(self) -> int:
        """
        Computes the stored energy, given the average water temperature and ambient temperature.
        """
        return (self.temperature - self.ambient_temperature) * self.tank_ws_per_k

    @stored_energy.setter
    def stored_energy(self, v: int):
        # Ws / (Ws/K) + K = K
        self.temperature = (v / self.tank_ws_per_k) + self.ambient_temperature

    @property
    def ambient_temperature(self) -> int:
        return self.state[1]

    @ambient_temperature.setter
    def ambient_temperature(self, v: int):
        self.state[1] = v

    @property
    def temperature(self):
        return self.state[0]

    @temperature.setter
    def temperature(self, v: int):
        self.state[0] = v

    @property
    def state_of_charge(self) -> float:
        min_stored_energy = (self.soft_min_temp - self.ambient_temperature) * self.tank_ws_per_k
        return (self.stored_energy - min_stored_energy) / self.capacity

    def determine_feasible_actions(self) -> np.ndarray:
        """
        The HWT is a passive element and thus has no actions.
        """
        return None

    def sample_state(self, temp_distribution: Tuple[List[Tuple[float,float]],List[float]]=([(60,80)],[1]), ambient_temperatures: List[int]=[20], **kwargs) -> np.ndarray:
        """
        Provides a standard sampling algorithm for system states. 

        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.

        ### Parameters
        temp_distribution ``{float:(float, float)}`` Dictionary, mapping the cummulative probability to intervals [starting_point, end_point). 
        To determine the temperature a random number is drawn from [0,1).
        The dictionary keys are sorted and each pair of neighboring keys (including 0.) spans an interval.
        Make sure to include key 1.0. Doing so, each interval is drawn with probability ``key(i+1) - key(i)``.
        The result is then drawn uniformly from the respective interval given as dictionary value.

        ambient_temperatures ``[int]`` possible choices for the ambient temperature

        """        
        # 1. sample a state and 2. update
        (lower_bound, upper_bound) = temp_distribution[0][np.random.choice(len(temp_distribution[0]), 1, p=temp_distribution[1])[0]]
        self.temperature = np.random.uniform(lower_bound, upper_bound)
        self.ambient_temperature = np.random.choice(ambient_temperatures)
        # 3. return the newly sampled state
        return self.state

    def transition(self, action=0, interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the systems state according to the provided parameters and returns the new state.
        ### Parameters
        action ``float`` action to take. May be ``None`` for passive systems.

        interaction ``np.ndarray`` possible interactions
        """
        dt = self._dt
        # only thermal interaction
        th_interaction = interaction[1]
        # compute this only once
        relative_loss_term = self.relative_loss * (dt/60/60) / 2
        # delta energy
        dE = 0
        if th_interaction > 0: # charging            
            dE = th_interaction * dt * self.charging_efficiency 
        elif th_interaction < 0: # discharging
            dE = th_interaction * dt / self.discharging_efficiency          

        new_energy = self.stored_energy * (1 - relative_loss_term) / (1 + relative_loss_term) + dE / (1 + relative_loss_term)
        
        if th_interaction > 0:
            # max temperature is restricted
            max_stored_energy = (self.max_temp - self.ambient_temperature) * self.tank_ws_per_k # Ws
            if new_energy > max_stored_energy:
                self.temperature = self.max_temp
                return self.state, np.array([interaction[0], (new_energy - max_stored_energy) / dt]) # Ws / s
        elif th_interaction < 0:
            # can only cool down to ambient temperature
            if new_energy < 0:
                self.temperature = self.ambient_temperature
                return self.state, np.array([interaction[0], new_energy / dt])
        self.stored_energy = new_energy
        
        return self.state, (interaction - np.array([0, th_interaction]))

    def forecast(self, time_step_count: int=1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return np.repeat(self.state.reshape(1,-1), time_step_count, axis=0), np.repeat(np.array([[False, True]], dtype=bool), time_step_count, axis=0) # best guess