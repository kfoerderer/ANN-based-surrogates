from modules.simulation.simulationmodel import SimulationModel
from typing import List, Tuple

import numpy as np

class EVSE(SimulationModel):
    """
    Electric vehicle supply equipment, only grid2vehicle.

    State (at t=0) / Forecasts (injected at given points in time)
    - capacity ``int``: Ws
    - soc_min ``float``: %
    - soc_max ``float``: %
    - soc ``float``: %
    - remaining_standing_time ``int``: s, only reduces once the BEV is available

    Note that there is no need for a 'time till arival' as 
    - the previous BEV needs to leave, allowing only idle afterwards
    - the parameters are updated on arrival

    Arguments
    - dt ``int``: length of a time step in seconds
    - actions ``np.ndarray``: array holding all possible actions
    - evse_actions ``np.ndarray``: array holding all actions performable by the EVSE
    - charging_efficiency ``float``: efficiency of charging
    - constraint_fuzziness ``float'': width of the region, with relaxed constraints
    """

    def __init__(self, dt: int, actions: np.ndarray, evse_actions: np.ndarray, charging_efficiency: float, correct_infeasible=False, constraint_fuzziness:float=0):
        super().__init__(dt, actions, correct_infeasible)

        self.evse_actions = evse_actions
        self.charging_efficiency = charging_efficiency

        self.state = np.array([0, 0., 0., 0., 0])
        self.hidden_state = np.array([np.empty((0,5)), np.empty((0,5),dtype=np.bool)]) # forecats, mask

        self.constraint_fuzziness = constraint_fuzziness

    def __repr__(self):
        return ('EVSE(state={}, charging_efficiency={}, correct_infeasible={}, constraint_fuzziness={}'
                ')').format(self.state, self.capacity, self.charging_efficiency, self.correct_infeasible, self.constraint_fuzziness)

    @property
    def capacity(self) -> int:
        return self.state[0]

    @capacity.setter
    def capacity(self, v: int):
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
    def soc(self) -> float:
        return self.state[3]

    @soc.setter
    def soc(self, v: float):
        self.state[3] = v
        self._feasible_actions = None

    @property
    def remaining_standing_time(self) -> int:
        return self.state[4]

    @remaining_standing_time.setter
    def remaining_standing_time(self, v: int):
        self.state[4] = v
        self._feasible_actions = None

    @property
    def forecast_series(self) -> np.ndarray:
        return self.hidden_state[0]

    @forecast_series.setter
    def forecast_series(self, v: np.ndarray):
        self.hidden_state = np.array([v, self.hidden_state[1]])

    @property
    def forecast_mask(self) -> np.ndarray:
        return self.hidden_state[1]

    @forecast_mask.setter
    def forecast_mask(self, v: np.ndarray):
        self.hidden_state = np.array([self.hidden_state[0], v])

    def determine_feasible_actions(self) -> np.ndarray:
        dt = self.dt
        remaining_standing_time = self.remaining_standing_time
        if remaining_standing_time <= 0:
            # car is not present, nothing can be done
            self._feasible_actions = np.zeros(1)
            return np.zeros(1)

        actions = np.copy(self.evse_actions)

        if self._training == True:
            # during training, stick to the exact constraints
            soc_max = self.soc_max
            soc_min = self.soc_min
        else:
            # during simulation, relax the constraints
            soc_max = min(self.soc_max + self.constraint_fuzziness, 1)
            soc_min = max(self.soc_min - self.constraint_fuzziness, 0)

        # determine max and min power
        max_power = self.capacity * (soc_max - self.soc)
        max_power *= 1 / self.charging_efficiency / dt # W
        max_power = min(max(actions), round(max_power))

        remaining_energy = max(0, self.capacity * (soc_min - self.soc)) / self.charging_efficiency # Ws
        if max_power == 0:
            time_required = 0
        else:
            time_required = remaining_energy / max_power # Ws / W = s

        if time_required >= remaining_standing_time: # > 0 (already checked above)
            # needs to charge as fast as possible
            min_power = max_power
        else:
            remaining_energy -= (remaining_standing_time - dt) * max_power # Ws - s * W = Ws
            min_power = max(remaining_energy, 0) / dt # Ws/s = W            

        # determine feasible actions        
        actions = actions[actions <= round(max_power,1)] # filter actions that are too large
        actions = actions[actions >= round(min_power,1)] # filter actions that are too small

        """
        if max_power > 0 and not (actions != 0).any(): # ("not full") and ("there is no option for charging", which includes "there is no option at all")
            # can happen when charging to 100% SOC
            # allow to charge less than dt seconds (only once, by allowing the highest action possible), but stick to the overall set of possible actions
            actions = np.array([0] + [max(self.actions[self.actions <= max_power])])
        """
        if len(actions) == 0:
            # can happen when charging to max SOC
            actions = np.zeros(1)

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

        self.soc += dE / self.capacity
        self.soc = min(1, self.soc)

        self.remaining_standing_time = max(0, self.remaining_standing_time-dt)

        if self.forecast_series.shape[0] > 0:
            # there is a forecast for this new time step
            if (self.forecast_mask[0] == True).any():
                # an update is required
                self.state = self.state * (1-self.forecast_mask[0]) + self.forecast_mask[0] * self.forecast_series[0]
            # discard first entry
            self.hidden_state = np.array([np.delete(self.forecast_series, 0, axis=0), np.delete(self.forecast_mask, 0, axis=0)])

        self._feasible_actions = None
        return self.state, interaction - np.array([action,0])

    def sample_state(self, 
                    possible_capacities: List[float]=[17.6, 27.2, 36.8, 37.9, 52, 70, 85, 100],
                    arrival_soc_distribution: Tuple[List[Tuple[float,float]],List[float]]=([(0,0.9)], [1]),
                    min_soc_distribution: Tuple[List[Tuple[float,float]],List[float]]=([(0.3,0.99)], [1]),
                    max_soc_distribution: Tuple[List[Tuple[float,float]],List[float]]=([(0.7,1),[1,1]], [0.8,0.2]),
                    possible_staying_times: List[float]=[i*900 for i in range(0,96+1)]+[i*900 for i in range(0,9)]*9, 
                    **kwargs) -> np.ndarray:
        """
        Provides samples for system states. 
        
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.

        Arguments
        - possible capacities: kWh
        - possible arrival socs: %
        - possible min socs: %
        - possible max socs: %
        - possible staying times: s
        """        
        # 1. sample a state & 2. update the state to the newly sampled state

        if np.random.random() < 0.05:
            # no BEV parked
            self.capacity = 1
            self.soc_min = 0
            self.soc_max = 1
            self.soc = 1
            self.remaining_standing_time = 0
        else:
            # more variety during training
            if self._training:
                # draw battery capacity
                self.capacity = np.random.uniform(10, max(possible_capacities) * 1.1) * 1000 * 60 * 60

                # draw SOC from distribution
                if np.random.random() < 0.2:
                    self.soc = np.random.uniform(0.9, 1)
                else:
                    self.soc = min(1, np.random.uniform(0, 1.1))

                # draw target soc
                self.soc_max = min(1, np.random.uniform(0, 1.1))
                self.soc_min = np.random.uniform(0, self.soc_max - min(self.evse_actions[self.evse_actions > 0])*self.dt/self.capacity)
                
                # draw remaining standing time
                self.remaining_standing_time = np.random.choice(possible_staying_times)
            else:
                # draw battery capacity
                self.capacity = np.random.choice(possible_capacities) * 1000 * 60 * 60

                # draw SOC from distribution
                (lower_bound, upper_bound) = arrival_soc_distribution[0][np.random.choice(len(arrival_soc_distribution[0]), 1, p=arrival_soc_distribution[1])[0]]
                self.soc = np.random.uniform(lower_bound, upper_bound)

                # draw target soc
                (lower_bound, upper_bound) = max_soc_distribution[0][np.random.choice(len(max_soc_distribution[0]), 1, p=max_soc_distribution[1])[0]]
                self.soc_max =  min(1-self.constraint_fuzziness, np.random.uniform(lower_bound, upper_bound))

                (lower_bound, upper_bound) = min_soc_distribution[0][np.random.choice(len(min_soc_distribution[0]), 1, p=min_soc_distribution[1])[0]]
                self.soc_min = min(self.soc_max - min(self.evse_actions[self.evse_actions > 0])*self.dt/self.capacity, np.random.uniform(lower_bound, upper_bound))

                # draw remaining standing time
                min_staying_time = np.ceil((self.soc_min - self.soc) * self.capacity / max(self.evse_actions) / self.dt) * self.dt # -> s
                self.remaining_standing_time = max(min_staying_time, np.random.choice(possible_staying_times))

        # 3. update feasible actions
        self._feasible_actions = None

        # 4. return the newly sampled state
        return self.state

    def forecast(self, time_step_count: int=1, **kwargs) -> np.ndarray:
        forecast = self.forecast_series
        mask = self.forecast_mask

        if forecast.shape[0] < time_step_count:
            # insufficient data, sample more

            dt = self.dt
            actions = self.evse_actions
            min_charge = min(actions[actions > 0]) * dt

            stay_queue = []
            # exp. distributed arrival times -> count is poisson distributed
            arrival_rate = kwargs.get('arrival_rate', 1/48) # 1/48 = twice a day on average
            possible_capacities = kwargs.get('possible_capacities', [17.6, 27.2, 36.8, 37.9, 52, 70, 85, 100])
            possible_arrival_socs = kwargs.get('possible_arrival_socs', np.linspace(0.1,0.9,17))
            
            if 'possible_min_socs' in kwargs:
                possible_min_socs = kwargs['possible_min_socs']
            elif self._training == True:
                possible_min_socs = np.linspace(0.3,0.95,70*2)
            else:
                possible_min_socs = np.array([i/100 for i in range(25, 96, 5)]) - self.constraint_fuzziness
            
            if 'possible_max_socs' in kwargs:
                possible_max_socs = kwargs['possible_max_socs']
            elif self._training == True:
                possible_max_socs = np.linspace(0.3,1,70*2+1)
            else:
                possible_max_socs = np.array([i/100 for i in range(30, 101, 5)]) - self.constraint_fuzziness
            
            possible_staying_times = kwargs.get('possible_staying_times', [i*900 for i in range(1,time_step_count+1)])

            if forecast.shape[0] == 0:
                capacity, min_soc, max_soc, soc, time = self.state
            else:
                capacity, min_soc, max_soc, soc, time = forecast[-1]

            # use a queueing system to generate arrivals
            for step in range(forecast.shape[0], time_step_count):
                # check for arrivals
                for arrival in range(np.random.poisson(arrival_rate)):
                    new_capacity = np.random.choice(possible_capacities) * 1000 * 60 * 60
                    new_soc = np.random.choice(possible_arrival_socs)
                    new_max_soc = min(1 - self.constraint_fuzziness, np.random.choice(possible_max_socs[possible_max_socs > new_soc]))
                    new_min_soc = min(new_max_soc - min_charge / new_capacity, np.random.choice(possible_min_socs[possible_min_socs > new_soc]))
                    
                    stay_queue.append((new_capacity, new_min_soc, new_max_soc, new_soc, np.random.choice(possible_staying_times)))
                
                if time > 0:
                    # current stay has not finished yet
                    time = max(0, time-dt)
                    #mask = np.concatenate((mask, [[False, False, False, False]]), axis=0)
                    mask = np.concatenate((mask, [[True, True, True, False, True]]), axis=0)

                elif len(stay_queue) > 0:
                    # charging station is empty
                    capacity, min_soc, max_soc, soc, time = stay_queue.pop(0)
                    # ensure staying time is plausible
                    min_staying_time = np.ceil((min_soc - soc) * capacity / max(actions) / dt) * dt # -> s (rounded up to a multiple of dt)
                    time = max(min_staying_time, time)
                    mask = np.concatenate((mask, [[True, True, True, True, True]]), axis=0)

                else:
                    # no car during this period
                    capacity = 1
                    min_soc = 0
                    max_soc = 1
                    soc = 1
                    time = 0
                    mask = np.concatenate((mask, [[True, True, True, True, True]]), axis=0)

                forecast = np.concatenate((forecast, [[capacity, min_soc, max_soc, soc, time]]), axis=0)
            
            self.hidden_state = np.array([forecast, mask])
        
        return np.concatenate(([self.state], forecast[:time_step_count-1]), axis=0), np.concatenate(([[False, False, False, False, False]], mask[:time_step_count-1]), axis=0)

