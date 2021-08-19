from modules.simulation.simulationmodel import SimulationModel
from typing import List, Tuple

import numpy as np

class Demand(SimulationModel):
    """
    Demand (passive, no actions).

    Please note: The HWT does NOT enforce the min and max temp boundaries.

    State = Forecast
    - demand ``float`` demand for current time step in W

    Note that transition() outputs the demand of the next period, remove this value by using the state normalization scalar

    Arguments
    - dt ``int`` delta time in seconds
    - demand_series ``np.ndaray`` an array holding one or multiple demand series in W. Shape [number of series, series length]
    - window_width ``int`` specifies the length of the range from which the values are sampled
    - demand_type ``np.ndarray`` vector to map demand onto the interaction
    - seconds_per_value ``int`` length of a time slot of the demand_series
    """

    def __init__(self, dt: int, demand_series: np.ndarray, demand_type: np.ndarray=np.array([0,1]), seconds_per_value: int=60):
        # passive system, do not call the constructor of super()
        super().__init__(dt, None, False)

        self.demand_series = demand_series
        self.window_width = int(dt / seconds_per_value)
        assert self.window_width > 0
        self.demand_type = demand_type

        self.state = np.array([0])        
        self.hidden_state = np.array([0, 0, np.empty([0,1])])

        self.possible_window_positions = np.arange(demand_series.shape[1] / self.window_width, dtype=int) * self.window_width

    def __repr__(self):
        return 'demand(state={}, hidden_state={})'.format(self.state, self.hidden_state)

    @property
    def demand(self):
        return self.state[0]

    @demand.setter
    def demand(self, v: int):
        self.state[0] = v

    @property
    def series_idx(self) -> int:
        return self.hidden_state[0]

    @series_idx.setter
    def series_idx(self, v: int):
        self.hidden_state[0] = v

    @property
    def window_position(self) -> int:
        return self.hidden_state[1]

    @window_position.setter
    def window_position(self, v: int):
        self.hidden_state[1] = v

    @property
    def forecast_series(self) -> np.ndarray:
        return self.hidden_state[2]

    @forecast_series.setter
    def forecast_series(self, v: np.ndarray):
        self.hidden_state[2] = v

    def determine_feasible_actions(self) -> np.ndarray:
        """
        Passive element, thus no actions.
        """
        return None

    def sample_state(self, window_position: int=-1, demand_interval: Tuple[int,int]=None, **kwargs) -> np.ndarray:
        """
        Sample the current demand

        Arguments
        - window_position ``int`` defines the position of the sampling window. The parameters specifies the lower end of the sampling range. Use a value lesser than 0 use a random window position.
        - demand_interval ``(int,int)`` during training this interval can be used to draw values from
        """        
        # 1. sample a state & 2. update the state to the newly sampled state
        if window_position < 0:
            window_position = np.random.choice(self.possible_window_positions)

        self.window_position = window_position
        self.series_idx = np.random.choice(self.demand_series.shape[0]) # pick a random series

        if self._training and (demand_interval is not None):
            self.state = np.array([np.random.uniform(demand_interval[0], demand_interval[1])])
        elif self._training:
            max_demand = np.max(self.demand_series)
            self.state = np.array([np.random.choice([0, max_demand, np.random.uniform(0, max_demand)], p=[0.1,0.01,0.89])])
        else:
            self.state = np.array([np.random.choice(self.demand_series[self.series_idx][window_position:(window_position+self.window_width)])])        
        
        self.forecast_series = np.empty([0,1])

        # 4. return the newly sampled state
        return self.state

    def forecast(self, time_step_count: int=1, **kwargs) -> np.ndarray:
        forecast = self.forecast_series

        if forecast.shape[0] < time_step_count-1:
            # more data is needed
            window_position = self.window_position
            window_width = self.window_width
            demand_series = self.demand_series
            series_idx = self.series_idx
            
            for step in range(forecast.shape[0], time_step_count-1):
                # move window                
                window_position += window_width
                window_position %= demand_series.shape[1]
                # create forecast
                new_state = np.random.choice(demand_series[series_idx][window_position:(window_position+window_width)])
                forecast = np.append(forecast, [[new_state]], axis=0)
            
            self.window_position = window_position
            self.forecast_series = forecast
        
        return np.concatenate((self.state, self.forecast_series[:time_step_count-1,0]), axis=0).reshape(-1,1), np.ones(time_step_count, dtype=bool).reshape(-1,1)

    def transition(self, action=0, interaction: np.ndarray=np.array([0,0])) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the systems state according to the provided parameters and returns the new state.
        
        Arguments
        - action ``float``: action to take. May be ``None`` for passive systems.
        - interaction ``np.ndarray``: possible interactions
        """
        demand = self.state

        forecast = self.forecast_series
        if forecast.shape[0] > 0:
            # there are forecasted values          
            # set next state according to forecast
            self.state = np.copy(forecast[0])
            # remove the old element
            self.forecast_series = np.delete(forecast, 0).reshape(-1,1)
        else:
            # move window
            window_position = (self.window_position + self.window_width) % self.demand_series.shape[1]
            self.window_position = window_position

            self.state = np.array([np.random.choice(self.demand_series[self.series_idx][window_position:(window_position+self.window_width)])])        
        
        return self.state, (interaction - demand * self.demand_type)