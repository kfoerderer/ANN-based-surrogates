from modules.simulation.simulationmodel import SimulationModel
from modules.simulation.individual.demand import Demand
from modules.simulation.individual.evse import EVSE
from modules.simulation.individual.bess import BESS
from modules.simulation.individual.chpp import CHPP
from modules.simulation.integrated.hwt_gcb import HWT_GCB
from typing import List, Tuple

from os import path

import numpy as np

class HoLL(SimulationModel):
    """
    A simplified model of the FZI House of Living Labs in Karlsruhe.

    Please note: The electrical output of the model does NOT include the non-DER consumption/production.

    State:
    - EVSE
        - capacity
        - SOC min
        - SOC max
        - SOC
        - remaining standing time
    - BESS
        - SOC
        - SOC min
        - SOC max
    - CHPP
        - mode
        - dwell time
        - min off time
        - min on time
    - HWT_GCB
        - GCB
            - mode: index of mode (0 to n)
        - HWT
            - temperature ``float`` C
            - ambient_temperature ``float`` C   
    - Demand (heat)
        - demand (forecast)

    Arguments`
    - constraint_fuzziness ``int`` radius of the border area in which the dwell time constraint is neglected
    """

    action_set_100w = np.linspace(-7900, 24400, 79 + 244 + 1)

    def __init__(self, dt: int, actions: np.ndarray, constraint_fuzziness: int=0, correct_infeasible=False):
        super().__init__(dt, actions, correct_infeasible)

        # reachable power
        # evse:     0       to  22 kW
        # bess:     -0.78       0.78 (x 3 = 2.34)
        # chpp:     -5.5        0
        # [demand:  -15 (PV)    50] non-DER
        # --------------------------
        # total:    -22.84      74.34 
        # -> well below the maximum of 3 * 120A * 230V
        # --------------------------
        # DERs:     -7.84       24.34
        # -> relevant action range for controlling the DERs
        # (el. demand is still relevant to determine how an action should be achieved)
        
        #electricity_demand_series = np.arange(96).reshape(-1,1) * 10
        #self.electricity_demand = Demand(dt, electricity_demand_series)    


        if path.isfile('data/holl_heat_demand.npy'):
            demand_series = np.load('data/holl_heat_demand.npy') # load stored series if available
        else:
            with open('data/holl_heat_demand.txt', 'r') as file:
                # load series
                demand_series = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[1], converters = {1: lambda s: float(s.strip() or np.nan)}) # read file, dismiss header
                
                # fill gaps
                idx = np.where(np.isnan(demand_series), 0, np.arange(demand_series.shape[0])) # create an array with indices, assigning 0's to gaps
                idx = np.maximum.accumulate(idx) # rolling maximum of previous value and next array element (=> replacing 0s with the previous non-gap idx)
                demand_series = demand_series[idx].reshape(-1, 24*60) # replace NaNs
            np.save('data/holl_heat_demand.npy', demand_series) # store result to avoid parsing the data again
        self.heat_demand = Demand(dt, demand_series)

        # evse
        # off, [3.6 kW to 22 kW]
        # [Tesla Model S] 70, 85 kWh
        # [Smart EQ fortwo] 17,6 kWh
        evse_actions = np.concatenate(([0], actions[(actions >= 3600) * (actions <= 22000)]))
        self.evse = EVSE(dt, actions, evse_actions, 1, correct_infeasible)

        # bess
        # 3 * 7.8 kWh C(10), 3*10kWh (C100)
        # => use C10, 780 W
        # 2% per month self discharge at 20°C, negligible (less than 1Wh per hour)
        # efficiency 0.78
        bess_actions = actions[(actions <= 3*780) & (actions >= 3*-780)]
        self.bess = BESS(dt, bess_actions, 3 * 7.8 * 1000 * 3600, 0.78, 1, 0., correct_infeasible)

        # chpp
        # 5.5 kW_el, 12.5 kW_th, >=10 minutes on / off    
        chpp_matrix = [
            [(0,0)              , (-5500/2,-12500/2)], # mode 0 = off (this is also required later)
            [(-5500/2,-12500/2) , (-5500,-12500)] # mode 1 = on
        ]
        # CHPP filters actions on its own
        self.chpp = CHPP(dt, actions, chpp_matrix, correct_infeasible=correct_infeasible)
        self.chpp.min_off_time = dt
        self.chpp.min_on_time = dt

        # gcb 
        # off -> on (~ 2min to reach 25 kW, and then ~13min to reach 60 kW) => 25 kWmin + (325 kWmin + 227.5 kWmin) = 577.5 kWmin -> / 15 min = 38.5 kW
        # on -> on 60 kW
        # on -> off (~2min to reach 0 kW) = 60 kWmin / 15 min = 4 kW
        #
        # hwt
        # 3300 l, 40 to 60° C, loss 693W at about 48°C, 16°C ambient
        # using 
        #   - 997 kg/m^3 => 3290.1 kg of water
        #   - 4190 J/(kg*K) => 48°C equals ~122.5379 kWh (at 16*C ambient temp.) 
        #   => 0.005655 1/h relative loss
        # 
        gcb_matrix = [
            [(0,0)      , (0,-38500)], 
            [(0,-4000)    , (0,-60000)]
        ]
        self.hwt_gcb = HWT_GCB(dt, 40, 60, 3.3, 1, 1, 0.005655, gcb_matrix)
        
        # compile the section arrays
        #sections = [self.electricity_demand.state.shape[0]]
        #sections.append(sections[-1] + self.evse.state.shape[0])
        sections = [self.evse.state.shape[0]]
        sections.append(sections[-1] + self.bess.state.shape[0])
        sections.append(sections[-1] + self.chpp.state.shape[0])
        sections.append(sections[-1] + self.hwt_gcb.state.shape[0])
        #sections.append(sections[-1] + self.heat_demand.state.shape[0])
        self.state_sections = sections

        self._constraint_fuzziness = constraint_fuzziness

        # performance optimization
        self.action_bins = [-float('inf')] +  [(l+r)/2 for l,r in zip(actions[:-1], actions[1:])] + [float('inf')]
        self.possible_action_combinations = None

    def __repr__(self):
        return 'HoLL({},{},{},{},{})'.format(self.evse, self.bess, self.chpp, self.hwt_gcb, self.heat_demand)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self.evse.state, self.bess.state, self.chpp.state, self.hwt_gcb.state, self.heat_demand.state))

    @state.setter
    def state(self, v: np.ndarray):
        if v is not None:
            self.evse.state, self.bess.state, self.chpp.state, self.hwt_gcb.state, self.heat_demand.state = np.split(v, self.state_sections)
            self.evse._feasible_actions = None
            self.bess._feasible_actions = None
            self.chpp._feasible_actions = None
            self._feasible_actions = None

    @property
    def hidden_state(self) -> np.ndarray:
        return np.array([self.evse.hidden_state, self.heat_demand.hidden_state])

    @hidden_state.setter
    def hidden_state(self, v: np.ndarray):
        if v is not None:
            self.evse.hidden_state, self.heat_demand.hidden_state = v[0], v[1]

    @property
    def constraint_fuzziness(self) -> np.ndarray:
        return self._constraint_fuzziness

    @constraint_fuzziness.setter
    def constraint_fuzziness(self, v: np.ndarray):
        self._constraint_fuzziness = v
        self.bess.constraint_fuzziness = v
        self.evse.constraint_fuzziness = v

    def determine_feasible_actions(self) -> np.ndarray:
        # gcb is holding ensuring the hwt min temperature -> only max temperature relevant
        chpp_actions = self.chpp.feasible_actions

        if not self._training:
            # simulation
            constraint_fuzziness = self.constraint_fuzziness * 100
            if self.hwt_gcb.temperature >= self.hwt_gcb.hwt.soft_max_temp + constraint_fuzziness:
                # reached max temp; must turn off
                chpp_actions = [self.chpp.state_matrix[self.chpp.mode][0][0]]
            elif self.hwt_gcb.temperature >= self.hwt_gcb.hwt.soft_max_temp - constraint_fuzziness and self.chpp.mode != 0:
                # allow turning off (but only if not already off)
                chpp_actions = [_[0] for _ in self.chpp.state_matrix[self.chpp.mode]]
        elif self.hwt_gcb.temperature >= self.hwt_gcb.hwt.soft_max_temp:
            # ANN
            # reached max temp; must turn off
            chpp_actions = [self.chpp.state_matrix[self.chpp.mode][0][0]]

        combinations = np.stack(np.meshgrid(self.evse.feasible_actions,
                                            self.bess.feasible_actions,
                                            chpp_actions),-1).reshape(-1,3)
        combinations = np.concatenate((np.sum(combinations, axis=1).reshape(-1,1), combinations), axis=1)
        combinations[:,0] = self.actions[np.digitize(combinations[:,0], self.action_bins)-1] # round to closest action
        self.possible_action_combinations = combinations
        self._feasible_actions = np.unique(combinations[:,0])

        return self._feasible_actions

    def transition(self, action, interaction: np.ndarray=np.zeros(2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the systems state according to the provided parameters and returns the new state.
        Arguments
        - dt: delta time in seconds
        - action: action to take. May be ``None`` for passive systems.
        - interaction: possible interactions
        """
        dt = self._dt
        action = self.correct_action(action)

        evse = self.evse
        chpp = self.chpp
        hwt = self.hwt_gcb.hwt

        # determine the best option for reaching the intended action
        chpp_actions = chpp.feasible_actions

        if self._training:
            # ANN
            if hwt.temperature >= hwt.soft_max_temp:
                # reached max temp -> turn off
                chpp.ignore_dwell_time_on = True
                chpp._feasible_actions = None
                chpp_actions = [chpp.state_matrix[chpp.mode][0][0]]
        else:
            # simulation
            constraint_fuzziness = self.constraint_fuzziness * 100
            if  hwt.temperature >= hwt.soft_max_temp + constraint_fuzziness:
                # use a higher boundary during the simulation
                # reached max temp -> turn off
                chpp.ignore_dwell_time_on = True # temp. too high -> may turn off
                chpp._feasible_actions = None
                chpp_actions = [chpp.state_matrix[chpp.mode][0][0]]

            if  hwt.temperature >= hwt.soft_max_temp - constraint_fuzziness and chpp.mode != 0:
                # allow turning off in the boundary region
                chpp.ignore_dwell_time_on = True # temp. too high -> may turn off
                chpp._feasible_actions = None
                chpp_actions = [_[0] for _ in chpp.state_matrix[chpp.mode]]

        possible_combinations = self.possible_action_combinations
        possible_combinations = possible_combinations[possible_combinations[:,0] == action][:,1:]
        chpp_actions = possible_combinations[:,2]

        #assert len(possible_combinations) > 0
        #assert np.isin(evse_actions, evse.feasible_actions).all()
        #assert np.isin(bess_actions, bess.feasible_actions).all()
        #assert np.isin(chpp_actions, chpp.feasible_actions).all()

        ##
        # Rules - MUST do
        ##
        
        # - turn off chpp when hwt temp is to high
        if hwt.temperature >= hwt.soft_max_temp:
            # forced stop
            chpp.ignore_dwell_time_on = True
            chpp._feasible_actions = None
            #assert np.isin(chpp_actions, [chpp.state_matrix[chpp.mode][0][0]]).all() # turn off
        
        ##
        # Rules - Priorities
        ##
        
        # chpp
        if len(chpp_actions) == 1:
            chpp_action = chpp_actions[0]
        else:
            # - use chpp > draw electricity from grid for "low" hwt temperatures
            if hwt.temperature <= (hwt.soft_max_temp + hwt.soft_min_temp) / 2:
                chpp_action = min(chpp_actions) # turn on
            else:
                chpp_action = max(chpp_actions) # turn off  

        # reduce possible action combinations
        possible_combinations = possible_combinations[possible_combinations[:,2] == chpp_action]
        evse_actions = possible_combinations[:,0]

        if len(evse_actions) <= 1:
            evse_action = evse_actions[0]
        else:
            # - load ev > load bess
            # - avoid transfering charge from bess to ev
            # is it possible to charge the ev without discharging the bess?
            evse_actions_without_bess_discharge = [evse for (evse, bess, chpp) in possible_combinations if bess >= 0]
            if len(evse_actions_without_bess_discharge) > 0:
                evse_action = max(evse_actions_without_bess_discharge)
            else:
                evse_action = min(evse_actions) # load as slowly as possible, as discharging the bess for the ev wastes energy

        # the solution for the bess can be computed from the other choices
        bess_action = action - chpp_action - evse_action
        # if it wasn't the rules would be:
        # - charge bess > feed into grid
        # - discharge bess > draw from grid

        #print(possible_combinations)        
        #print('chpp {}, evse {}, bess {} -> {}'.format(chpp_action, evse_action, bess_action, action))

        # do transitions
        #_, _ = self.electricity_demand.transition(None) # non-DER consumption/production is not part of the load schedule
        _, interaction = evse.transition(evse_action)
        _, interaction = self.bess.transition(bess_action, interaction)
        _, interaction = chpp.transition(chpp_action, interaction)
        _, interaction = self.heat_demand.transition(None, interaction)
        _, interaction = self.hwt_gcb.transition(None, interaction)

        self._feasible_actions = None

        return self.state, interaction

    def sample_state(self, **kwargs) -> np.ndarray:
        """
        [Optional] May be used to provide a standard sampling algorithm for system states. 
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.
        """
        #self.electricity_demand.sample_state(**kwargs)
        self.evse.sample_state(**kwargs)
        self.bess.sample_state(**kwargs)
        self.chpp.sample_state(**kwargs)
        #self.chpp.min_off_time = self._dt
        #self.chpp.min_on_time = self._dt
        self.hwt_gcb.sample_state(**kwargs)
        self.heat_demand.sample_state(**kwargs)
        self._feasible_actions = None
        return self.state

    def forecast(self, time_step_count: int=1, **kwargs) -> np.ndarray:
        # state = (self.electricity_demand.state, self.evse.state, self.bess.state, self.chpp.state, self.hwt_gcb.state, self.heat_demand.state)

        #electricity_demand_forecast, electricity_demand_mask = self.electricity_demand.forecast(time_step_count, **kwargs)
        evse_forecast, evse_mask = self.evse.forecast(time_step_count, **kwargs)
        bess_forecast, bess_mask = self.bess.forecast(time_step_count, **kwargs)
        chpp_forecast, chpp_mask = self.chpp.forecast(time_step_count, **kwargs)
        hwt_gcb_forecast, hwt_gcb_mask = self.hwt_gcb.forecast(time_step_count, **kwargs)
        heat_demand_forecast, heat_demand_mask = self.heat_demand.forecast(time_step_count, **kwargs)
        
        return  np.concatenate((evse_forecast, bess_forecast, chpp_forecast, hwt_gcb_forecast, heat_demand_forecast), axis=1),  \
                np.concatenate((evse_mask, bess_mask, chpp_mask, hwt_gcb_mask, heat_demand_mask), axis=1)

    def train(self, mode: bool=True):
        self._training = mode
        #self.electricity_demand.train(mode)
        self.evse.train(mode)
        self.bess.train(mode)
        self.chpp.train(mode)
        self.hwt_gcb.train(mode)
        self.heat_demand.train(mode)
        self._feasible_actions = None
