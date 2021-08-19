from modules.simulation.simulationmodel import SimulationModel
from modules.simulation.individual.bess import BESS
from modules.simulation.individual.chpp import CHPP
from modules.simulation.individual.hwt import HWT
from modules.simulation.individual.demand import Demand
from typing import List, Tuple

import numpy as np

class BESS_CHPP_HWT(SimulationModel):
    """
    A system combining a BESS, and a CHPP with an HWT

    State
    - BESS
        - stored energy
        - soc min
        - soc max
    - Demand
        - heat demand
    - CHPP
        - mode
        - dwell time
        - min off time
        - min on time
    - HWT
        - temperature
        - ambient temperature

    Arguments
    - bess ``BESS``
    - chpp ``CHPP``
    - hwt ``HWT``
    - demand ``Demand``: heat demand
    - constraint_fuzziness ``int`` radius of the border area in which the dwell time constraint is neglected
    """
    def __init__(self, dt: int, actions: np.ndarray, bess, chpp: CHPP, hwt: HWT, demand: Demand, constraint_fuzziness: float=0, correct_infeasible=False):
        super().__init__(dt, actions, correct_infeasible)

        self.bess = bess
        self.chpp = chpp
        self.hwt = hwt
        self.demand = demand

        self._constraint_fuzziness = constraint_fuzziness
        self.bess.constraint_fuzziness = constraint_fuzziness

    def __repr__(self):
        return 'BESS_CHPP_HWT({},{},{},{},constraint_fuzziness={})'.format(self.bess, self.chpp, self.hwt, self.demand, self.constraint_fuzziness)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self.bess.state, self.demand.state, self.chpp.state, self.hwt.state))

    @state.setter
    def state(self, v: np.ndarray):
        if v is not None:
            self.bess.state, v = np.split(v, [self.bess.state.shape[0]])
            self.demand.state, v = np.split(v, [self.demand.state.shape[0]])
            self.chpp.state, self.hwt.state = np.split(v, [self.chpp.state.shape[0]])
            self.bess._feasible_actions = None
            self.chpp._feasible_actions = None
            self._feasible_actions = None

    @property
    def hidden_state(self) -> np.ndarray:
        return self.demand.hidden_state

    @hidden_state.setter
    def hidden_state(self, v: np.ndarray):
        if v is not None:
            self.demand.hidden_state = v

    @property
    def constraint_fuzziness(self) -> np.ndarray:
        return self._constraint_fuzziness

    @constraint_fuzziness.setter
    def constraint_fuzziness(self, v: np.ndarray):
        self._constraint_fuzziness = v
        self.bess.constraint_fuzziness = v

    def determine_feasible_actions(self) -> np.ndarray:
        bess = self.bess
        chpp = self.chpp

        # determine possible chpp actions when ignoring the soft constraints
        all_chpp_actions = [_[0] for _ in chpp.state_matrix[chpp.mode]]
        feasible_chpp_actions = chpp.feasible_actions

        hwt = self.hwt
        constraint_fuzziness = self.constraint_fuzziness * 100
        # if hwt bounds are voilated, forget about the running time constraints. (equality for consistency with MILP)
        if self._training is True:
            # ANN data, enforce soft constraints on boundary, prevents actions near boundary
            if hwt.temperature <= hwt.soft_min_temp:
                # reached boundary, turn on or remain on
                feasible_chpp_actions = np.array([min(all_chpp_actions)]) # chpp output has negative sign, that is, min(.) means turn on
            elif hwt.temperature <= hwt.soft_min_temp + constraint_fuzziness and chpp.mode != 0:
                # on while inside boundary region, prevent turning off
                feasible_chpp_actions = np.array([min(all_chpp_actions)]) # on

            elif hwt.temperature >= hwt.soft_max_temp:
                # reached boundary, turn off or remain off
                feasible_chpp_actions = np.array([max(all_chpp_actions)]) # off
            elif hwt.temperature >= hwt.soft_max_temp - constraint_fuzziness and chpp.mode == 0:
                # off while inside boundary region, prevent turning on
                feasible_chpp_actions = np.array([max(all_chpp_actions)]) # off
                
        else:
            # Sim. data, relax the boundary a little
            if hwt.temperature <= hwt.soft_min_temp - constraint_fuzziness:
                # reached relaxed boundary, turn on
                feasible_chpp_actions = np.array([min(all_chpp_actions)]) # chp output has negative sign -> turn on
            elif hwt.temperature <= hwt.soft_min_temp + constraint_fuzziness and chpp.mode == 0:
                # off while inside boundary region, allow turning on
                feasible_chpp_actions = np.array(all_chpp_actions)          
           
            elif hwt.temperature >= hwt.soft_max_temp + constraint_fuzziness:
                # reached relaxed boundary, turn off
                feasible_chpp_actions = np.array([max(all_chpp_actions)])
            elif hwt.temperature >= hwt.soft_max_temp - constraint_fuzziness and chpp.mode != 0:
                # on while inside boundary region, allow turning off
                feasible_chpp_actions = np.array(all_chpp_actions)

        actions = self.actions
        self._feasible_actions = np.unique([actions[np.abs(actions - (a + b)).argmin()] for a in feasible_chpp_actions for b in bess.feasible_actions])
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

        chpp = self.chpp

        # determine possible chpp actions when ignoring the soft constraints
        chpp_actions = [_[0] for _ in chpp.state_matrix[chpp.mode]]
        chpp_action = None

        hwt = self.hwt
        constraint_fuzziness = self.constraint_fuzziness * 100
        if self._training is True:
            # ANN data
            if hwt.temperature < hwt.soft_min_temp:
                # must start up
                # this condition overrides the dwell time constraints
                chpp.ignore_dwell_time_off = True # temp. too low -> may turn on
                chpp._feasible_actions = None
                chpp_action = min(chpp_actions)
            elif hwt.temperature > hwt.soft_max_temp:
                # must shut down
                # this condition overrides the dwell time constraints
                chpp.ignore_dwell_time_on = True # temp. too high -> may turn off
                chpp._feasible_actions = None
                chpp_action = max(chpp_actions)

        else:
            # simulation data
            if hwt.temperature < hwt.soft_min_temp + constraint_fuzziness:
                # may start up
                # this condition overrides the dwell time constraints
                chpp.ignore_dwell_time_off = True # temp. too low -> may turn on
                chpp._feasible_actions = None
            elif hwt.temperature > hwt.soft_max_temp - constraint_fuzziness:
                # may shut down
                # this condition overrides the dwell time constraints
                chpp.ignore_dwell_time_on = True # temp. too high -> may turn off
                chpp._feasible_actions = None

        if chpp_action is None:
            possible_combinations = [(a, b) for a in self.bess.feasible_actions for b in chpp.feasible_actions if a + b == action]
            chpp_actions = [_[1] for _ in possible_combinations]
            # use chpp > draw electricity from grid for "low" hwt temperatures
            if hwt.temperature <= (hwt.soft_max_temp + hwt.soft_min_temp) / 2:
                chpp_action = min(chpp_actions) # turn on
            else:
                chpp_action = max(chpp_actions) # turn off  
        # chpp action is determined, only bess action is missing
        bess_action = action - chpp_action

        _, interaction = chpp.transition(chpp_action, interaction)
        _, interaction = self.demand.transition(None, interaction)
        _, interaction = hwt.transition(None, interaction)
        _, interaction = self.bess.transition(bess_action, interaction)

        self._feasible_actions = None

        return self.state, interaction

    def sample_state(self, **kwargs) -> np.ndarray:
        """
        [Optional] May be used to provide a standard sampling algorithm for system states. 
        To lower the number of required method calls during data generation, the internal state should be overwritten with the newly sampled one which is then also returned.
        """
        self.chpp.sample_state(**kwargs)
        self.hwt.sample_state(**kwargs)
        self.demand.sample_state(**kwargs)
        self.bess.sample_state(**kwargs)
        self._feasible_actions = None
        return self.state

    def forecast(self, time_step_count: int=1, **kwargs) -> np.ndarray:
        bess_forecast, bess_mask = self.bess.forecast(time_step_count, **kwargs)
        demand_forecast, demand_mask = self.demand.forecast(time_step_count, **kwargs)
        chpp_forecast, chpp_mask = self.chpp.forecast(time_step_count, **kwargs)
        hwt_forecast, hwt_mask = self.hwt.forecast(time_step_count, **kwargs)
        return np.concatenate((bess_forecast, demand_forecast, chpp_forecast, hwt_forecast), axis=1), \
                np.concatenate((bess_mask, demand_mask, chpp_mask, hwt_mask), axis=1)

    def train(self, mode: bool=True):
        self._training = mode
        self.bess.train(mode)
        self.chpp.train(mode)
        self.hwt.train(mode)
        self.demand.train(mode)
        self._feasible_actions = None