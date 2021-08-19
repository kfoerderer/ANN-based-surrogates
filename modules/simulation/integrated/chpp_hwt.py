from modules.simulation.simulationmodel import SimulationModel
from modules.simulation.individual.chpp import CHPP
from modules.simulation.individual.hwt import HWT
from modules.simulation.individual.demand import Demand
from typing import List, Tuple

import numpy as np

class CHPP_HWT(SimulationModel):
    """
    A system combining a CHPP with an HWT

    Arguments
    - chpp ``CHPP``
    - hwt ``HWT``
    - demand ``Demand``
    - constraint_fuzziness ``int`` radius of the border area in which the dwell time constraint is neglected
    """
    def __init__(self, chpp: CHPP, hwt: HWT, demand: Demand, constraint_fuzziness: int=0):
        super().__init__(chpp._dt, chpp.actions, chpp.correct_infeasible)

        self.chpp = chpp
        self.hwt = hwt
        self.demand = demand

        self.constraint_fuzziness = constraint_fuzziness

    def __repr__(self):
        return 'CHPP_HWT({},{},{},constraint_fuzziness={})'.format(self.chpp, self.hwt, self.demand, self.constraint_fuzziness)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self.demand.state, self.chpp.state, self.hwt.state))

    @state.setter
    def state(self, v: np.ndarray):
        if v is not None:
            self.demand.state, v = np.split(v, [self.demand.state.shape[0]])
            self.chpp.state, self.hwt.state = np.split(v, [self.chpp.state.shape[0]])
            self.chpp._feasible_actions = None
            self._feasible_actions = None

    @property
    def hidden_state(self) -> np.ndarray:
        return self.demand.hidden_state

    @hidden_state.setter
    def hidden_state(self, v: np.ndarray):
        if v is not None:
            self.demand.hidden_state = v

    def determine_feasible_actions(self) -> np.ndarray:
        chpp = self.chpp
        all_actions = [_[0] for _ in chpp.state_matrix[chpp.mode]]
        feasible_actions = chpp.feasible_actions

        hwt = self.hwt
        constraint_fuzziness = self.constraint_fuzziness * 100
        # if hwt bounds are voilated, forget about the running time constraints. (equality for consistency with MILP)
        if self._training is True:
            # ANN data, enforce soft constraints on boundary, prevents actions near boundary
            # a safety marging is trained into the model here, instead of adding it to the state vector
            if hwt.temperature <= hwt.soft_min_temp:
                # reached boundary, turn on or remain on
                feasible_actions = np.array([min(all_actions)]) # chpp output has negative sign, that is, min(.) means turn on
            elif hwt.temperature <= hwt.soft_min_temp + constraint_fuzziness and chpp.mode != 0: # safety margin
                # on while inside boundary region, prevent turning off
                feasible_actions = np.array([min(all_actions)]) # on

            elif hwt.temperature >= hwt.soft_max_temp:
                # reached boundary, turn off or remain off
                feasible_actions = np.array([max(all_actions)]) # off
            elif hwt.temperature >= hwt.soft_max_temp - constraint_fuzziness and chpp.mode == 0: # safety margin
                # off while inside boundary region, prevent turning on
                feasible_actions = np.array([max(all_actions)]) # off
                
        else:
            # Sim. data, relax the boundary a little
            if hwt.temperature <= hwt.soft_min_temp - constraint_fuzziness:
                # reached relaxed boundary, turn on
                feasible_actions = np.array([min(all_actions)]) # chp output has negative sign -> turn on
            elif hwt.temperature <= hwt.soft_min_temp + constraint_fuzziness and chpp.mode == 0:
                # off while inside boundary region, allow turning on
                feasible_actions = np.array(all_actions)          
           
            elif hwt.temperature >= hwt.soft_max_temp + constraint_fuzziness:
                # reached relaxed boundary, turn off
                feasible_actions = np.array([max(all_actions)])
            elif hwt.temperature >= hwt.soft_max_temp - constraint_fuzziness and chpp.mode != 0:
                # on while inside boundary region, allow turning off
                feasible_actions = np.array(all_actions)
          
        self._feasible_actions = feasible_actions
        return feasible_actions

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

        hwt = self.hwt
        constraint_fuzziness = self.constraint_fuzziness * 100
        if self._training is True:
            # ANN data
            if hwt.temperature < hwt.soft_min_temp:
                # must start up
                # this condition overrides the dwell time constraints
                self.chpp.ignore_dwell_time_off = True # temp. too low -> may turn on
                self.chpp._feasible_actions = None
            elif hwt.temperature > hwt.soft_max_temp:
                # must shut down
                # this condition overrides the dwell time constraints
                self.chpp.ignore_dwell_time_on = True # temp. too high -> may turn off
                self.chpp._feasible_actions = None

        else:
            # simulation data
            if hwt.temperature < hwt.soft_min_temp + constraint_fuzziness:
                # may start up
                # this condition overrides the dwell time constraints
                self.chpp.ignore_dwell_time_off = True # temp. too low -> may turn on
                self.chpp._feasible_actions = None
            elif hwt.temperature > hwt.soft_max_temp - constraint_fuzziness:
                # may shut down
                # this condition overrides the dwell time constraints
                self.chpp.ignore_dwell_time_on = True # temp. too high -> may turn off
                self.chpp._feasible_actions = None

        chpp_state, interaction = self.chpp.transition(action, interaction)
        _, interaction = self.demand.transition(None, interaction)
        hwt_state, interaction = hwt.transition(None, interaction)

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
        self._feasible_actions = None
        return self.state

    def forecast(self, time_step_count: int=1, **kwargs) -> np.ndarray:
        demand_forecast, demand_mask = self.demand.forecast(time_step_count, **kwargs)
        chpp_forecast, chpp_mask = self.chpp.forecast(time_step_count, **kwargs)
        hwt_forecast, hwt_mask = self.hwt.forecast(time_step_count, **kwargs)
        return np.concatenate((demand_forecast, chpp_forecast, hwt_forecast), axis=1), np.concatenate((demand_mask, chpp_mask, hwt_mask), axis=1)

    def train(self, mode: bool=True):
        self._training = mode
        self.chpp.train(mode)
        self.hwt.train(mode)
        self.demand.train(mode)
        self._feasible_actions = None