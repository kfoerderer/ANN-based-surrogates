import pytest

#import sys, os, shutil
#if os.getcwd() not in sys.path: sys.path.append('/home/foerderer/model-learning')

from modules.simulation.individual.bess import BESS
from modules.simulation.individual.chpp import CHPP
from modules.simulation.individual.hwt import HWT
from modules.simulation.individual.demand import Demand
from modules.simulation.integrated.bess_chpp_hwt import BESS_CHPP_HWT
from modules.optimization.targetdeviation import TargetDeviationMILP

import numpy as np

schedule_length = 48
test_count = 5
np.random.seed(1924)

with open('data/heat_demand.txt', 'r') as file:
    demand_series = np.loadtxt(file, delimiter='\t', skiprows=1) # read file, dismiss header
    demand_series = demand_series.transpose(1,0) # dim 0 identifies the series
    demand_series *= 1000 # kW -> W

allow_infeasible_actions = True
hwt_volume = 0.750
hwt_min_temp = 60.
hwt_max_temp = 80.
relative_loss = HWT.relative_loss_from_energy_label(12, 5.93, hwt_volume, 45)

state_matrix = [
    [(0,0)      , (-4000,-10000)],
    [(-1000,-2000) , (-5500,-12500)]
]
bess = BESS(900, BESS.create_action_set(-5000, 5000, 100 + 1), 13500 * 60 * 60, 1, 1, 0, True )
chpp = CHPP(900, CHPP.create_action_set(-5500, 0, 56), state_matrix, correct_infeasible=allow_infeasible_actions)
hwt = HWT(900, hwt_min_temp, hwt_max_temp, hwt_volume, 1, 1, relative_loss)
demand = Demand(900, demand_series)
actions = BESS_CHPP_HWT.create_action_set(-10500, 5000, 105 + 50 + 1)
sim_model= BESS_CHPP_HWT(900, actions, bess, chpp, hwt, demand, 1/100, correct_infeasible=True)

from pyomo.opt import TerminationCondition
def test_random_schedules():

    def run_test(sim_model, schedule_length, test_count):
        for i in range(test_count):
            schedule = []

            # sample state
            state = sim_model.sample_state()

            target_feasible = True #np.random.random() < 0.5 # sim infeasible does not imply MILP infeasible
            feasible = True

            # prepare MILP
            milp = TargetDeviationMILP(sim_model.dt, schedule_length)
            milp.add_constraints(sim_model)

            for step in range(schedule_length):
                feasible_actions = sim_model.feasible_actions
                if target_feasible:
                    action = np.random.choice(feasible_actions)
                else:
                    action = np.random.choice(sim_model.actions)
                    if not action in feasible_actions:
                        feasible = False
                schedule.append(action)
                sim_model.transition(action)
            
            # solve
            milp.create_objective(schedule)
            result = milp.solve(verbose=False)

            assert result.solver.termination_condition != TerminationCondition.infeasible
            #if result.solver.termination_condition == TerminationCondition.infeasible:
            #temp = [milp.model._hwt_theta[i].value for i in milp.model.t]

            if feasible:
                assert np.isclose(milp.model.obj.expr(), 0)
            else:
                assert milp.model.obj.expr() > 0

    sim_model.eval()

    sim_model.constraint_fuzziness = 0
    run_test(sim_model, schedule_length, test_count)
    sim_model.constraint_fuzziness = 0.01
    run_test(sim_model, schedule_length, test_count)
    sim_model.constraint_fuzziness = 0.02
    run_test(sim_model, schedule_length, test_count)
    