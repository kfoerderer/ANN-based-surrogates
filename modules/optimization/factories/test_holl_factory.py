import pytest

from modules.simulation.integrated.holl import HoLL
from modules.simulation.individual.hwt import HWT
from modules.simulation.individual.demand import Demand
from modules.optimization.targetdeviation import TargetDeviationMILP

import numpy as np
from pyomo.opt import TerminationCondition

schedule_length = 48
test_count = 5
np.random.seed(1924)

sim_model= HoLL(900, HoLL.action_set_100w, 0, correct_infeasible=True)

def test_random_schedules():

    def run_test(sim_model, schedule_length, test_count):
        for i in range(test_count):
            schedule = []

            # sample state
            state = sim_model.sample_state()
            forecast, mask = sim_model.forecast(schedule_length)

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
            temp = np.array([milp.model._hwtGcb_hwt_theta[i].value for i in milp.model.t])

            if feasible:
                assert np.isclose(milp.model.obj.expr(), 0)
            else:
                assert milp.model.obj.expr() > 0

            # temperature may diverge, as the MILP only tries to reproduce the el. power

    sim_model.eval()

    sim_model.constraint_fuzziness = 0
    run_test(sim_model, schedule_length, test_count)
    sim_model.constraint_fuzziness = 0.01
    run_test(sim_model, schedule_length, test_count)
    sim_model.constraint_fuzziness = 0.02
    run_test(sim_model, schedule_length, test_count)
