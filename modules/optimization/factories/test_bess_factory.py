import pytest

from modules.simulation.individual.bess import BESS
from modules.optimization.targetdeviation import TargetDeviationMILP

import numpy as np

schedule_length = 48
test_count = 5
np.random.seed(1924)
actions = BESS.create_action_set(-1000, 1000, 201)

def test_random_schedules():

    def run_test(sim_model, schedule_length, test_count):
        for i in range(test_count):
            schedule = []

            # sample state
            sim_model.sample_state()

            target_feasible = np.random.random() < 0.5
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
            milp.solve(verbose=False)

            if feasible:
                assert np.isclose(milp.model.obj.expr(), 0)
            else:
                assert milp.model.obj.expr() > 0

    sim_model = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., True)
    sim_model.eval()

    sim_model.constraint_fuzziness = 0
    run_test(sim_model, schedule_length, test_count)
    #sim_model.constraint_fuzziness = 0.01
    #run_test(sim_model, schedule_length, test_count)
    #sim_model.constraint_fuzziness = 0.02
    #run_test(sim_model, schedule_length, test_count)
    
