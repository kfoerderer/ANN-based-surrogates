import pytest

from modules.simulation.integrated.aggregated_bess import AggregatedBESS
from modules.optimization.targetdeviation import TargetDeviationMILP

import numpy as np

schedule_length = 48
test_count = 4
np.random.seed(1924)
actions = AggregatedBESS.create_action_set(-100000, 100000, 201)
bess_count = 100
capacities = np.random.choice([1000, 2000, 50000, 10000], bess_count, replace=True) * 3600
max_charging_powers = np.random.choice([100, 250, 500, 1000], bess_count, replace=True)
max_discharging_powers = -max_charging_powers # symmetric
chargin_efficiencies = np.random.choice([0.9] + [1]*9, bess_count, replace=True)
discharging_efficiencies = np.random.choice([0.9] + [1]*9, bess_count, replace=True)
relative_losses = np.random.choice([0.01] + [0]*9, bess_count, replace=True)

def test_random_schedules():

    def run_test(sim_model, schedule_length, test_count):
        for i in range(test_count):
            schedule = []

            # sample state
            sim_model.sample_state()

            target_feasible =  True #np.random.random() < 0.5 # sim infeasible does not imply MILP infeasible
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
            milp.solve(verbose=True)

            if feasible:
                assert np.isclose(milp.model.obj.expr(), 0)
            else:
                assert milp.model.obj.expr() > 0

    sim_model = AggregatedBESS(900, actions, capacities, max_charging_powers, max_discharging_powers, 
                                chargin_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    sim_model.eval()

    sim_model.constraint_fuzziness = 0
    run_test(sim_model, schedule_length, test_count)
    #sim_model.constraint_fuzziness = 0.01
    #run_test(sim_model, schedule_length, test_count)
    #sim_model.constraint_fuzziness = 0.02
    #run_test(sim_model, schedule_length, test_count)