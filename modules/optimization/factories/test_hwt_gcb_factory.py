import pytest

from pyomo.core.base import Constraint

#import sys, os, shutil
#if os.getcwd() not in sys.path: sys.path.append('/home/foerderer/model-learning')

from modules.simulation.integrated.hwt_gcb import HWT_GCB
from modules.simulation.individual.hwt import HWT
from modules.simulation.individual.demand import Demand
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
    [(0,0)      , (0,-10000)],
    [(0,-2000)  , (0,-15000)]
]
demand = Demand(900, demand_series)
sim_model= HWT_GCB(900, 40, 60, 1, 1, 1, 0.01, state_matrix)

from pyomo.opt import TerminationCondition
def test_random_schedules():

    def run_test(sim_model, schedule_length, test_count):
        for i in range(test_count):
            schedule = []

            # sample state
            state = sim_model.sample_state()
            demand.sample_state()
            forecast, mask = demand.forecast(schedule_length)

            # prepare MILP
            milp = TargetDeviationMILP(sim_model.dt, schedule_length)
            milp.add_constraints(sim_model)
            def con_p_th(model, t):
                return getattr(model, '_P_th')[t] == int(-forecast[t])
            setattr(milp.model, 'con_P_th', Constraint(milp.model.t, rule=con_p_th))

            sim_temp = []
            for step in range(schedule_length):
                schedule.append(0)
                sim_model.transition(0, np.array([0, -forecast[step]]))
                sim_temp.append(sim_model.temperature)
            
            # solve
            milp.create_objective(schedule)
            result = milp.solve(verbose=False)

            assert result.solver.termination_condition != TerminationCondition.infeasible
            #if result.solver.termination_condition == TerminationCondition.infeasible:
            temp = np.array([milp.model._hwt_theta[i].value for i in milp.model.t])

            assert np.isclose(milp.model.obj.expr(), 0)
            assert max(temp - sim_temp) < 0.01 # accommodate imprecision

            off = [milp.model._gcb_b_off[i].value for i in milp.model.t]
            i_off = (state[0] ==  0)
            previous_temp = state[1]
            for step in range(schedule_length):
                if i_off == True and previous_temp <= sim_model.hwt.soft_min_temp:
                    i_off = False
                elif i_off == False and  previous_temp >= sim_model.hwt.soft_max_temp:
                    i_off = True
                assert i_off == off[step]
                previous_temp = temp[step]

    sim_model.eval()

    sim_model.constraint_fuzziness = 0
    run_test(sim_model, schedule_length, test_count)