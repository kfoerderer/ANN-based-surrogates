import pytest

from modules.simulation.integrated.chpp_hwt import CHPP, HWT, Demand, CHPP_HWT

import numpy as np

with open('data/total_thermal_demand.txt', 'r') as file:
    demand_series = np.loadtxt(file, delimiter='\t', skiprows=1) # read file, dismiss header
    demand_series = demand_series.transpose(1,0) # dim 0 identifies the series
    demand_series *= 1000 # kW -> W

allow_infeasible_actions = True
hwt_volume = 0.750
hwt_min_temp = 60.
hwt_max_temp = 80.
relative_loss = HWT.relative_loss_from_energy_label(12, 5.93, hwt_volume, 45)

state_matrix = [
    [(0,0)      , (-4000,-1)],
    [(-1000,-2) , (-5500,-3)]
]
chpp = CHPP(900, CHPP.create_action_set(-5500, 0, 56), state_matrix, correct_infeasible=allow_infeasible_actions)
hwt = HWT(900, hwt_min_temp, hwt_max_temp, hwt_volume, 1, 1, relative_loss)
demand = Demand(900, demand_series)
model= CHPP_HWT(chpp, hwt, demand, 1/100)

# constants
water_heat_capacity = 4190 # J/(kg*K)
water_density = 997 # kg/m^3
ws_per_j = 1. # Ws/J

def test_state_manipulation():
    model.demand.demand = 11
    model.chpp.mode = 1
    model.chpp.dwell_time = 900
    model.chpp.min_off_time = 1800
    model.chpp.min_on_time = 2700
    model.hwt.temperature = 55
    model.hwt.ambient_temperature = 17

    assert (model.state == [11,1,900,1800,2700,55,17]).all()

def test_get_feasible_actions():
    model.demand.demand = 11
    model.chpp.mode = 1
    model.chpp.dwell_time = 900
    model.chpp.min_off_time = 1800
    model.chpp.min_on_time = 1800
    model.hwt.temperature = 70
    model.hwt.ambient_temperature = 17

    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-5500]).all() # must remain running

    # top, lower boundary region
    model.chpp.mode = 1
    model.chpp.dwell_time = 900
    model.hwt.temperature = 79.5
    assert np.isin(model.determine_feasible_actions(), [-1000, -5500]).all() # may turn off
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-5500]).all() # must remain running

    model.chpp.mode = 0
    model.chpp.dwell_time = 1800
    assert np.isin(model.determine_feasible_actions(), [0]).all() # must remain stopped
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [0, -4000]).all() # may turn on

    # top, upper boundary region
    model.chpp.mode = 1
    model.chpp.dwell_time = 900
    model.hwt.temperature = 80.5
    assert np.isin(model.determine_feasible_actions(), [-1000, -5500]).all() # may turn off
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-1000]).all() # must turn off
    model.eval()

    # top, above boundary region
    model.chpp.mode = 1
    model.chpp.dwell_time = 900
    model.hwt.temperature = 81.5
    assert np.isin(model.determine_feasible_actions(), [-1000]).all() # must turn off
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-1000]).all() # must turn off
    model.eval()
    
    # bottom, upper boundary region
    model.chpp.mode = 0
    model.chpp.dwell_time = 900
    model.hwt.temperature = 60.5
    assert np.isin(model.determine_feasible_actions(), [-4000, 0]).all() # may turn on
    model.train()
    assert np.isin(model.determine_feasible_actions(), [0]).all() # must remain off

    model.chpp.mode = 1
    model.chpp.dwell_time = 1800
    assert np.isin(model.determine_feasible_actions(), [-5500]).all() # must remain running
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-5500, -1000]).all() # may turn off

    # bottom, lower boundary region
    model.chpp.mode = 0
    model.chpp.dwell_time = 900
    model.hwt.temperature = 59.5
    assert np.isin(model.determine_feasible_actions(), [-4000, 0]).all() # may turn on
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-4000]).all() # must turn on
    model.eval()

    # bottom, below boundary region
    model.chpp.mode = 0
    model.chpp.dwell_time = 900
    model.hwt.temperature = 58.5
    assert np.isin(model.determine_feasible_actions(), [-4000]).all() # must turn on
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-4000]).all() # must turn on
    model.eval()


def test_state_transition():
    def reset_state(mode, dwell_time, temperature):
        model.demand.demand = 11
        model.chpp.mode = mode
        model.chpp.dwell_time = dwell_time
        model.chpp.min_off_time = 1800
        model.chpp.min_on_time = 1800
        model.hwt.temperature = temperature
        model.hwt.ambient_temperature = 17

    reset_state(1, 900, 70)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-5500]).all() # must remain running

    # top, lower boundary region
    reset_state(1, 900, 79.5)
    assert np.isin(model.determine_feasible_actions(), [-1000, -5500]).all() # may turn off
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 0

    reset_state(1, 900, 79.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-5500]).all() # must remain running
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 1

    reset_state(0, 1800, 79.5)
    assert np.isin(model.determine_feasible_actions(), [0]).all() # must remain stopped
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 0
    
    reset_state(0, 1800, 79.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [0, -4000]).all() # may turn on
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 1

    # top, upper boundary region
    reset_state(1, 900, 80.5)
    assert np.isin(model.determine_feasible_actions(), [-1000, -5500]).all() # may turn off
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 0

    reset_state(1, 900, 80.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-1000]).all() # must turn off
    state, interaction = model.transition(-5500)
    assert model.chpp.mode == 0
    
    # top, above boundary region
    reset_state(1, 900, 81.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-1000]).all() # must turn off
    state, interaction = model.transition(-5500)
    assert model.chpp.mode == 0

    reset_state(1, 900, 88)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-1000]).all() # must turn off
    state, interaction = model.transition(-5500)
    assert model.chpp.mode == 0

    reset_state(1, 900, 80.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-1000]).all() # must turn off
    state, interaction = model.transition(-5500)
    assert model.chpp.mode == 0

    # bottom, upper boundary region
    reset_state(0, 900, 60.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-4000, 0]).all() # may turn on
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 1

    reset_state(0, 900, 60.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [0]).all() # must remain off
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 0

    reset_state(1, 1800, 60.5)
    assert np.isin(model.determine_feasible_actions(), [-5500]).all() # must remain running
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 1
    
    reset_state(1, 1800, 60.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-5500, -1000]).all() # may turn off
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 0

    # bottom, lower boundary region
    reset_state(0, 900, 59.5)
    assert np.isin(model.determine_feasible_actions(), [-4000, 0]).all() # may turn on
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 1
    
    reset_state(0, 900, 59.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-4000]).all() # must turn on
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1

    # bottom, below boundary region
    reset_state(0, 900, 58.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-4000]).all() # must turn on
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1

    reset_state(0, 900, 58.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-4000]).all() # must turn on
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1

    model.eval()