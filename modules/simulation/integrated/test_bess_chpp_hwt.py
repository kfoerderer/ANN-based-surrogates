import pytest

from modules.simulation.integrated.bess_chpp_hwt import BESS, CHPP, HWT, Demand, BESS_CHPP_HWT

import numpy as np

with open('data/total_thermal_demand.txt', 'r') as file:
    demand_series = np.loadtxt(file, delimiter='\t', skiprows=1) # read file, dismiss header
    demand_series = demand_series.transpose(1,0) # dim 0 identifies the series
    demand_series *= 1000 # kW -> W

hwt_volume = 0.750
hwt_min_temp = 60.
hwt_max_temp = 80.
relative_loss = HWT.relative_loss_from_energy_label(12, 5.93, hwt_volume, 45)

state_matrix = [
    [(0,0)      , (-4000,-1)],
    [(-1000,-2) , (-5500,-3)]
]
bess = BESS(900, BESS.create_action_set(-5000, 5000, 100 + 1), 13500 * 60 * 60, 1, 1, 0, True )
chpp = CHPP(900, CHPP.create_action_set(-5500, 0, 55 + 1), state_matrix, True)
hwt = HWT(900, hwt_min_temp, hwt_max_temp, hwt_volume, 1, 1, relative_loss)
demand = Demand(900, demand_series)
actions = BESS_CHPP_HWT.create_action_set(-10500, 5000, 105 + 50 + 1)
model= BESS_CHPP_HWT(900, actions, bess, chpp, hwt, demand, 0.01, True)

# constants
water_heat_capacity = 4190 # J/(kg*K)
water_density = 997 # kg/m^3
ws_per_j = 1. # Ws/J

def test_state_manipulation():
    model.bess.stored_energy = 123
    model.bess.soc_min = 2
    model.bess.soc_max = 4
    model.demand.demand = 11
    model.chpp.mode = 1
    model.chpp.dwell_time = 900
    model.chpp.min_off_time = 1800
    model.chpp.min_on_time = 2700
    model.hwt.temperature = 55
    model.hwt.ambient_temperature = 17

    assert (model.state == [123,2,4,11,1,900,1800,2700,55,17]).all()

    model.bess.soc_min = 0
    model.bess.soc_max = 1

def test_get_feasible_actions():
    def reset_state(mode, dwell_time, temperature, stored_energy=0):
        model.bess.stored_energy = stored_energy
        model.demand.demand = 11
        model.chpp.mode = mode
        model.chpp.dwell_time = dwell_time
        model.chpp.min_off_time = 1800
        model.chpp.min_on_time = 1800
        model.hwt.temperature = temperature
        model.hwt.ambient_temperature = 17

    bess_actions  = model.bess.feasible_actions

    reset_state(1, 900, 70)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-5500] + bess_actions).all() # must remain running
    assert len(model.feasible_actions) == len([-5500] + bess_actions)

    # top, lower boundary region
    reset_state(1, 900, 79.5)
    assert np.isin(model.determine_feasible_actions(), [-1000, -5500] + bess_actions.transpose().reshape(-1, 1).repeat(2, axis=1)).all() # may turn off

    reset_state(1, 900, 79.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-5500] + bess_actions).all() # must remain running

    reset_state(0, 1800, 79.5)
    assert np.isin(model.determine_feasible_actions(), [0] + bess_actions).all() # must remain stopped
    
    reset_state(0, 1800, 79.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [0, -4000] + bess_actions.transpose().reshape(-1, 1).repeat(2, axis=1)).all() # may turn on

    # top, upper boundary region
    reset_state(1, 900, 80.5)
    assert np.isin(model.determine_feasible_actions(), [-1000, -5500] + bess_actions.transpose().reshape(-1, 1).repeat(2, axis=1)).all() # may turn off

    reset_state(1, 900, 80.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-1000] + bess_actions).all() # must turn off
    
    # top, above boundary region
    reset_state(1, 900, 81.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-1000] + bess_actions).all() # must turn off

    reset_state(1, 900, 80.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-1000] + bess_actions).all() # must turn off

    # bottom, upper boundary region
    reset_state(0, 900, 60.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-4000, 0] + bess_actions.transpose().reshape(-1, 1).repeat(2, axis=1)).all() # may turn on
    assert len(model.feasible_actions) == len(np.unique([-4000, 0] + bess_actions.transpose().reshape(-1, 1).repeat(2, axis=1)))

    reset_state(0, 900, 60.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [0] + bess_actions).all() # must remain off

    reset_state(1, 1800, 60.5)
    assert np.isin(model.determine_feasible_actions(), [-5500] + bess_actions).all() # must remain running
    
    reset_state(1, 1800, 60.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-5500, -1000] + bess_actions.transpose().reshape(-1, 1).repeat(2, axis=1)).all() # may turn off

    # bottom, lower boundary region
    reset_state(0, 900, 59.5)
    assert np.isin(model.determine_feasible_actions(), [-4000, 0] + bess_actions.transpose().reshape(-1, 1).repeat(2, axis=1)).all() # may turn on
    
    reset_state(0, 900, 59.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-4000] + bess_actions).all() # must turn on

    # bottom, below boundary region
    reset_state(0, 900, 58.5)
    model.eval()
    assert np.isin(model.determine_feasible_actions(), [-4000] + bess_actions).all() # must turn on

    reset_state(0, 900, 58.5)
    model.train()
    assert np.isin(model.determine_feasible_actions(), [-4000] + bess_actions).all() # must turn on

    model.eval()

    # brute force
    np.random.seed(1924)
    for i in range(1000):
        state = model.sample_state()
        model.feasible_actions

    model.bess.soc_min = 0
    model.bess.soc_max = 1


def test_state_transition():
    def reset_state(mode, dwell_time, temperature, stored_energy=0):
        model.bess.stored_energy = stored_energy
        model.demand.demand = 11
        model.chpp.mode = mode
        model.chpp.dwell_time = dwell_time
        model.chpp.min_off_time = 1800
        model.chpp.min_on_time = 1800
        model.hwt.temperature = temperature
        model.hwt.ambient_temperature = 17
        model._feasible_actions = None

    model.eval()
    model.bess.soc_min = 0
    model.bess.soc_max = 1

    reset_state(0, 2700, 75, 0)
    state, interaction = model.transition(-5500) # chpp needs to run, since bess is empty
    assert model.chpp.mode == 1
    assert bess.stored_energy == 0
    assert interaction[0] == 4000

    reset_state(1, 2700, 75, 0)
    state, interaction = model.transition(-5000) # chpp needs to run, since bess is empty
    assert model.chpp.mode == 1
    assert bess.stored_energy == 500 * 900
    assert interaction[0] == 5000
    
    reset_state(1, 2700, 75, 5000*900)
    state, interaction = model.transition(-5000) # high tank temp., use bess and switch off chpp (=1000W)
    assert model.chpp.mode == 0 
    assert bess.stored_energy == 1000*900
    assert interaction[0] == 5000

    reset_state(0, 2700, 65, model.bess.capacity)
    state, interaction = model.transition(-5500) # chpp should run, since temp. is low
    assert model.chpp.mode == 1
    assert bess.stored_energy == model.bess.capacity - (5500 - 4000) * 900
    assert interaction[0] == 5500

    reset_state(1, 2700, 65, 5000*900)
    state, interaction = model.transition(-5000) # chpp should run, since temp. is low
    assert model.chpp.mode == 1
    assert bess.stored_energy == 5500 * 900
    assert interaction[0] == 5000

    # top, lower boundary region
    reset_state(1, 900, 79.5)
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 0  # may turn off
    assert bess.stored_energy == 0
    assert interaction[0] == 1000

    reset_state(1, 900, 79.5)
    model.train()
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 1 # must remain running, BESS will cover the difference
    assert bess.stored_energy == (5500-1000) * 900
    assert interaction[0] == 1000

    reset_state(1, 900, 79.5, model.bess.capacity)
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 1 # must remain running, BESS can not cover the difference
    assert np.isclose(bess.stored_energy, model.bess.capacity)
    assert interaction[0] == 5500

    reset_state(0, 1800, 79.5)
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 0 # must remain stopped
    assert bess.stored_energy == 0 
    assert interaction[0] == 0
    
    reset_state(0, 1800, 79.5)
    model.eval()
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 1 # may turn on, as battery is empty
    assert bess.stored_energy == 0 
    assert interaction[0] == 4000

    reset_state(0, 1800, 79.5, model.bess.capacity)
    model.eval()
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 0 # may turn on, but won't as battery can be discharged
    assert bess.stored_energy == model.bess.capacity - 4000 * 900
    assert interaction[0] == 4000

    # top, upper boundary region
    reset_state(1, 900, 80.5) 
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 0 # may turn off
    assert bess.stored_energy == 0
    assert interaction[0] == 1000

    reset_state(1, 900, 80.5)
    model.train() 
    state, interaction = model.transition(-5500)
    assert model.chpp.mode == 0 # must turn off
    assert bess.stored_energy == 0
    assert interaction[0] == 1000 # -1000 is closer to -5500 than 0
    
    # top, above boundary region
    reset_state(1, 900, 81.5)
    model.eval()
    state, interaction = model.transition(-5500)
    assert model.chpp.mode == 0 # must turn off
    assert bess.stored_energy == 0 
    assert interaction[0] == 1000

    reset_state(1, 900, 80.5)
    model.train()
    state, interaction = model.transition(-5500)
    assert model.chpp.mode == 0 # must turn off
    assert bess.stored_energy == 0
    assert interaction[0] == 1000 

    ###
    # bottom, upper boundary region
    reset_state(0, 900, 60.5)
    model.eval()
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 1 # may turn on
    assert bess.stored_energy == 0
    assert interaction[0] == 4000 

    reset_state(0, 900, 60.5)
    model.train()
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 0 # must remain off
    assert bess.stored_energy == 0 
    assert interaction[0] == 0

    reset_state(1, 1800, 60.5)
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 1 # must remain running, bess will cover the difference
    assert bess.stored_energy == (5500-1000)*900
    assert interaction[0] == 1000
    
    reset_state(1, 1800, 60.5, model.bess.capacity)
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 1 # must remain running, bess can not cover the difference
    assert bess.stored_energy == model.bess.capacity
    assert interaction[0] == 5500

    reset_state(1, 1800, 60.5)
    model.eval()
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 1 # may turn off, but won't as battery can charge
    assert bess.stored_energy == (5500-1000) * 900
    assert interaction[0] == 1000

    reset_state(1, 1800, 60.5, model.bess.capacity)
    state, interaction = model.transition(-1000)
    assert model.chpp.mode == 0 # may turn off, as battery is already full
    assert bess.stored_energy == model.bess.capacity
    assert interaction[0] == 1000

    # bottom, lower boundary region
    reset_state(0, 900, 59.5)
    state, interaction = model.transition(-4000)
    assert model.chpp.mode == 1 # may turn on
    assert bess.stored_energy == 0 
    assert interaction[0] == 4000
    
    reset_state(0, 900, 59.5)
    model.train()
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1 # must turn on, but bess has free capacity
    assert bess.stored_energy == 4000 * 900
    assert interaction[0] == 0

    reset_state(0, 900, 59.5, model.bess.capacity)
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1 # must turn on, and bess is full
    assert bess.stored_energy == model.bess.capacity
    assert interaction[0] == 4000

    # bottom, below boundary region
    reset_state(0, 900, 58.5)
    model.eval()
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1 # must turn on, but bess has free capacity
    assert bess.stored_energy == 4000 * 900
    assert interaction[0] == 0

    reset_state(0, 900, 58.5, model.bess.capacity)
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1 # must turn on, and bess is full
    assert bess.stored_energy == model.bess.capacity
    assert interaction[0] == 4000

    reset_state(0, 900, 58.5)
    model.train()
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1 # must turn on, but bess has free capacity
    assert bess.stored_energy == 4000 * 900
    assert interaction[0] == 0

    reset_state(0, 900, 58.5, model.bess.capacity)
    state, interaction = model.transition(-0)
    assert model.chpp.mode == 1 # must turn on, and bess is full
    assert bess.stored_energy == model.bess.capacity
    assert interaction[0] == 4000

    model.eval()

