import pytest

from modules.simulation.integrated.hwt_gcb import HWT_GCB

import numpy as np

actions = HWT_GCB.create_action_set(-1000, 1000, 201)
state_matrix = [
    [(0,0), (0,-600)],
    [(0,-400), (0,-1000)],
]

# constants
water_heat_capacity = 4190 # J/(kg*K)
water_density = 997 # kg/m^3
ws_per_j = 1. # Ws/J

def test_state_manipulation():
    v = 1. # m^3
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix)
    tank_ws_per_k = v * water_density * water_heat_capacity * ws_per_j # Ws/K
    hwt_gcb.stored_energy = tank_ws_per_k * 50 # 70°C    
    assert tank_ws_per_k * 50 == hwt_gcb.stored_energy
    assert hwt_gcb.hwt.capacity == (80 - 60) * tank_ws_per_k
    assert 70 ==  hwt_gcb.temperature
    assert np.isclose(hwt_gcb.state_of_charge, 0.5)
    hwt_gcb.temperature = 60
    assert np.isclose(tank_ws_per_k * 40, hwt_gcb.stored_energy)

    hwt_gcb.temperature = 70
    hwt_gcb.ambient_temperature = 30
    assert 30 == hwt_gcb.ambient_temperature
    assert np.isclose(hwt_gcb.state_of_charge, 0.5)
    assert np.isclose(tank_ws_per_k * 40, hwt_gcb.stored_energy)

    hwt_gcb.mode = 0
    assert hwt_gcb.mode == 0

    hwt_gcb.mode = 1
    assert hwt_gcb.mode == 1
    

def test_get_feasible_actions():
    v = 1. # m^3
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix)
    assert hwt_gcb.feasible_actions is None

def test_state_transition():
    v = 1. # m^3
    hwt_gcb = HWT_GCB(3600, 60, 80, v, 1., 1., 0., state_matrix)
    tank_ws_per_k = v * water_density * water_heat_capacity * ws_per_j # Ws/K

    # too cold, gcb must start
    hwt_gcb.temperature = 50
    state, interaction = hwt_gcb.transition(0, np.array([0, 0]))
    assert (hwt_gcb.state == state).all()
    assert np.isclose(hwt_gcb.temperature, 50 + 3600*600/tank_ws_per_k)
    assert (np.array([0, 0]) == interaction).all()
    
    # too cold, gcb must remain running
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix)
    hwt_gcb.temperature = 50
    hwt_gcb.mode = 1
    state, interaction = hwt_gcb.transition(0, np.array([0, 0]))
    assert (hwt_gcb.state == state).all()
    assert np.isclose(hwt_gcb.temperature, 50 + 900*1000/tank_ws_per_k)
    assert (np.array([0, 0]) == interaction).all()
    
    # too cold, gcb must start, but consumption is equally high
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix)
    hwt_gcb.temperature = 50
    state, interaction = hwt_gcb.transition(0, np.array([-1000, -600]))
    assert np.isclose(hwt_gcb.temperature, 50)
    assert (np.array([-1000, 0]) == interaction).all()

    # temperature is ok
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix)
    hwt_gcb.temperature = 70
    state, interaction = hwt_gcb.transition(0, np.array([-1000, 0]))
    assert np.isclose(hwt_gcb.temperature, 70)
    assert (np.array([-1000, 0]) == interaction).all()

    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix)
    hwt_gcb.temperature = 70
    state, interaction = hwt_gcb.transition(0, np.array([1000, 1000]))
    assert np.isclose(hwt_gcb.temperature, 70 + 900 * 1000/tank_ws_per_k)
    assert (np.array([1000, 0]) == interaction).all()

    hwt_gcb.temperature = 70
    state, interaction = hwt_gcb.transition(0, np.array([-1000, -1000]))
    assert np.isclose(hwt_gcb.temperature, 70 - 900 * 1000/tank_ws_per_k)
    assert (np.array([-1000, 0]) == interaction).all()

    # temperature is too high, gcb must stop
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix)
    hwt_gcb.temperature = 85
    hwt_gcb.mode = 1
    state, interaction = hwt_gcb.transition(0, np.array([-1000, 0]))
    assert np.isclose(hwt_gcb.temperature, 85 + 900 * 400/tank_ws_per_k)
    assert (np.array([-1000, 0]) == interaction).all()

    # charging efficiency
    hwt_gcb = HWT_GCB(900, 60, 80, v, 0.5, 1., 0., state_matrix)
    hwt_gcb.temperature = 70
    state, interaction = hwt_gcb.transition(0, np.array([0, 1000]))
    assert np.isclose(hwt_gcb.temperature, 70 + 900 * 500/tank_ws_per_k)
    assert (np.array([0, 0]) == interaction).all()

    # discharging efficiency
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 0.5, 0., state_matrix)
    hwt_gcb.temperature = 70
    state, interaction = hwt_gcb.transition(0, np.array([0, -1000]))
    assert np.isclose(hwt_gcb.temperature, 70 - 900 * 2000/tank_ws_per_k)
    assert (np.array([0, 0]) == interaction).all()

    # loss
    hwt_gcb = HWT_GCB(3600, 60, 80, v, 1., 1., 1., state_matrix)
    hwt_gcb.temperature = 80
    energy = hwt_gcb.stored_energy
    state, interaction = hwt_gcb.transition(0, np.zeros(2))
    assert np.isclose(hwt_gcb.stored_energy, 1./3 * energy)

    # max and min temp
    hwt_gcb = HWT_GCB(900, 60, 80, v, 1., 1., 0., state_matrix, max_temp=80)
    hwt_gcb.temperature = 80
    state, interaction = hwt_gcb.transition(0, np.array([0, 1000]))
    assert np.isclose(hwt_gcb.temperature, 80)
    assert (np.array([0, 1000]) == interaction).all()

    hwt_gcb.temperature = 80 - 100 * 900 / tank_ws_per_k # K - (W * s)/(Ws/K)
    state, interaction = hwt_gcb.transition(0, np.array([0, 1000]))
    assert np.isclose(hwt_gcb.temperature, 80)
    assert (np.array([0, 900]) == interaction).all()