import pytest

from modules.simulation.individual.hwt import HWT

import numpy as np

actions = HWT.create_action_set(-1000, 1000, 201)

# constants
water_heat_capacity = 4190 # J/(kg*K)
water_density = 997 # kg/m^3
ws_per_j = 1. # Ws/J

def test_state_manipulation():
    v = 1. # m^3
    hwt = HWT(900, 60, 80, v, 1., 1., 0.)
    tank_ws_per_k = v * water_density * water_heat_capacity * ws_per_j # Ws/K
    hwt.stored_energy = tank_ws_per_k * 50 # 70°C    
    assert tank_ws_per_k * 50 == hwt.stored_energy
    assert hwt.capacity == (80 - 60) * tank_ws_per_k
    assert 70 ==  hwt.temperature
    assert np.isclose(hwt.state_of_charge, 0.5)
    hwt.temperature = 60
    assert np.isclose(tank_ws_per_k * 40, hwt.stored_energy)

    hwt.temperature = 70
    hwt.ambient_temperature = 30
    assert 30 == hwt.ambient_temperature
    assert np.isclose(hwt.state_of_charge, 0.5)
    assert np.isclose(tank_ws_per_k * 40, hwt.stored_energy)
    
def test_get_feasible_actions():
    v = 1. # m^3
    hwt = HWT(900, 60, 80, v, 1., 1., 0.)
    hwt.stored_energy = 500 * 60 * 60
    assert hwt.feasible_actions is None


def test_state_transition():
    v = 1. # m^3
    hwt = HWT(3600, 60, 80, v, 1., 1., 0.)
    tank_ws_per_k = v * water_density * water_heat_capacity * ws_per_j # Ws/K

    hwt.temperature = 50
    state, interaction = hwt.transition(0, np.array([0, 0]))
    assert (hwt.state == state).all()
    assert hwt.temperature == 50
    assert (np.array([0, 0]) == interaction).all()
    
    hwt = HWT(900, 60, 80, v, 1., 1., 0.)
    hwt.temperature = 70
    state, interaction = hwt.transition(0, np.array([1000, 1000]))
    assert np.isclose(hwt.temperature, 70 + 900 * 1000/tank_ws_per_k)
    assert (np.array([1000, 0]) == interaction).all()

    hwt.temperature = 70
    state, interaction = hwt.transition(0, np.array([-1000, -1000]))
    assert np.isclose(hwt.temperature, 70 - 900 * 1000/tank_ws_per_k)
    assert (np.array([-1000, 0]) == interaction).all()

    # charging efficiency
    hwt = HWT(900, 60, 80, v, 0.5, 1., 0.)
    hwt.temperature = 70
    state, interaction = hwt.transition(0, np.array([0, 1000]))
    assert np.isclose(hwt.temperature, 70 + 900 * 500/tank_ws_per_k)
    assert (np.array([0, 0]) == interaction).all()

    # discharging efficiency
    hwt = HWT(900, 60, 80, v, 1., 0.5, 0.)
    hwt.temperature = 70
    state, interaction = hwt.transition(0, np.array([0, -1000]))
    assert np.isclose(hwt.temperature, 70 - 900 * 2000/tank_ws_per_k)
    assert (np.array([0, 0]) == interaction).all()

    # loss
    hwt = HWT(3600, 60, 80, v, 1., 1., 1.)
    hwt.temperature = 80
    energy = hwt.stored_energy
    state, interaction = hwt.transition(0, np.zeros(2))
    assert np.isclose(hwt.stored_energy, 1./3 * energy)

    # max and min temp
    hwt = HWT(900, 60, 80, v, 1., 1., 0., max_temp=80)
    hwt.temperature = 80
    state, interaction = hwt.transition(0, np.array([0, 1000]))
    assert np.isclose(hwt.temperature, 80)
    assert (np.array([0, 1000]) == interaction).all()

    hwt.temperature = 80 - 100 * 900 / hwt.tank_ws_per_k # K - (W * s)/(Ws/K)
    state, interaction = hwt.transition(0, np.array([0, 1000]))
    assert np.isclose(hwt.temperature, 80)
    assert (np.array([0, 900]) == interaction).all()

    hwt.temperature = hwt.ambient_temperature
    state, interaction = hwt.transition(0, np.array([0, -1000]))
    assert np.isclose(hwt.temperature, hwt.ambient_temperature)
    assert (np.array([0, -1000]) == interaction).all()

    hwt.temperature = hwt.ambient_temperature + 100 * 900 / hwt.tank_ws_per_k # K + (W * s)/(Ws/K)
    state, interaction = hwt.transition(0, np.array([0, -1000]))
    assert np.isclose(hwt.temperature, hwt.ambient_temperature)
    assert np.isclose(np.array([0, -900]), interaction).all()
