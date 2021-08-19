import pytest

from modules.simulation.individual.hp import HP

import numpy as np

actions = HP.create_action_set(-1000, 1000, 201)

def test_state_manipulation():
    hp = HP(900, actions, 1., False)
    hp.specify_modes([0, 1000])

    hp.cold_sink_temperature = 12
    hp.hot_sink_temperature = 63
    assert hp.cold_sink_temperature == 12
    assert hp.hot_sink_temperature == 63
    assert np.isclose(hp.cop, (273.15+63)/(63-12))

    # reduced efficiency
    hp = HP(900, actions, 0.5, False)
    hp.specify_modes([0, 1000])

    hp.cold_sink_temperature = 12
    hp.hot_sink_temperature = 70
    assert np.isclose(hp.cop, 0.5*(273.15+70)/(70-12))

def test_get_feasible_actions():
    hp = HP(900, actions, 1., False)
    assert hp.feasible_actions.shape[0] == 101

    hp.specify_modes([0, 500, 1000])
    assert np.setdiff1d(hp.feasible_actions, [0, 500, 1000]).shape[0] == 101 - 3

def test_state_transition():
    # feasible actions
    hp = HP(900, actions, 1., False)
    hp.specify_modes([0, 500, 1000])

    hp.cold_sink_temperature = 10
    hp.hot_sink_temperature = 60
    state, interaction = hp.transition(0)
    assert (state == hp.state).all()
    assert (interaction == np.zeros(2)).all()

    cop = hp.cop
    state, interaction = hp.transition(1000)
    assert (state == hp.state).all()
    assert (interaction == np.array([-1000, 1000 * cop])).all()

    state, interaction = hp.transition(500)
    assert (state == hp.state).all()
    assert (interaction == np.array([-500, 500 * cop])).all()

    exception_raised = False
    try:
        state, interaction = hp.transition(-1000)
    except ValueError:
        exception_raised = True
    assert exception_raised

    # infeasible actions
    hp = HP(900, actions, 1., True)
    hp.specify_modes([0, 500, 1000])

    hp.cold_sink_temperature = 10
    hp.hot_sink_temperature = 60
    state, interaction = hp.transition(-5000)
    assert (state == hp.state).all()
    assert (interaction == np.zeros(2)).all()

    cop = hp.cop
    state, interaction = hp.transition(800) # is corrected to 1000
    assert (state == hp.state).all()
    assert (interaction == np.array([-1000, -1000 * cop])).all