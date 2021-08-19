import pytest

from modules.simulation.individual.chpp import CHPP

import numpy as np

actions = CHPP.create_action_set(-1000, 1000, 201)
state_matrix = [
    [(0,0), (-200,-400), (-400,-900)],
    [(-300,-600), (-500,-1000), (-700,-1400)],
    [(-600,-1100), (-800,-1600), (-1000,-2000)]
]

def test_state_manipulation():
    chpp = CHPP(900, actions, state_matrix, False)

    chpp.mode = 0
    assert chpp.mode == 0
    
    chpp.dwell_time = 1234
    assert chpp.dwell_time == 1234

    chpp.min_off_time = 4321
    assert chpp.min_off_time == 4321

    chpp.min_on_time = 2431
    assert chpp.min_on_time == 2431

def test_get_feasible_actions():
    # general
    chpp = CHPP(900, actions, state_matrix, False)

    chpp.mode = 0
    chpp.dwell_time = 900
    chpp.min_off_time = 3600
    chpp.min_on_time = 1800
    assert (chpp.feasible_actions == np.array([0])).all()

    chpp.dwell_time = 900 * 6
    assert np.setxor1d(chpp.feasible_actions, [0, -200, -400]).shape[0] == 0

    chpp.mode = 1
    assert np.setxor1d(chpp.feasible_actions, [-300, -500, -700]).shape[0] == 0

    chpp.dwell_time = 900
    assert np.setxor1d(chpp.feasible_actions, [-500, -700]).shape[0] == 0
    
    chpp.mode = 2
    assert np.setxor1d(chpp.feasible_actions, [-800, -1000]).shape[0] == 0

def test_state_transition():
    # feasible actions
    chpp = CHPP(900, actions, state_matrix, False)

    chpp.mode = 0
    chpp.dwell_time = 900
    chpp.min_off_time = 3600
    chpp.min_on_time = 1800
    state, interaction = chpp.transition(0)
    assert (state == chpp.state).all()
    assert chpp.mode == 0
    assert chpp.dwell_time == 1800
    assert chpp.min_off_time == 3600
    assert chpp.min_on_time == 1800
    assert (interaction == np.zeros(2)).all()

    chpp.dwell_time = 3600 * 12
    state, interaction = chpp.transition(-400) # switch on
    assert chpp.mode == 2
    assert chpp.dwell_time == 900
    assert chpp.min_off_time == 3600
    assert chpp.min_on_time == 1800
    assert np.isclose(interaction, np.array([400, 900])).all()

    state, interaction = chpp.transition(-1000, np.array([-2000, -2000])) # remain on
    assert chpp.mode == 2
    assert chpp.dwell_time == 1800
    assert chpp.min_off_time == 3600
    assert chpp.min_on_time == 1800
    assert np.isclose(interaction, np.array([-2000, -2000]) + np.array([1000, 2000])).all()

    state, interaction = chpp.transition(-800) # remain on, but in another mode
    assert chpp.mode == 1
    assert chpp.dwell_time == 2700
    assert chpp.min_off_time == 3600
    assert chpp.min_on_time == 1800
    assert np.isclose(interaction, np.array([800, 1600])).all()

    state, interaction = chpp.transition(-300) # shut off
    assert chpp.mode == 0
    assert chpp.dwell_time == 900
    assert chpp.min_off_time == 3600
    assert chpp.min_on_time == 1800
    assert np.isclose(interaction, np.array([300, 600])).all()

    # infeasible actions
    chpp = CHPP(900, actions, state_matrix, True)

    chpp.mode = 0
    chpp.dwell_time = 900
    chpp.min_off_time = 3600
    chpp.min_on_time = 1800
    state, interaction = chpp.transition(-1000)
    assert (state == chpp.state).all()
    assert chpp.mode == 0
    assert chpp.dwell_time == 1800
    assert chpp.min_off_time == 3600
    assert chpp.min_on_time == 1800
    assert (interaction == np.zeros(2)).all()

    chpp.mode = 2
    chpp.dwell_time = 900
    chpp.min_off_time = 3600
    chpp.min_on_time = 1800
    state, interaction = chpp.transition(0)
    assert chpp.mode == 1
    assert chpp.dwell_time == 1800
    assert chpp.min_off_time == 3600
    assert chpp.min_on_time == 1800
    assert (interaction == np.array([800, 1600])).all()