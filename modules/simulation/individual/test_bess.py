import pytest

from modules.simulation.individual.bess import BESS

import numpy as np

actions = BESS.create_action_set(-1000, 1000, 201)

def test_state_manipulation():
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., False)

    bess.stored_energy = 500 * 60 * 60
    assert 500 * 60 * 60 == bess.stored_energy
    assert np.isclose(bess.state_of_charge, 0.5)

    bess.stored_energy = 100 * 60 * 60
    assert 100 * 60 * 60 == bess.stored_energy
    assert np.isclose(bess.state_of_charge, 0.1)

def test_get_feasible_actions():
    # general
    bess = BESS(3600, actions, 1000 * 60 * 60, 1., 1.,  0., False)
    bess.stored_energy = 500 * 60 * 60
    assert bess.feasible_actions.shape[0] == 101

    bess = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., False)
    bess.stored_energy = 500 * 60 * 60
    assert bess.feasible_actions.shape[0] == 201

    bess.stored_energy = 0 * 60 * 60
    assert (bess.feasible_actions == actions[100:]).all()

    bess.stored_energy = 3 * 60 * 60
    assert bess.feasible_actions.shape[0] == 102

    bess.stored_energy = 1000 * 60 * 60
    assert bess.feasible_actions.shape[0] == 101

    # eval()
    bess.soc_min = 0.9
    assert bess.feasible_actions.shape[0] == 41
    assert (bess.feasible_actions >= -400).all()

    bess.constraint_fuzziness = 0.02
    assert bess.determine_feasible_actions().shape[0] == 49
    assert (bess.feasible_actions >= -480).all()

    bess.train()
    assert bess.determine_feasible_actions().shape[0] == 41
    assert (bess.feasible_actions >= -400).all()

    bess.stored_energy = 500 * 60 * 60
    bess.soc_min = 0.5
    bess.soc_max = 0.6
    assert bess.determine_feasible_actions().shape[0] == 41
    assert (bess.feasible_actions <= 400).all()

    bess.eval()
    bess.constraint_fuzziness = 0.04
    assert bess.determine_feasible_actions().shape[0] == 73
    assert (bess.feasible_actions <= 560).all()
    assert (bess.feasible_actions >= -160).all()

    bess.soc_max = 1
    bess.soc_min = 0
    bess.constraint_fuzziness = 0

    # charging efficiency
    bess = BESS(900, actions, 1000 * 60 * 60, 0.5, 1., 0., False)
    
    bess.stored_energy = 875 * 60 * 60
    assert bess.feasible_actions.shape[0] == 201
    
    bess.stored_energy = 876 * 60 * 60
    assert bess.feasible_actions.shape[0] == 200

    # min soc
    bess = BESS(900, actions, 1000 * 60 * 60, 0.5, 1, 0., False)    
    bess.stored_energy = 0
    bess.soc_min = 0.1
    assert (bess.feasible_actions >= 800).all()

    # discharging efficiency
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 0.1, 0., False)
    
    bess.stored_energy = 250 * 60 * 60
    assert bess.feasible_actions.shape[0] == 101 + 10

    # max soc
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 0.5, 0., False)    
    bess.stored_energy = 1000 * 60 * 60
    bess.soc_max = 0.9
    assert (bess.feasible_actions <= -200).all()

    # loss
    bess = BESS(3600, actions, 1000 * 60 * 60, 1., 1., 1., False)
    
    bess.stored_energy = 1000 * 60 * 60
    assert bess.feasible_actions.shape[0] == 151

def test_state_transition():
    # general, feasible action
    bess = BESS(3600, actions, 1000 * 60 * 60, 1., 1., 0., False)    
    bess.stored_energy = 500 * 60 * 60
    state, interaction = bess.transition(0)
    assert (bess.state == state).all()
    assert 500 * 60 * 60 == bess.stored_energy
    assert (np.array([0,0]) == interaction).all()
    
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., False)
    bess.stored_energy = 500 * 60 * 60
    state, interaction = bess.transition(1000)
    assert np.isclose(750 * 60 * 60, bess.stored_energy)
    assert (np.array([-1000,0]) == interaction).all()
    
    bess = BESS(3600, actions, 1000 * 60 * 60, 1., 1., 0., False)    
    bess.stored_energy = 750 * 60 * 60
    state, interaction = bess.transition(-740)
    assert np.isclose(10 * 60 * 60, bess.stored_energy)
    assert (np.array([+740,0]) == interaction).all()

    # interaction
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., False)
    bess.stored_energy = 500 * 60 * 60
    state, interaction = bess.transition(-1000, np.array([1000, 500]))
    assert (np.array([2000,500]) == interaction).all()
    
    # charging efficiency
    bess = BESS(3600, actions, 1000 * 60 * 60, 0.5, 1., 0., False)
    bess.stored_energy = 0 * 60 * 60
    state, interaction = bess.transition(1000)
    assert np.isclose(bess.state_of_charge, 0.5)
    
    # discharging efficiency
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 0.5, 0., False)    
    bess.stored_energy = 1000 * 60 * 60
    state, interaction = bess.transition(-1000)
    assert np.isclose(bess.state_of_charge, 0.5)

    # loss
    bess = BESS(3600, actions, 1000 * 60 * 60, 1., 1., 1., False)    
    bess.stored_energy = 1000 * 60 * 60
    state, interaction = bess.transition(0)
    assert np.isclose(bess.state_of_charge, 1./3)
    
    # infeasible action
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., False)
    bess.stored_energy = 1000 * 60 * 60
    exception_raised = False
    try:
        bess.transition(1000)
    except ValueError:
        exception_raised = True
    assert exception_raised
    
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., True)
    bess.stored_energy = 1000 * 60 * 60
    state, interaction = bess.transition(1000)
    assert 1000 * 60 * 60 == bess.stored_energy
    assert (np.array([0,0]) == interaction).all()
    
    state, interaction = bess.transition(10)
    assert 1000 * 60 * 60 == bess.stored_energy
    assert (np.array([0,0]) == interaction).all()
    
    bess.stored_energy = 0 * 60 * 60
    state, interaction = bess.transition(-1000)
    assert 0 * 60 * 60 == bess.stored_energy
    assert (np.array([0,0]) == interaction).all()
    
    state, interaction = bess.transition(-10)
    assert 0 * 60 * 60 == bess.stored_energy
    assert (np.array([0,0]) == interaction).all()

def test_sample_state():
    bess = BESS(900, actions, 1000 * 60 * 60, 1., 1., 0., True)

    for i in range(100):
        bess.sample_state()

    bess.eval()
    for i in range(100):
        bess.sample_state()
        assert bess.state_of_charge >= bess.soc_min
        assert bess.state_of_charge <= bess.soc_max