import pytest

from modules.simulation.individual.evse import EVSE

import numpy as np

actions = EVSE.create_action_set(0, 2000, 201)
evse_actions = actions[actions <= 1000]
evse_actions = evse_actions[(evse_actions >= 100) + (evse_actions == 0)]

def test_state_manipulation():
    capacity = 1000 * 60 * 60
    evse = EVSE(900, actions, evse_actions, 1., False)

    evse.capacity = capacity
    evse.soc = 455 * 60 * 60 / capacity
    evse.soc_min = 800 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 9 * 900

    assert evse.capacity == capacity
    assert evse.soc == 455 * 60 * 60 / capacity
    assert evse.soc_min == 800 * 60 * 60 / capacity
    assert evse.soc_max == 1
    assert evse.remaining_standing_time == 9 * 900

def test_get_feasible_actions():
    capacity = 1000 * 60 * 60
    # stored energy
    evse = EVSE(900, actions, evse_actions, 1., False)
    evse.capacity = capacity
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 3600
    evse.soc = 500 * 60 * 60 / capacity
    assert evse.feasible_actions.shape[0] == 101 - 9

    evse.soc = 800 * 60 * 60 / capacity
    assert evse.feasible_actions.shape[0] == 81 - 9

    evse.soc = 1000 * 60 * 60 / capacity
    assert evse.feasible_actions.shape[0] == 1

    # stored energy target
    evse.soc = 0
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 0
    assert (evse.feasible_actions == 0).all()
    evse.remaining_standing_time = -900
    assert (evse.feasible_actions == 0).all()
    
    evse.remaining_standing_time = 60 * 60
    assert (evse.feasible_actions == 1000).all()
    evse.remaining_standing_time = 60
    assert (evse.feasible_actions == 1000).all()
    assert evse.feasible_actions.shape[0] == 1

    evse.soc_min = 900 * 60 * 60 / capacity
    evse.soc_max = 1
    assert (evse.feasible_actions >= 600).all()

    evse.soc = 800 * 60 * 60 / capacity
    evse.soc_min = 0.8
    evse.soc_max = 0.9
    evse.remaining_standing_time = 900
    assert evse.feasible_actions.shape[0] == 41 - 9
    evse.soc_max = 1

    evse = EVSE(3600, actions, evse_actions, 1., False)
    evse.capacity = capacity
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 3600
    evse.soc = 500 * 60 * 60 / capacity
    assert evse.feasible_actions.shape[0] == 1

    evse.soc = 0
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 60 * 60
    assert (evse.feasible_actions == 1000).all()

    evse.remaining_standing_time = 60
    evse.soc_min = 900 * 60 * 60 / capacity
    assert (evse.feasible_actions >= 900).all()
    assert evse.feasible_actions.shape[0] == 1

    # charging efficiency
    evse = EVSE(900, actions, evse_actions, 0.5, False)
    evse.capacity = capacity
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 3600
    
    evse.soc = 875 * 60 * 60 / capacity
    assert evse.feasible_actions.shape[0] == 101 - 9
    
    evse.soc = 876 * 60 * 60 / capacity
    assert evse.feasible_actions.shape[0] == 100 - 9

    evse.soc = 825 * 60 * 60 / capacity # 175 Wh missing, 15min => max 700W @ 100% efficiecny => 1.4kW @ 0.5 efficiency
    evse.remaining_standing_time = 1800 # => min 400W 
    assert (evse.feasible_actions >= 400).all()
    assert evse.feasible_actions.shape[0] == 61

    # final charging period
    evse = EVSE(900, actions, evse_actions, 1, False)
    evse.capacity = capacity
    evse.soc = 0.9
    evse.soc_min = 1
    evse.soc_max = 1
    
    evse.remaining_standing_time = 0
    assert np.isin(evse.feasible_actions, [0]).all()

    evse.remaining_standing_time = 900
    assert np.isin(evse.feasible_actions, [400]).all()

    evse.soc = 0.999
    assert len(evse.feasible_actions) > 0
    assert np.isin(evse.feasible_actions, [0]).all() # max action lesser than or equal to max power

    evse.soc = 0.998
    assert len(evse.feasible_actions) > 0
    assert np.isin(evse.feasible_actions, [0]).all() # max action lesser than or equal to max power
    
    evse.soc = 0.997
    assert len(evse.feasible_actions) == 1
    assert np.isin(evse.feasible_actions, [0]).all() # [0] and [max action lesser than or equal to max power]

    # relaxed constraints
    evse = EVSE(900, actions, evse_actions, 1, False)
    evse.capacity = capacity
    
    evse.train()
    evse.constraint_fuzziness = 0.01 
    # train() -> fuzziness doesn't matter
    evse.soc = 0.9
    evse.soc_min = 0.98 # >= 320W
    evse.soc_max = 0.99 # <= 360W
    evse.remaining_standing_time = 900
    assert evse.feasible_actions.shape[0] == 5
    assert (evse.feasible_actions >= 240).all()
    assert (evse.feasible_actions <= 360).all()

    evse.eval()
    # fuzziness = 1% 
    # => 
    # soc_min = 0.97
    # soc_max = 1
    print(evse.feasible_actions)
    assert evse.feasible_actions.shape[0] == 13
    assert (evse.feasible_actions >= 280).all()
    assert (evse.feasible_actions <= 400).all()

def test_state_transition():
    capacity = 1000 * 60 * 60
    # general, feasible action
    evse = EVSE(3600, actions, evse_actions, 1., False)
    evse.capacity = capacity
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 3600 * 2
    evse.soc = 500 * 60 * 60 / capacity
    state, interaction = evse.transition(0)
    assert evse.capacity == capacity
    assert (evse.state == state).all()
    assert evse.remaining_standing_time == 3600
    assert evse.soc == 500 * 60 * 60 / capacity
    assert evse.soc_min == 1000 * 60 * 60 / capacity
    assert (interaction == np.array([0,0])).all()

    evse.capacity = 2 * 1000 * 60 * 60 # different capacity
    evse.soc_min = 1000 * 60 * 60 / evse.capacity
    evse.remaining_standing_time = 3600 * 2
    evse.soc = 500 * 60 * 60 / evse.capacity
    state, interaction = evse.transition(1000)
    assert evse.capacity == 2 * 1000 * 60 * 60
    assert (evse.state == state).all()
    assert evse.remaining_standing_time == 3600
    assert evse.soc == 1500 * 60 * 60 / evse.capacity
    assert evse.soc_min == 1000 * 60 * 60 / evse.capacity
    assert (interaction == np.array([-1000,0])).all()
    
    evse = EVSE(900, actions, evse_actions, 1., False)
    evse.capacity = capacity
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 3600
    evse.soc = 500 * 60 * 60 / capacity
    state, interaction = evse.transition(1000)
    assert evse.capacity == capacity
    assert evse.charging_efficiency == 1.
    assert evse._dt == 900
    assert np.isclose(750 * 60 * 60 / capacity, evse.soc)
    assert evse.remaining_standing_time == 2700
    assert evse.soc_min == 1000 * 60 * 60 / capacity
    assert (interaction == np.array([-1000, 0])).all()
    
    # interaction
    evse.soc = 500 * 60 * 60 / capacity
    state, interaction = evse.transition(1000, np.array([250, 500]))
    assert (np.array([-750, 500]) == interaction).all()
    
    # charging efficiency
    evse = EVSE(3600, actions, evse_actions, 0.5, False)
    evse.capacity = capacity
    
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 3600 * 2
    evse.soc = 0 * 60 * 60 / capacity
    state, interaction = evse.transition(1000)
    assert np.isclose(evse.soc, 0.5)
    
    # evse not available
    evse.remaining_standing_time = 0
    state, interaction = evse.transition(0)
    assert evse.remaining_standing_time == 0
    
    # infeasible action    
    exception_raised = False
    try:
        evse.transition(1000)
    except ValueError:
        exception_raised = True
    assert exception_raised
    
    evse = EVSE(900, actions, evse_actions, 1., True)
    evse.capacity = capacity
    
    evse.soc_min = 1000 * 60 * 60 / capacity
    evse.soc_max = 1
    evse.remaining_standing_time = 3600
    evse.soc = 500 * 60 * 60 / capacity
    state, interaction = evse.transition(-1000)
    assert evse.soc == 500 * 60 * 60 / capacity
    assert (interaction == np.array([0, 0])).all()

    state, interaction = evse.transition(200)
    assert np.isclose(evse.soc, 550 * 60 * 60 / capacity)
    assert (interaction == np.array([-200, 0])).all()

    state, interaction = evse.transition(0)
    assert np.isclose(evse.soc, 750 * 60 * 60 / capacity)
    assert (interaction == np.array([-800, 0])).all()

    evse.soc = 1000 * 60 * 60 / capacity
    state, interaction = evse.transition(1000)
    assert np.isclose(evse.soc, 1000 * 60 * 60 / capacity)
    assert (interaction == np.array([0, 0])).all()

    state, interaction = evse.transition(1000)
    assert (interaction == np.array([0, 0])).all()

       # final charging period
    evse = EVSE(900, actions, evse_actions, 1, True)
    evse.capacity = capacity
    evse.soc_min = 1
    evse.soc_max = 1
    evse.remaining_standing_time = 900

    evse.soc = 0.999
    state, interaction = evse.transition(10)
    assert evse.soc == 0.999 # action = 0

    evse.soc = 0.997
    evse.remaining_standing_time = 900
    print(evse.feasible_actions)
    state, interaction = evse.transition(0)
    assert evse.soc == 0.997
    evse.remaining_standing_time = 900
    state, interaction = evse.transition(10)
    assert evse.soc == 0.997 # action = 0

def test_forecast():
    evse = EVSE(3600, actions, evse_actions, 1., False)
    np.random.seed(1924) # floor it
    evse.sample_state()
    forecast, mask = evse.forecast(256)
    assert (evse.state == forecast[0]).all()
    for i in range(len(forecast)):
        #print('{}:{},{},{} vs {}'.format(i,evse.state, evse.forecast_series[0], evse.forecast_mask[0], forecast[i]))
        assert evse.capacity == forecast[i,0]
        assert evse.soc_min == forecast[i,1]
        assert evse.soc_max == forecast[i,2]
        assert evse.remaining_standing_time == forecast[i,4]
        state, interaction = evse.transition(evse.feasible_actions[0])

        assert evse.soc_min < evse.soc_max

        if i+1 < len(forecast):
            if evse.state[0] != forecast[i,0]:
                assert mask[i+1,0] == True
            if evse.state[1] != forecast[i,1]:
                assert mask[i+1,1] == True
            if evse.state[2] != forecast[i,2]:
                assert mask[i+1,2] == True
            if evse.state[4] != forecast[i,4]:
                assert mask[i+1,4] == True

def test_sample_state():
    evse = EVSE(3600, actions, evse_actions, 1., False)
    np.random.seed(1924) # floor it
    evse.train()
    for i in range(1000):
        evse.sample_state()

        assert evse.soc >= 0 and evse.soc <= 1
        assert evse.soc_min >= 0 and evse.soc_min <= 1
        assert evse.soc_max >= 0 and evse.soc_max <= 1

    evse.eval()
    for i in range(1000):
        evse.sample_state()

        assert evse.soc >= 0 and evse.soc <= 1
        assert evse.soc_min >= 0 and evse.soc_min <= 1
        assert evse.soc_max >= 0 and evse.soc_max <= 1