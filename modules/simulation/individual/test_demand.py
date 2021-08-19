import pytest

from modules.simulation.individual.demand import Demand

import numpy as np

def test_forecast():
    # general
    demand = Demand(900, np.arange(96).reshape(1,-1), seconds_per_value=900)

    assert demand.state == np.zeros(1)

    forecast, mask = demand.forecast(48)    
    assert (forecast == np.arange(48).reshape(-1,1)).all()
    assert (mask == np.ones(48).reshape(-1,1)).all()

    forecast, mask = demand.forecast(10)    
    assert (forecast == np.arange(10).reshape(-1,1)).all()
    assert (mask == np.ones(10).reshape(-1,1)).all()

    forecast, mask = demand.forecast(96)    
    assert (forecast == np.arange(96).reshape(-1,1)).all()
    assert (mask == np.ones(96).reshape(-1,1)).all()

    # sampling
    demand = Demand(3600, np.arange(96).reshape(1,-1), seconds_per_value=900)

    np.random.seed(1924)
    initial_states = []
    for i in range(10):
        state = demand.sample_state()
        initial_states.append(state)
        forecast, mask = demand.forecast(24)
        assert forecast[0] == state

        window_position = int(state / 4) * 4
        for step in range(1, forecast.shape[0]):
            window_position = (window_position + 4) % 96
            assert np.isin(forecast[step], window_position + np.arange(4)) 

    assert np.unique(initial_states).shape[0] > 1

def test_transition():
    demand = Demand(900, np.arange(96).reshape(1,-1), seconds_per_value=900)

    assert demand.state == np.zeros(1)
    forecast, mask = demand.forecast(48)

    state, interaction = demand.transition(900)  
    assert forecast[1] == state
    assert interaction[1] == 0
    forecast2, mask2 = demand.forecast(47)
    assert (forecast[1:] == forecast2).all()

    state, interaction = demand.transition(900)  
    assert forecast[2] == state
    assert interaction[1] == -1
    forecast2, mask2 = demand.forecast(46)
    assert (forecast[2:] == forecast2).all()
