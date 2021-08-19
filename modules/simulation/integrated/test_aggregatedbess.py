import pytest

from modules.simulation.integrated.aggregated_bess import AggregatedBESS

import numpy as np

actions = AggregatedBESS.create_action_set(-10000, 10000, 401)
# 10k in steps of 50W (i.e. 200 steps)
#
#   5*1k + 5*0.1k = 5.5k reachable = 110 steps
#

def test_state_manipulation():
    capacities = [1000 * 60 * 60] * 5 + [100 * 60 * 60] * 5
    max_charging_powers = [1000] * 5 + [100] * 5
    max_discharging_powers = [-1000] * 5 + [-100] * 5
    charging_efficiencies = [1] * 10 
    discharging_efficiencies = [1] * 10
    relative_losses = [0] * 10
    
    bess = AggregatedBESS(900, actions, capacities, max_charging_powers, max_discharging_powers, charging_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    bess.stored_energy = np.arange(10)/10. * capacities
    assert np.isclose(np.arange(10)/10. * capacities, bess.stored_energy).all()
    assert np.isclose(bess.state_of_charge, np.arange(10)/10).all()

    bess.stored_energy = (1-np.arange(10)/10.) * capacities
    assert np.isclose((1-np.arange(10)/10.) * capacities, bess.stored_energy).all()
    assert np.isclose(bess.state_of_charge, (1-np.arange(10)/10)).all()

def test_get_feasible_actions():
    # general
    capacities = np.array([1000 * 60 * 60] * 5 + [100 * 60 * 60] * 5)
    max_charging_powers = [1000] * 5 + [100] * 5
    max_discharging_powers = [-1000] * 5 + [-100] * 5
    charging_efficiencies = [1] * 10 
    discharging_efficiencies = [1] * 10
    relative_losses = [0] * 10
    bess = AggregatedBESS(900, actions, capacities, max_charging_powers, max_discharging_powers, charging_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    
    bess.stored_energy = capacities
    assert bess.feasible_actions.shape[0] == 111
    assert (bess.feasible_actions <= 0).all()

    bess.stored_energy = capacities * 0.5
    assert bess.feasible_actions.shape[0] == 221

    bess.stored_energy = capacities * 0
    assert bess.feasible_actions.shape[0] == 111
    assert (bess.feasible_actions >= 0).all()

    bess.stored_energy = np.array([capacities[0]] + [0] * 9)
    assert bess.feasible_actions.shape[0] == 1 + 20 + 90
    assert (bess.feasible_actions >= -1000).all()
    assert (bess.feasible_actions <= 4500).all()

    bess.stored_energy = np.array([capacities[0] * 0.5] + [0] * 8 + [capacities[-1] * 1])
    assert bess.feasible_actions.shape[0] == 1 + 40 + 88 + 2
    assert (bess.feasible_actions >= -1100).all()
    assert (bess.feasible_actions <= 5400).all()

    # eval()
    bess = AggregatedBESS(3600, actions, capacities, max_charging_powers, max_discharging_powers, charging_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    bess.soc_min = 0.9
    bess.stored_energy = capacities * 0
    assert bess.feasible_actions.shape[0] == 12 # 4950 W up to 5500 W
    assert (bess.feasible_actions >= 4950).all()
    assert (bess.feasible_actions <= 5500).all()

    bess.constraint_fuzziness = 0.1
    bess._feasible_actions = None
    assert bess.feasible_actions.shape[0] == 23
    assert (bess.feasible_actions >= 4400).all()
    assert (bess.feasible_actions <= 5500).all()

    bess.train()
    assert bess.feasible_actions.shape[0] == 12 # 4950 W up to 5500 W
    assert (bess.feasible_actions >= 4950).all()
    assert (bess.feasible_actions <= 5500).all()

    bess.stored_energy = capacities * 0.5
    bess.soc_min = 0.5
    bess.soc_max = 0.6
    assert bess.determine_feasible_actions().shape[0] == 12
    assert (bess.feasible_actions <= 550).all()
    assert (bess.feasible_actions >= 0).all()

    bess.eval()
    assert bess.determine_feasible_actions().shape[0] == 1 + 11 + 22
    assert (bess.feasible_actions <= 1100).all()
    assert (bess.feasible_actions >= -550).all()

    bess.soc_max = 1
    bess.soc_min = 0
    bess.constraint_fuzziness = 0

    # charging efficiency
    charging_efficiencies = [0.5] * 5 + [0.25] * 5
    bess = AggregatedBESS(3600, actions, capacities, max_charging_powers, max_discharging_powers, charging_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    
    bess.stored_energy = capacities * 0.9
    assert bess.feasible_actions.shape[0] == 100 + 20 + 4  # 500 W + 50 W (without losses) => 1000 W + 200 W
    
    bess.stored_energy = capacities * 0.95
    assert bess.feasible_actions.shape[0] == 105 + 12
    
    # discharging efficiency
    charging_efficiencies = [1] * 10 
    discharging_efficiencies = [0.5] * 5 + [0.25] * 5
    bess = AggregatedBESS(3600, actions, capacities, max_charging_powers, max_discharging_powers, charging_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    
    bess.stored_energy = capacities * 0.1 # 550 Wh in total
    assert bess.feasible_actions.shape[0] == 100 + 5  # 500 W + 50 W (without losses) => 250 W + 12.5 W
    
    # loss
    discharging_efficiencies = [1] * 10
    relative_losses = [1] * 5 + [0.5] * 5
    bess = AggregatedBESS(3600, actions, capacities, max_charging_powers, max_discharging_powers, charging_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    
    bess.stored_energy = capacities
    # new state 0% => 50% energy lost & 25% energy lost => 5*500 + 5*75 = 2500 + 375 = 2875
    # new state 100% => 100% energy lost & 50% energy lost => 5*1000 + 5*50 = 5250
    print(bess.feasible_actions)
    assert (bess.feasible_actions <= 5250).all()
    assert (bess.feasible_actions >= -2850).all()
    assert bess.feasible_actions.shape[0] == 57 + 1 + 105 # 2500W + 500 W

def test_state_transition():
    # general, feasible action
    capacities = np.array([1000 * 60 * 60] * 5 + [100 * 60 * 60] * 5)
    max_charging_powers = [250] * 5 + [100] * 5
    max_discharging_powers = [-250] * 5 + [-100] * 5
    charging_efficiencies = [1] * 10 
    discharging_efficiencies = [1] * 10
    relative_losses = [0] * 10
    bess = AggregatedBESS(3600, actions, capacities, max_charging_powers, max_discharging_powers, charging_efficiencies, discharging_efficiencies, relative_losses, True, 0)
    
    bess.stored_energy = capacities * 0.5
    state, interaction = bess.transition(0)
    assert (bess.state == state).all()
    assert (capacities * 0.5 == bess.stored_energy).all()
    assert (np.array([0,0]) == interaction).all()
    
    state, interaction = bess.transition(100)
    assert np.isclose(capacities * 0.5 + ([250*100/(250*5+50*5)*3600] * 5 + [50*100/(250*5+50*5)*3600] * 5), bess.stored_energy).all()
    assert (np.array([-100,0]) == interaction).all()
    
    bess.stored_energy = np.concatenate((capacities[:5] * 0.5, capacities[5:] * 0.1))
    state, interaction = bess.transition(100)
    assert np.isclose(([500*3600 + 250*100/(250*5+90*5)*3600] * 5 + [10*3600 + 90*100/(250*5+90*5)*3600] * 5), bess.stored_energy).all() # max 90 W for small BESS
    assert (np.array([-100,0]) == interaction).all()

    bess.stored_energy = np.array([500*3600] * 5 + [30 * 3600] * 5)
    state, interaction = bess.transition(200)
    assert np.isclose(([500*3600 + 200*250/(250*5+70*5)*3600] * 5 + [30*3600 + 200*70/(250*5+70*5)*3600] * 5), bess.stored_energy).all() # charging for small BESS is limited to 70 by SOC
    assert (np.array([-200,0]) == interaction).all()

    bess.stored_energy = np.array([500*3600] * 5 + [10 * 3600] * 5)
    state, interaction = bess.transition(-1000)
    assert np.isclose(([500*3600 + -1000*250/(250*5+10*5)*3600] * 5 + [10*3600 + -1000*10/(250*5+10*5)*3600] * 5), bess.stored_energy).all()
    assert (np.isclose(np.array([1000,0]), interaction)).all()

    bess.stored_energy = np.array([500*3600] * 5 + [40 * 3600] * 5)
    state, interaction = bess.transition(-1000)
    # 100W are needed to reach 0.4 SOC
    # smaller BESS is limited by its SOC of 0.4
    assert np.isclose(([500*3600 - 1000*250/(250*5+40*5)*3600] * 5 + [40*3600 - 1000*40/(250*5+40*5)*3600] * 5), bess.stored_energy).all() 
    assert (np.isclose(np.array([1000,0]), interaction)).all()

    bess.stored_energy = capacities * 0
    state, interaction = bess.transition(250 * 5 + 100 * 5)
    assert np.isclose(([250*3600] * 5 + [100*3600] * 5), bess.stored_energy).all()
    assert (np.array([-350*5,0]) == interaction).all()

    state, interaction = bess.transition(-350*5)
    assert np.isclose(([0] * 10), bess.stored_energy).all()
    assert (np.array([350*5,0]) == interaction).all()

    bess.stored_energy = capacities
    state, interaction = bess.transition(-250 * 5 - 100 * 5)
    assert np.isclose(([750*3600] * 5 + [0*3600] * 5), bess.stored_energy).all()
    assert (np.array([350*5,0]) == interaction).all()

    state, interaction = bess.transition(+350*5)
    assert np.isclose(capacities, bess.stored_energy).all()
    assert (np.array([-350*5,0]) == interaction).all()
