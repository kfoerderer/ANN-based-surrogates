import pytest

from modules.simulation.integrated.holl import HoLL

import numpy as np


holl = HoLL(3600, HoLL.action_set_100w, 0, True)

def test_state_manipulation():
    actions = holl.actions

    evse = holl.evse
    bess = holl.bess
    chpp = holl.chpp
    hwt_gcb = holl.hwt_gcb
    #el_demand = holl.electricity_demand
    th_demand = holl.heat_demand

    #el_demand.demand = 11
    evse.capacity = 12
    evse.soc_min = 0.2
    evse.soc_max = 0.4
    evse.soc = 0.6
    evse.remaining_standing_time = 13
    bess.stored_energy = 14
    bess.soc_min = 0.1
    bess.soc_max = 0.9
    chpp.mode = 1
    chpp.dwell_time = 15
    chpp.min_off_time = 16
    chpp.min_on_time = 17
    hwt_gcb.mode = 1
    hwt_gcb.temperature = 18
    hwt_gcb.ambient_temperature = 19
    th_demand.demand = 20

    assert (holl.state == np.concatenate((evse.state, 
        bess.state, chpp.state, hwt_gcb.state, holl.heat_demand.state), axis=0)).all()
    print(holl.state)
    assert (holl.state == [12,0.2,0.4,0.6,13,14,0.1,0.9,1,15,16,17,1,18,19,20]).all()
    
    
def test_get_feasible_actions():
    actions = holl.actions 

    evse = holl.evse
    bess = holl.bess
    chpp = holl.chpp
    hwt_gcb = holl.hwt_gcb
    #el_demand = holl.electricity_demand
    th_demand = holl.heat_demand

    ##
    # Battery empty, chpp off
    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.0
    evse.soc_min = 0.55
    evse.soc_max = 0.9
    evse.remaining_standing_time = 3600
    bess.stored_energy = 0
    bess.soc_min = 0
    bess.soc_max = 1
    chpp.mode = 0
    chpp.dwell_time = 900
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 70
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 0

    bess_actions = bess.feasible_actions
    chpp_actions = chpp.feasible_actions
    evse_actions = evse.feasible_actions
    assert ((chpp_actions == 0)).all()
    assert ((bess_actions >= 0) * (bess_actions <= 3*780)).all()
    assert ((evse_actions >= 5500) * (evse_actions <= 9000)).all()
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)   

    ##
    # battery has some charge
    bess.stored_energy = 1000 * 3600  

    bess_actions = bess.feasible_actions
    chpp_actions = [0]
    evse_actions = evse.feasible_actions
    assert ((bess_actions >= 3*-780) * (bess_actions <= 3*780)).all()
    assert ((evse_actions >= 5500) * (evse_actions <= 10000)).all()
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)   

    ## 
    # longer standing time
    bess.stored_energy = 0
    evse.remaining_standing_time = 3600 * 2

    bess_actions = bess.feasible_actions
    chpp_actions = [0]
    evse_actions = evse.feasible_actions
    assert ((bess_actions >= 0) * (bess_actions <= 3*780)).all()
    assert ((evse_actions >= 0) * (evse_actions <= 10000)).all()
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)  

    ##
    # evse needs to be fully charged
    evse.soc = 0.5
    evse.soc_min = 1.0
    evse.soc_max = 1.0
    evse.remaining_standing_time = 3600 * 2

    bess_actions = bess.feasible_actions
    chpp_actions = [0]
    evse_actions = evse.feasible_actions
    assert ((bess_actions >= 0) * (bess_actions <= 3*780)).all()
    assert ((evse_actions >= 0) * (evse_actions <= 10000)).all()
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)   
     
    evse.remaining_standing_time = 3600
    evse_actions = evse.feasible_actions
    assert (evse_actions == actions[np.abs(actions-5000).argmin()]).all()
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible) 
    assert len(expected_feasible) == len(bess_actions) # chpp and evse are already determined

    ##
    # chpp may change mode
    # at first temp. is too high, so it has to stay off
    chpp.dwell_time = 2700

    bess_actions = bess.feasible_actions
    chpp_actions = chpp.feasible_actions
    evse_actions = evse.feasible_actions
    assert ((bess_actions >= 0) * (bess_actions <= 3*780)).all()
    assert len(chpp_actions) == 2
    assert ((chpp_actions <= 0) * (chpp_actions >= -3000)).all()
    assert (evse_actions == actions[np.abs(actions-5000).argmin()]).all()
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in [0] for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)   

    hwt_gcb.temperature = 50 # chpp may now turn on
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)   

    ##
    # every action feasible
    bess.stored_energy = 10000 * 3600
    evse.capacity = 100 * 1000 * 3600
    evse.soc_min = 0.5
    evse.soc_max = 1.0

    bess_actions = bess.feasible_actions
    chpp_actions = chpp.feasible_actions
    evse_actions = evse.feasible_actions
    assert len(bess_actions) == len(bess.actions)
    assert len(chpp_actions) == 2
    assert len(evse_actions) == len(evse.evse_actions)
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)  
    assert (holl.feasible_actions >= actions[np.abs(actions-(-2750-(780*3))).argmin()]).all()
    assert max(holl.feasible_actions) > 24000

    chpp.mode = 1
    chpp_actions = chpp.feasible_actions
    assert len(chpp_actions) == 2
    expected_feasible = np.unique([actions[np.abs(actions-(a+b+c)).argmin()] for a in bess_actions for b in chpp_actions for c in evse_actions])
    assert np.isin(holl.determine_feasible_actions(), expected_feasible).all()
    assert len(holl.feasible_actions) == len(expected_feasible)  
    assert min(holl.feasible_actions) < -7500

def test_state_transition():
    actions = holl.actions
    holl._feasible_actions = None

    evse = holl.evse
    bess = holl.bess
    chpp = holl.chpp
    hwt_gcb = holl.hwt_gcb
    #el_demand = holl.electricity_demand
    th_demand = holl.heat_demand

    ##
    # Battery empty, chpp off
    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.0
    evse.soc_min = 0.55
    evse.soc_max = 1.0
    evse.remaining_standing_time = 3600
    bess.stored_energy = 0
    bess.soc_min = 0
    bess.soc_max = 1
    chpp.mode = 0
    chpp.dwell_time = 900
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 70
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 10000

    th_ws_per_k = hwt_gcb.hwt.tank_ws_per_k
    stored_th_energy = hwt_gcb.hwt.stored_energy
    assert stored_th_energy == th_ws_per_k * (70-20) # before step
    relative_th_loss = hwt_gcb.hwt.relative_loss / 2
    stored_th_energy = stored_th_energy * (1 - relative_th_loss) / (1 + relative_th_loss) + (-10000 * 1 * 3600) / (1 + relative_th_loss) # expected after step

    state, interaction = holl.transition(10700) # max evse (10k) + max bess(700 <= 3*780)
    assert (state == holl.state).all()
    assert bess.stored_energy == 700 * 3600 * bess.charging_efficiency
    assert np.isclose(evse.soc, 1)
    assert chpp.mode == 0
    assert chpp.dwell_time == 900 + 3600
    assert hwt_gcb.mode == 0
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == -10700

    ##
    # battery has some charge, longer evse standing time
    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.0
    evse.soc_min = 0.55
    evse.soc_max = 1.0
    evse.remaining_standing_time = 2 * 3600
    bess.stored_energy = 1000 * 3600  
    chpp.mode = 0
    chpp.dwell_time = 1800
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 70
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 10000

    state, interaction = holl.transition(-700) # evse (0) + min bess(-700 >= -780)
    assert (state == holl.state).all()
    assert bess.stored_energy == (1000 - 700) * 3600
    assert evse.soc == 0
    assert chpp.mode == 0
    assert chpp.dwell_time == 1800 + 3600
    assert hwt_gcb.mode == 0
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == 700

    ##
    # evse needs to be fully charged
    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.989 # 110 Wh missing (-> 100 W Action)
    evse.soc_min = 1
    evse.soc_max = 1.0
    evse.remaining_standing_time = 2 * 3600
    bess.stored_energy = 1000 * 3600  
    chpp.mode = 0
    chpp.dwell_time = 1800
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 70
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 10000

    state, interaction = holl.transition(200) # evse (0) + bess (200W)
    assert (state == holl.state).all()
    assert bess.stored_energy == (1000 + 200 * bess.charging_efficiency) * 3600
    assert evse.soc == 0.989
    assert evse.remaining_standing_time == 3600
    assert chpp.mode == 0
    assert chpp.dwell_time == 1800 + 3600
    assert hwt_gcb.mode == 0
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == -200

    # chpp may now change the mode but the temp. is too high, so it has to stay off
    state, interaction = holl.transition(200) # evse (0) + bess(200) + chpp (0)
    assert (state == holl.state).all()
    assert bess.stored_energy == (1000 + (200 + 200) * bess.charging_efficiency) * 3600
    assert evse.soc == 0.989
    assert evse.remaining_standing_time == 0
    assert chpp.mode == 0
    assert chpp.dwell_time == 1800 + 2 * 3600
    assert hwt_gcb.mode == 0
    assert interaction[0] == -200

    ##
    # chpp may change mode
    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.5
    evse.soc_min = 0.45
    evse.soc_max = 1.0
    evse.remaining_standing_time = 3600
    bess.stored_energy = 1000 * 3600
    chpp.mode = 0
    chpp.dwell_time = 2700
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 45
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 10000

    th_ws_per_k = hwt_gcb.hwt.tank_ws_per_k
    stored_th_energy = hwt_gcb.hwt.stored_energy
    assert stored_th_energy == th_ws_per_k * (45-20) # before step
    stored_th_energy = stored_th_energy * (1 - relative_th_loss) / (1 + relative_th_loss) + ((-10000 + 12500/2)  * 1 * 3600) / (1 + relative_th_loss) # expected after step

    state, interaction = holl.transition(-2750 + 5000 + 700) # chpp, evse, bess
    assert (state == holl.state).all()
    assert bess.stored_energy == (1000 + 700 * bess.charging_efficiency ) * 3600
    assert np.isclose(evse.soc, 1)
    assert chpp.mode == 1
    assert chpp.dwell_time == 3600
    assert hwt_gcb.mode == 0
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == -(-2800 + 5000 + 700)

    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.5
    evse.soc_min = 0.45
    evse.soc_max = 1.0
    evse.remaining_standing_time = 3600
    bess.stored_energy = 1000 * 3600
    chpp.mode = 0
    chpp.dwell_time = 2700
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 45
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 10000

    state, interaction = holl.transition(-2750 + 5000 + 500) # chpp, evse, bess
    assert (state == holl.state).all()
    assert bess.stored_energy == (1000 + 500 * bess.charging_efficiency ) * 3600
    assert np.isclose(evse.soc, 1)
    assert chpp.mode == 1
    assert chpp.dwell_time == 3600
    assert hwt_gcb.mode == 0
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == -(-2800 + 5000 + 500)

    ##
    # gcb needs to turn on
    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.5
    evse.soc_min = 0.45
    evse.soc_max = 1.0
    evse.remaining_standing_time = 3600
    bess.stored_energy = 0
    chpp.mode = 0
    chpp.dwell_time = 2700
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 35
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 10000

    th_ws_per_k = hwt_gcb.hwt.tank_ws_per_k
    stored_th_energy = hwt_gcb.hwt.stored_energy
    assert stored_th_energy == th_ws_per_k * (35-20) # before step
    stored_th_energy = stored_th_energy * (1 - relative_th_loss) / (1 + relative_th_loss) + ((-10000 + 12500/2 + 38500)  * 1 * 3600) / (1 + relative_th_loss) # expected after step

    state, interaction = holl.transition(800)
    assert (state == holl.state).all()
    assert bess.stored_energy == 0
    assert np.isclose(evse.soc, 0.5 + 0.36)
    assert chpp.mode == 1
    assert chpp.dwell_time == 3600
    assert hwt_gcb.mode == 1
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == -(-2800 + 3600)
    assert interaction[1] == 0

    #el_demand.demand = 30000
    evse.capacity = 10 * 1000 * 3600
    evse.soc = 0.4
    evse.soc_min = 0.45
    evse.soc_max = 1.0
    evse.remaining_standing_time = 3600
    bess.stored_energy = 0
    chpp.mode = 0
    chpp.dwell_time = 2700
    chpp.min_off_time = 2700
    chpp.min_on_time = 2700
    hwt_gcb.mode = 0
    hwt_gcb.temperature = 35
    hwt_gcb.ambient_temperature = 20
    th_demand.demand = 10000

    state, interaction = holl.transition(3600) # chpp -2800, evse 6000, bess 400 
    assert (state == holl.state).all()
    assert np.isclose(bess.stored_energy, 400 * 3600 * bess.charging_efficiency)
    assert np.isclose(evse.soc, 1)
    assert chpp.mode == 1
    assert chpp.dwell_time == 3600
    assert hwt_gcb.mode == 1
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == -3600
    assert interaction[1] == 0

    th_demand.demand = 10000
    stored_th_energy = stored_th_energy * (1 - relative_th_loss) / (1 + relative_th_loss) + ((-10000 + 12500 + 60000)  * 1 * 3600) / (1 + relative_th_loss) # expected after step

    state, interaction = holl.transition(-5500)
    assert chpp.mode == 1
    assert chpp.dwell_time == 2 * 3600
    assert hwt_gcb.mode == 1
    assert np.isclose(hwt_gcb.temperature, stored_th_energy/th_ws_per_k+20)
    assert interaction[0] == 5500
    assert interaction[1] == 0 

def test_forecast_transition_interplay():
    holl2 = HoLL(3600, HoLL.action_set_100w, 0, True)

    holl.eval()
    holl2.eval()

    np.random.seed(1924)
    for i in range(5):
        holl.sample_state()
        forecast, mask = holl.forecast(48)

        for j in range(47):
            holl2.state = np.copy(holl.state)
            feasible = holl.feasible_actions
            feasible2 = holl2.feasible_actions
            assert np.array_equal(feasible, feasible2)
            action = np.random.choice(feasible)
            state, interaction = holl.transition(action)
            state2, interaction2 = holl2.transition(action)
            state2 = state2 * (1-mask[j+1]) + forecast[j+1] * mask[j+1]
            assert np.isclose(state, state2).all()
            assert np.isclose(interaction, interaction2).all()
        


#"""