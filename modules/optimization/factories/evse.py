from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

from ...simulation.individual.evse import EVSE

def factory(model: ConcreteModel, name: str, evse: EVSE, **kwargs):
    def s(key, value):
        setattr(model, name+'_'+key, value)

    def g(key):
        return getattr(model, name+'_'+key)

    s('b_charging', Var(model.t, within=Binary))
    s('Q', Var(model.t, within=NonNegativeReals))
    s('P_el', Var(model.t, within=Reals))

    actions = evse.evse_actions
    min_charging = min(actions[actions > 0])
    max_charging = max(actions[actions > 0])

    forecast, forecast_mask = evse.forecast(max(model.t)+1)

    if evse._training:
        constraint_fuzziness = 0
    else:
        constraint_fuzziness = evse.constraint_fuzziness

    # minimum and maximum charge
    def con_energy(model, t):
        max_charge = forecast[t][0] * max(forecast[t][3], forecast[t][2] + constraint_fuzziness) # soc may already be higher than the target (-> do nothing)
        min_charge = max(0, forecast[t][0] * (forecast[t][1] - constraint_fuzziness) - max(float(forecast[t][4])-model.dt,0) * max_charging) # Q is shifted by 1, Q_0 is the state at t=1
        return inequality(min_charge, g('Q')[t], max_charge)
    s('con_energy', Constraint(model.t, rule=con_energy))

    # min. charging power (integrating min and max causes issues with pyomo)
    def con_charging_min(model, t):
        bev_present = (forecast[t][4] > 0) * 1 # assuming the standing time is a multiple of dt (which should always be the case)
        return (bev_present * min_charging) * g('b_charging')[t] <= g('P_el')[t]
    s('con_charging_min', Constraint(model.t, rule=con_charging_min))

    # max. charging power (integrating min and max causes issues with pyomo)
    def con_charging_max(model, t):
        bev_present = (forecast[t][4] > 0) * 1 # assuming the standing time is a multiple of dt (which should always be the case)
        return g('P_el')[t] <= (bev_present * max_charging) * g('b_charging')[t]
    s('con_charging_max', Constraint(model.t, rule=con_charging_max))

    # state computation
    def con_state(model, t):    
        dQ = model.dt * g('P_el')[t] * evse.charging_efficiency
        if t > 0 and forecast[t][4] < forecast[t-1][4]: # don't use mask here (already used to force fixed inputs)
            return g('Q')[t] == (g('Q')[t-1] + dQ) # previous charging process 
        else:
            return g('Q')[t] == (forecast[t][0] * forecast[t][3] + dQ) # new charging process
    s('con_state', Constraint(model.t, rule=con_state))


    