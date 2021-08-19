from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

from ...simulation.individual.hwt import HWT

def factory(model: ConcreteModel, name: str, hwt: HWT, **kwargs):
    def s(key, value):
        setattr(model, name+'_'+key, value)

    def g(key):
        return getattr(model, name+'_'+key)

    s('theta', Var(model.t, within=NonNegativeReals))
    s('b_charging', Var(model.t, within=Binary))
    s('P_pos', Var(model.t, within=NonNegativeReals))
    s('P_neg', Var(model.t, within=NonNegativeReals))
    s('P_th', Var(model.t, within=Reals))

    # minimum and maximum temperature
    def con_temp(model, i):
        return inequality(hwt.ambient_temperature, g('theta')[i], hwt.max_temp)
    s('con_temp', Constraint(model.t, rule=con_temp))
    
    # either charging or discharging
    def con_charging(model, t):
        return g('P_pos')[t] <= g('b_charging')[t] * model.M
    s('con_charging', Constraint(model.t, rule=con_charging))
                                        
    # either charging or discharging
    def con_discharging(model, t):
        return g('P_neg')[t] <= (1 - g('b_charging')[t]) * model.M
    s('con_discharging', Constraint(model.t, rule=con_discharging))

    # power
    def con_power(model, t):
        return g('P_th')[t] == g('P_pos')[t] - g('P_neg')[t]
    s('con_power', Constraint(model.t, rule=con_power))

    # state computation
    def con_state(model, t):
        if t > 0:
            stored_energy = (g('theta')[t-1] - hwt.ambient_temperature) * hwt.tank_ws_per_k
        else:
            stored_energy = hwt.stored_energy

        dQ = model.dt * (g('P_pos')[t] * hwt.charging_efficiency - g('P_neg')[t] / hwt.discharging_efficiency)
        relative_loss_term = (model.dt / 60. / 60) / 2 * hwt.relative_loss
        return g('theta')[t] == (stored_energy * (1 - relative_loss_term) + dQ) / (1 + relative_loss_term) / hwt.tank_ws_per_k + hwt.ambient_temperature
    s('con_state', Constraint(model.t, rule=con_state))

    

    