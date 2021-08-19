from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

from ...simulation.individual.bess import BESS

def factory(model: ConcreteModel, name: str, bess: BESS, **kwargs):
    def s(key, value):
        setattr(model, name+'_'+key, value)

    def g(key):
        return getattr(model, name+'_'+key)

    s('b_charging', Var(model.t, within=Binary))
    s('P_pos', Var(model.t, within=NonNegativeReals))
    s('P_neg', Var(model.t, within=NonNegativeReals))
    s('Q', Var(model.t, within=NonNegativeReals))
    s('P_el', Var(model.t, within=Reals))

    # minimum and maximum charge
    def con_energy(model, i):
        return inequality(0, g('Q')[i], bess.capacity)
    s('con_energy', Constraint(model.t, rule=con_energy))

    # either charging or discharging
    def con_charging(model, i):
        return g('P_pos')[i] <= g('b_charging')[i] * max(bess.actions)
    s('con_charging', Constraint(model.t, rule=con_charging))
                                        
    # either charging or discharging
    def con_discharging(model, i):
        return g('P_neg')[i] <= (1 - g('b_charging')[i]) * abs(min(bess.actions))
    s('con_discharging', Constraint(model.t, rule=con_discharging))

    # state computation
    def con_state(model, i):         
        dQ = model.dt * (g('P_pos')[i] * bess.charging_efficiency - g('P_neg')[i] / bess.discharging_efficiency)
        relative_loss_term = (model.dt / 60. / 60) / 2 * bess.relative_loss
        if i > 0:
            return g('Q')[i] == (g('Q')[i-1] * (1 - relative_loss_term) + dQ) / (1 + relative_loss_term)
        else:
            return g('Q')[i] == (bess.stored_energy * (1 - relative_loss_term) + dQ) / (1 + relative_loss_term)
    s('con_state', Constraint(model.t, rule=con_state))

    # power
    def con_power(model, i):
        return g('P_el')[i] == g('P_pos')[i] - g('P_neg')[i]
    s('con_power', Constraint(model.t, rule=con_power))

    