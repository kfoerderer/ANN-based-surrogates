from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

from ..constraintfactory import ConstraintFactory

from ...simulation.integrated.holl import HoLL

def factory(model: ConcreteModel, name: str, holl: HoLL, **kwargs):
    # evse, bess and chpp/hwt/gcb/demand are independent of each other
    # => use the factories for the individual system

    evse = name+'_evse'
    bess = name+'_bess'
    chpp = name+'_chpp'
    hwt_gcb = name+'_hwtGcb'
    hwt = hwt_gcb + '_hwt'
    
    heat_demand_series, _ = holl.heat_demand.forecast(len(model.t))

    ConstraintFactory.add_to_model(model, evse, holl.evse)
    ConstraintFactory.add_to_model(model, bess, holl.bess)
    ConstraintFactory.add_to_model(model, chpp, holl.chpp, soft_dwell_time_constraints=True)
    ConstraintFactory.add_to_model(model, hwt_gcb, holl.hwt_gcb)

    def s(key, value, name=name):
        setattr(model, name+'_'+key, value)

    def g(key, name=name):
        return getattr(model, name+'_'+key)
    
    s('P_el', Var(model.t, within=Reals))

    # power
    def con_power(model, i):
        return g('P_el')[i] == g('P_el', evse)[i] + g('P_el', bess)[i] + g('P_el', chpp)[i]
    s('con_power', Constraint(model.t, rule=con_power))

    # th. Power
    def con_th_power(model, t):
        return g('P_th', hwt_gcb)[t] == -g('P_th', chpp)[t] - float(heat_demand_series[t])
    s('con_th_power', Constraint(model.t, rule=con_th_power))

    ##
    # CHPP control
    fuzziness = holl.constraint_fuzziness
    max_temp = holl.hwt_gcb.hwt.soft_max_temp

    def con_allow_deactivate_on_high_temp(model, t):
        if t > 0:
            return g('theta', hwt)[t-1] - (max_temp - fuzziness*100) <= model.M * g('b_allow_off', chpp)[t]
        return g('b_allow_off', chpp)[t] == (holl.hwt_gcb.temperature > max_temp - fuzziness*100) * 1
    s('con_allow_deactivate_on_high_temp', Constraint(model.t, rule=con_allow_deactivate_on_high_temp))

    def con_upper_temp_satisfied(model, t):
        if t > 0: 
            return -g('theta',hwt)[t-1] + (max_temp - fuzziness*100) <= model.M * (1-g('b_allow_off', chpp)[t])
        # == 0 is captured above
        return Constraint.Skip
    s('con_upper_temp_satisfied', Constraint(model.t, rule=con_upper_temp_satisfied))

    def con_deactivate_on_high_temp(model, t):
        if t > 0:
            return  g('theta',hwt)[t-1] - (max_temp + fuzziness*100) <= model.M * g('b_force_off', chpp)[t]
        return g('b_force_off', chpp)[t] == (holl.hwt_gcb.temperature > max_temp+fuzziness*100) * 1
    s('con_deactivate_on_high_temp', Constraint(model.t, rule=con_deactivate_on_high_temp))

    def con_force_off_not_allowed(model, t):
        return g('b_force_off', chpp)[t] <= g('b_allow_off', chpp)[t]
    s('con_force_off_not_allowed', Constraint(model.t, rule=con_force_off_not_allowed))

    def con_never_allow_on(model, t):
        return g('b_allow_on', chpp)[t] == 0
    s('con_never_allow_on', Constraint(model.t, rule=con_never_allow_on))

    def con_never_force_on(model, t):
        return g('b_force_on', chpp)[t] == 0
    s('con_never_force_on', Constraint(model.t, rule=con_never_force_on))

