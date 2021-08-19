from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

from ..constraintfactory import ConstraintFactory
from ...simulation.integrated.chpp_hwt import CHPP_HWT

def factory(model: ConcreteModel, name: str, chpp_hwt: CHPP_HWT, **kwargs):
    def s(key, value, name=name):
        setattr(model, name+'_'+key, value)

    def g(key, name=name):
        return getattr(model, name+'_'+key)

    hwt = name+'_hwt'
    chpp = name+'_chpp'

    heat_demand_series, _ = chpp_hwt.demand.forecast(max(model.t)+1)

    ConstraintFactory.add_to_model(model, hwt, chpp_hwt.hwt)
    ConstraintFactory.add_to_model(model, chpp, chpp_hwt.chpp, soft_dwell_time_constraints=True)
    
    s('P_el', Var(model.t, within=Reals))

    # el. Power
    def con_el_power(model, t):
        return g('P_el')[t] == g('P_el', chpp)[t]
    s('con_el_power', Constraint(model.t, rule=con_el_power))

    # th. Power
    def con_th_power(model, t):
        return g('P_th', hwt)[t] == - (g('P_th', chpp)[t] + float(heat_demand_series[t]))
    s('con_th_power', Constraint(model.t, rule=con_th_power))

    fuzziness = chpp_hwt.constraint_fuzziness
    min_temp = chpp_hwt.hwt.soft_min_temp
    max_temp = chpp_hwt.hwt.soft_max_temp

    # allow (de-)activation
    #
    def con_allow_activate_on_low_temp(model, t):
        if t == 0:
            return g('b_allow_on', chpp)[t] == (chpp_hwt.hwt.temperature < min_temp + fuzziness*100) * 1
        return (min_temp + fuzziness*100) - g('theta',hwt)[t-1] <= model.M * g('b_allow_on', chpp)[t]
    s('con_allow_activate_on_low_temp', Constraint(model.t, rule=con_allow_activate_on_low_temp))

    def con_lower_temp_satisfied(model, t):
        if t == 0: 
            # == 0 is captured above
            return Constraint.Skip
        return -(min_temp + fuzziness*100) + g('theta',hwt)[t-1] <= model.M * (1-g('b_allow_on', chpp)[t])
    s('con_lower_temp_satisfied', Constraint(model.t, rule=con_lower_temp_satisfied))
    
    def con_allow_deactivate_on_high_temp(model, t):
        if t == 0:
            return g('b_allow_off', chpp)[t] == (chpp_hwt.hwt.temperature > max_temp - fuzziness*100) * 1
        return g('theta', hwt)[t-1] - (max_temp - fuzziness*100) <= model.M * g('b_allow_off', chpp)[t]
    s('con_allow_deactivate_on_high_temp', Constraint(model.t, rule=con_allow_deactivate_on_high_temp))

    def con_upper_temp_satisfied(model, t):
        if t == 0: 
            # == 0 is captured above
            return Constraint.Skip
        return -g('theta',hwt)[t-1] + (max_temp - fuzziness*100) <= model.M * (1-g('b_allow_off', chpp)[t])
    s('con_upper_temp_satisfied', Constraint(model.t, rule=con_upper_temp_satisfied))
    
    # force (de-)activation
    #
    def con_activate_on_low_temp(model, t):
        if t == 0:
            return g('b_force_on', chpp)[t] == (chpp_hwt.hwt.temperature < min_temp-fuzziness*100) * 1
        return (min_temp - fuzziness*100) - g('theta',hwt)[t-1] <= model.M * g('b_force_on', chpp)[t]
    s('con_activate_on_low_temp', Constraint(model.t, rule=con_activate_on_low_temp))

    def con_force_on_not_allowed(model, t):
        return g('b_force_on', chpp)[t] <= g('b_allow_on', chpp)[t]
    s('con_force_on_not_allowed', Constraint(model.t, rule=con_force_on_not_allowed))
    
    def con_deactivate_on_high_temp(model, t):
        if t == 0:
            return g('b_force_off', chpp)[t] == (chpp_hwt.hwt.temperature > max_temp+fuzziness*100) * 1
        return  g('theta',hwt)[t-1] - (max_temp + fuzziness*100) <= model.M * g('b_force_off', chpp)[t]
    s('con_deactivate_on_high_temp', Constraint(model.t, rule=con_deactivate_on_high_temp))

    def con_force_off_not_allowed(model, t):
        return g('b_force_off', chpp)[t] <= g('b_allow_off', chpp)[t]
    s('con_force_off_not_allowed', Constraint(model.t, rule=con_force_off_not_allowed))
