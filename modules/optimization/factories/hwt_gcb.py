from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

import math

from ..constraintfactory import ConstraintFactory
from ...simulation.integrated.hwt_gcb import HWT_GCB

def factory(model: ConcreteModel, name: str, hwt_gcb: HWT_GCB, **kwargs):
    def s(key, value, name=name):
        setattr(model, name+'_'+key, value)

    def g(key, name=name):
        return getattr(model, name+'_'+key)

    hwt = name+'_hwt'
    gcb = name+'_gcb'

    # add HWT model
    ConstraintFactory.add_to_model(model, hwt, hwt_gcb.hwt)

    ##
    # build GCB model
    state_matrix = hwt_gcb.state_matrix
    modes_count = len(state_matrix)

    # state
    s('modes', Set(initialize=list(range(modes_count))), gcb)
    for i in range(modes_count):
        for j in range(modes_count):
            s('b_mode_{}_{}'.format(i,j), Var(model.t, within=Binary), gcb)
    s('b_off', Var(model.t, within=Binary), gcb)

    # power
    s('P_th', Var(model.t, within=Reals), gcb)

    # control
    s('b_start', Var(model.t, within=Binary), gcb)
    s('b_stop', Var(model.t, within=Binary), gcb)

    # only one mode at the same time
    def con_distinct_mode(model, t):
        total = 0
        for i in range(modes_count):
            for j in range(modes_count):
                total += g('b_mode_{}_{}'.format(i,j), gcb)[t]
        return total == 1
    s('con_distinct_mode', Constraint(model.t, rule=con_distinct_mode), gcb)

    # a mode can only be left if it was entered before
    def con_mode_continuity(model, mode, t):
        # determine if the previous transition lead to 'mode' (=1)
        if t == 0:
            if hwt_gcb.mode == mode:
                previous = 1
            else:
                previous = 0
        else:    
            previous = 0
            for i in range(modes_count):
                previous += g('b_mode_{}_{}'.format(i,mode), gcb)[t-1]
        # does the next transition start in 'mode' (=1)?
        current = 0
        for i in range(modes_count):
            current += g('b_mode_{}_{}'.format(mode,i), gcb)[t]
        return previous == current
    s('con_mode_continuity', Constraint(g('modes', gcb), model.t, rule=con_mode_continuity), gcb)

    # make the model more readble by introducing b_off
    def con_detect_off_state(model, t):
        return g('b_off', gcb)[t] == sum(g('b_mode_{}_{}'.format(i, 0), gcb)[t] for i in range(modes_count))
    s('con_detect_off_state', Constraint(model.t, rule=con_detect_off_state), gcb)

    # th. Power
    def con_th_power(model, t):
        total = 0
        for i in range(modes_count):
            for j in range(modes_count):
                total += g('b_mode_{}_{}'.format(i,j), gcb)[t] * state_matrix[i][j][1]
        return g('P_th', gcb)[t] == total
    s('con_th_power', Constraint(model.t, rule=con_th_power), gcb)

    ##
    # connect HWT and GCB
    # (hysteresis control)

    hwt_model = hwt_gcb.hwt
    min_temp = hwt_model.soft_min_temp
    max_temp = hwt_model.soft_max_temp

    def con_low_temp_reached(model, t):
        if t == 0:
            return g('b_start', gcb)[t] == (hwt_model.temperature < min_temp) * 1
        return (min_temp) - g('theta', hwt)[t-1] <= model.M * g('b_start', gcb)[t]
    s('con_low_temp_reached', Constraint(model.t, rule=con_low_temp_reached))

    def con_lower_temp_satisfied(model, t):
        if t == 0: 
            # == 0 is captured above
            return Constraint.Skip
        return -(min_temp) + g('theta', hwt)[t-1] <= model.M * (1-g('b_start', gcb)[t])
    s('con_lower_temp_satisfied', Constraint(model.t, rule=con_lower_temp_satisfied))
    
    def con_high_temp_reached(model, t):
        if t == 0:
            return g('b_stop', gcb)[t] == (hwt_model.temperature > max_temp) * 1
        return g('theta', hwt)[t-1] - (max_temp) <= model.M * g('b_stop', gcb)[t]
    s('con_high_temp_reached', Constraint(model.t, rule=con_high_temp_reached))

    def con_upper_temp_satisfied(model, t):
        if t == 0: 
            # == 0 is captured above
            return Constraint.Skip
        return -g('theta', hwt)[t-1] + (max_temp) <= model.M * (1-g('b_stop', gcb)[t])
    s('con_upper_temp_satisfied', Constraint(model.t, rule=con_upper_temp_satisfied))

    # stop on upper limit
    def con_stop(model, t):
        return g('b_off', gcb)[t] >= g('b_stop', gcb)[t]
    s('con_stop', Constraint(model.t, rule=con_stop))

    # start on lower limit
    def con_start(model, t):
        return (1-g('b_off', gcb)[t]) >= g('b_start', gcb)[t]
    s('con_start', Constraint(model.t, rule=con_start))

    # do not start until lower limit is reached
    def con_remain_stopped(model, t):
        if t > 0:
            return g('b_off', gcb)[t] >= g('b_off', gcb)[t-1] - g('b_start', gcb)[t]
        return g('b_off', gcb)[t] >= 1* (hwt_gcb.mode == 0) - g('b_start', gcb)[t]
    s('con_remain_stopped', Constraint(model.t, rule=con_remain_stopped))

    # do not stop until upper limit is reached
    def con_remain_running(model, t):
        if t > 0:
            return (1-g('b_off', gcb)[t]) >= (1-g('b_off', gcb)[t-1]) - g('b_stop', gcb)[t]
        return (1-g('b_off', gcb)[t]) >= 1* (hwt_gcb.mode != 0) - g('b_stop', gcb)[t]
    s('con_remain_running', Constraint(model.t, rule=con_remain_running))

    ##
    # total power
    s('P_el', Var(model.t, within=Reals))
    s('P_th', Var(model.t, within=Reals))

    # el. Power
    def con_total_el_power(model, t):
        return g('P_el')[t] == 0
    s('con_total_el_power', Constraint(model.t, rule=con_total_el_power))

    # th. Power
    def con_total_th_power(model, t):
        return g('P_th', hwt)[t] == -g('P_th', gcb)[t] + g('P_th')[t]
    s('con_total_th_power', Constraint(model.t, rule=con_total_th_power))

