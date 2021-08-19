from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

import math

from ...simulation.individual.chpp import CHPP

def factory(model: ConcreteModel, name: str, chpp: CHPP, soft_dwell_time_constraints=False, **kwargs):
    def s(key, value):
        setattr(model, name+'_'+key, value)

    def g(key):
        return getattr(model, name+'_'+key)

    state_matrix = chpp.state_matrix
    modes_count = len(state_matrix)

    # state
    s('modes', Set(initialize=list(range(modes_count))))
    for i in range(modes_count):
        for j in range(modes_count):
            s('b_mode_{}_{}'.format(i,j), Var(model.t, within=Binary))
    s('b_off', Var(model.t, within=Binary))

    # power
    s('P_el', Var(model.t, within=Reals))
    s('P_th', Var(model.t, within=Reals))
    
    # forced mode changes
    if soft_dwell_time_constraints:
        s('b_allow_on', Var(model.t, within=Binary))
        s('b_allow_off', Var(model.t, within=Binary))
        s('b_force_on', Var(model.t, within=Binary))
        s('b_force_off', Var(model.t, within=Binary))

    # only one mode at the same time
    def con_distinct_mode(model, t):
        total = 0
        for i in range(modes_count):
            for j in range(modes_count):
                total += g('b_mode_{}_{}'.format(i,j))[t]
        return total == 1
    s('con_distinct_mode', Constraint(model.t, rule=con_distinct_mode))

    # a mode can only be left if it was entered before
    def con_mode_continuity(model, mode, t):
        # determine if the previous transition lead to 'mode' (=1)
        if t == 0:
            if chpp.mode == mode:
                previous = 1
            else:
                previous = 0
        else:    
            previous = 0
            for i in range(modes_count):
                previous += g('b_mode_{}_{}'.format(i,mode))[t-1]
        # does the next transition start in 'mode' (=1)?
        current = 0
        for i in range(modes_count):
            current += g('b_mode_{}_{}'.format(mode,i))[t]
        return previous == current
    s('con_mode_continuity', Constraint(g('modes'), model.t, rule=con_mode_continuity))

    # make the model more readble by introducing b_off
    def con_detect_off_state(model, t):
        return g('b_off')[t] == sum(g('b_mode_{}_{}'.format(i, 0))[t] for i in range(modes_count))
    s('con_detect_off_state', Constraint(model.t, rule=con_detect_off_state))

    # el. Power
    def con_el_power(model, t):
        total = 0
        for i in range(modes_count):
            for j in range(modes_count):
                total += g('b_mode_{}_{}'.format(i,j))[t] * state_matrix[i][j][0]
        return g('P_el')[t] == total
    s('con_el_power', Constraint(model.t, rule=con_el_power))

    # th. Power
    def con_th_power(model, t):
        total = 0
        for i in range(modes_count):
            for j in range(modes_count):
                total += g('b_mode_{}_{}'.format(i,j))[t] * state_matrix[i][j][1]
        return g('P_th')[t] == total
    s('con_th_power', Constraint(model.t, rule=con_th_power))

    if soft_dwell_time_constraints:
        # force on
        def con_force_on(model, t):
            return g('b_off')[t] <= (1-g('b_force_on')[t])
        s('con_force_on', Constraint(model.t, rule=con_force_on))

        # force off
        def con_force_off(model, t):
            return g('b_off')[t] >= g('b_force_off')[t]
        s('con_force_off', Constraint(model.t, rule=con_force_off))

    # min dwell time 'off', turning 'on'
    def con_min_off(model, t):
        min_off = math.ceil(chpp.min_off_time / (model.dt.value))
        if min_off == 0:
            return Constraint.Skip

        if soft_dwell_time_constraints:
            ignore_dwell_time = (g('b_allow_on')[t] + g('b_force_on')[t]) * model.M
        else:
            ignore_dwell_time = 0

        if t-min_off >= 0:    
            # if switching to "on" in i then had to be "off" for at least min_off
            return  ignore_dwell_time + sum(g('b_off')[j] for j in range(t-min_off,t)) >= min_off * (-g('b_off')[t] + g('b_off')[t-1])
        elif chpp.mode != 0:
            # chpp is initially running and can not have reached min_off yet
            # con_min_on is responsible for guaranteeing min_on
            if t == 0:
                return Constraint.Skip
            # if it is deactivated during this period, it needs to stay off
            return g('b_off')[t] >= g('b_off')[t-1] - ignore_dwell_time
        else: # chpp.mode == 0
            # initial dwell time needs to be considered
            dwell_time = math.floor(chpp.dwell_time / model.dt.value)
            # chpp is initially idling
            if dwell_time + t < min_off:
                # has not been idling long enough yet, remain "off"
                if t == 0:
                    return g('b_off')[0] >= 1 - ignore_dwell_time
                else:
                    # only force off if it has not been forced on before
                    return g('b_off')[t] >= g('b_off')[t-1] - ignore_dwell_time
            if t == 0:
                # if chpp is initially off, it has been off for at least min_off
                # if chpp is initially on, it may remain on
                return Constraint.Skip
            # it is t >= min_off - dwell_time, was it off the whole time?
            return  ignore_dwell_time + sum(g('b_off')[j] for j in range(0,t)) >= t * (-g('b_off')[t] + g('b_off')[t-1])
    s('con_min_off', Constraint(model.t, rule=con_min_off))

    # min dwell time 'on', turning 'off'
    def con_min_on(model, t):
        min_on = math.ceil(chpp.min_on_time / (model.dt.value))
        if min_on == 0:
            return Constraint.Skip

        if soft_dwell_time_constraints:
            ignore_dwell_time = (g('b_allow_off')[t] + g('b_force_off')[t]) * model.M
        else:
            ignore_dwell_time = 0

        # if switching to "off" in i then had to be "on" for at least min_on
        if t-min_on >= 0:    
            return  ignore_dwell_time + sum((1-g('b_off')[j]) for j in range(t-min_on,t)) >= min_on * (g('b_off')[t] - g('b_off')[t-1])
        elif chpp.mode == 0:
            # chpp is initially idling and can not have reached min_on
            # con_min_off is responsible for guaranteeing min_off
            if t == 0:
                return Constraint.Skip
            # if it is activated during this period, it needs to stay on
            return g('b_off')[t] <= g('b_off')[t-1] + ignore_dwell_time
        else: # chpp.mode != 0      
            # initial dwell time needs to be considered
            dwell_time = math.floor(chpp.dwell_time / model.dt.value)
            if dwell_time + t < min_on:
                # has not been running long enough yet, remain "on"
                if t == 0:
                    return g('b_off')[0] <= ignore_dwell_time
                else:
                    # only force off if it has not been forced on before
                    return g('b_off')[t] <= g('b_off')[t-1] + ignore_dwell_time
            if t == 0:
                # min_on already reached
                return Constraint.Skip
            # it is t >= min_on - dwell_time, was it on the whole time?
            return  ignore_dwell_time + sum((1-g('b_off')[j]) for j in range(0,t)) >= t * (g('b_off')[t] - g('b_off')[t-1])
    s('con_min_on', Constraint(model.t, rule=con_min_on))