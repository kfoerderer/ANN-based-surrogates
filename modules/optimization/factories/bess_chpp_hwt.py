from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

from .bess import factory as bess_factory
from .chpp_hwt import factory as chpp_hwt_factory

from ...simulation.integrated.chpp_hwt import CHPP_HWT
from ...simulation.integrated.bess_chpp_hwt import BESS_CHPP_HWT

def factory(model: ConcreteModel, name: str, bess_chpp_hwt: BESS_CHPP_HWT, **kwargs):
    # bess and chpp/hwt/demand are independent of each other
    # => use the two factories for the individual system

    bess_factory(model, name + '_bess', bess_chpp_hwt.bess)

    chpp_hwt = CHPP_HWT(bess_chpp_hwt.chpp, bess_chpp_hwt.hwt, bess_chpp_hwt.demand, bess_chpp_hwt.constraint_fuzziness)
    chpp_hwt_factory(model, name + '_', chpp_hwt)

    def s(key, value, name=name):
        setattr(model, name+'_'+key, value)

    def g(key, name=name):
        return getattr(model, name+'_'+key)
    
    s('P_el', Var(model.t, within=Reals))

    # power
    def con_power(model, i):
        return g('P_el')[i] == g('P_el', name+'_bess')[i] + g('P_el', name+'_')[i]
    s('con_power', Constraint(model.t, rule=con_power))

    