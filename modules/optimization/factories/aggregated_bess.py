from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint
from pyomo.environ import inequality

from ..constraintfactory import ConstraintFactory

from ...simulation.integrated.aggregated_bess import AggregatedBESS
from ...simulation.individual.bess import BESS

def factory(model: ConcreteModel, name: str, aggregatedBess: AggregatedBESS, **kwargs):
    capacities = aggregatedBess.capacities
    charging_efficiencies = aggregatedBess.charging_efficiencies
    discharging_efficiencies = aggregatedBess.discharging_efficiencies
    max_charging_powers = aggregatedBess.max_charging_powers
    max_discharging_powers = aggregatedBess.max_discharging_powers
    relative_losses = aggregatedBess.relative_losses
    constraint_fuzziness = aggregatedBess.constraint_fuzziness # is not used in BESS model, though
    
    stored_energy = aggregatedBess.stored_energy

    for i in range(len(aggregatedBess.capacities)):
        actions = [max_discharging_powers[i], max_charging_powers[i]]
        tmp_bess = BESS(900, actions, capacities[i], charging_efficiencies[i], discharging_efficiencies[i], relative_losses[i], True, constraint_fuzziness)
        tmp_bess.stored_energy = stored_energy[i]
        ConstraintFactory.add_to_model(model, '_{}'.format(i), tmp_bess)

    def s(key, value, name=name):
        setattr(model, name+'_'+key, value)

    def g(key, name=name):
        return getattr(model, name+'_'+key)
    
    s('P_el', Var(model.t, within=Reals))

    # power
    def con_power(model, i):
        total_power = 0
        for idx in range(len(aggregatedBess.capacities)):
            total_power += g('P_el', name+'_{}'.format(idx))[i]
        return g('P_el')[i] == total_power
    s('con_power', Constraint(model.t, rule=con_power))

    