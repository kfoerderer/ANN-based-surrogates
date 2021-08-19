from pyomo.core.base import ConcreteModel, Objective, Constraint, Set, Param, Var, NonNegativeReals, Reals
from pyomo.environ import minimize, SolverFactory

from .constraintfactory import ConstraintFactory
from ..simulation.simulationmodel import SimulationModel

class _ZerosList:
    def __getitem__(self, i):
        return 0

class TargetDeviationMILP:

    def __init__(self, time_step, time_step_count):

        model = ConcreteModel()
        model.t = Set(initialize=list(range(time_step_count)))
        model.dt = Param(initialize=time_step)
        model.M = Param(initialize=10000000)
        model.P_total = Var(model.t, within=Reals)

        self.model = model
        self.components = []
        self.objective_created = False

    def add_constraints(self, target_model: SimulationModel, name='', **kwargs):
        if name in self.components:
            raise ValueError('Name \'{}\' already registered'.format(name))
        if self.objective_created is True:
            raise RuntimeError('Objective has already been created')
        ConstraintFactory.add_to_model(self.model, name, target_model, **kwargs)
        self.components.append(name)

    def create_objective(self, target_profile: [float]):
        """
        ### Parameters
        target_profile ```[float]``` Target load in W
        """
        if self.objective_created:
            raise RuntimeError('Objective has already been created')
        self.objective_created = True
        if len(self.components) == 0:
            raise RuntimeError('No constraints have been added to the model')

        model = self.model
        model.P_target = Param(model.t, initialize={i: v for i, v in enumerate(target_profile)})
        model.dP_abs = Var(model.t, within=NonNegativeReals)

        def objective(model):
            return sum((model.dP_abs[i]) for i in model.t)
        model.obj = Objective(rule=objective, sense=minimize)

        # difference positive
        def con_dP_pos(model, i):
            return (model.P_target[i] - model.P_total[i]) <= model.dP_abs[i]
        model.con_dP_pos = Constraint(model.t, rule=con_dP_pos)

        # difference positive
        def con_dP_neg(model, i):
            return (-model.P_target[i] + model.P_total[i]) <= model.dP_abs[i]
        model.con_dP_neg = Constraint(model.t, rule=con_dP_neg)

        # total power
        zeros = _ZerosList()
        def con_P_total(model, i):
            total = 0
            for name in self.components:
                total += getattr(model, name+'_P_el', zeros)[i]
            return model.P_total[i] == total
        model.con_P_total = Constraint(model.t, rule=con_P_total)

    def solve(self, solver='gurobi', timelimit=60, mipgap=0.05, verbose=True):
        if self.objective_created is False:
            raise RuntimeError('No objective')

        solver = SolverFactory(solver)
        solver.options['timelimit'] = timelimit
        solver.options['mipgap'] = mipgap
        return solver.solve(self.model, tee=verbose)
    