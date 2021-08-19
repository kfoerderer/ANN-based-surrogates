from pyomo.core.base import ConcreteModel, Objective, Constraint, Set, Param, Var, NonNegativeReals, Reals
from pyomo.environ import minimize, SolverFactory

from .constraintfactory import ConstraintFactory
from ..simulation.simulationmodel import SimulationModel

class _ZerosList:
    def __getitem__(self, i):
        return 0

class CostMILP:

    def __init__(self, time_step, time_step_count):

        model = ConcreteModel()
        model.t = Set(initialize=list(range(time_step_count)))
        model.dt = Param(initialize=time_step)
        model.M = Param(initialize=10000000)
        model.P_total = Var(model.t, within=Reals)

        self.model = model
        self.components = []
        self.objective_created = False

    def add_constraints(self, simulation_model: SimulationModel, name='', **kwargs):
        if name in self.components:
            raise ValueError('Name \'{}\' already registered'.format(name))
        if self.objective_created is True:
            raise RuntimeError('Objective has already been created')
        ConstraintFactory.add_to_model(self.model, name, simulation_model, **kwargs)
        self.components.append(name)

    def create_objective(self, price_profile: [float]):
        """
        ### Parameters
        price_profile ```[float]``` prices in cents/kWh
        """
        if self.objective_created:
            raise RuntimeError('Objective has already been created')
        self.objective_created = True
        if len(self.components) == 0:
            raise RuntimeError('No constraints have been added to the model')

        model = self.model
        model.price = Param(model.t, initialize={i: v for i, v in enumerate(price_profile)})

        def objective(model):
            # cts / kWh * (W * k) * (s * h/s) = cts
            return sum((model.price[i] * (model.P_total[i]/1000) * (model.dt/60/60)) for i in model.t)
        model.obj = Objective(rule=objective, sense=minimize)

        # total power
        zeros = _ZerosList()
        def con_P_total(model, i):
            total = 0
            for name in self.components:
                total += getattr(model, name+'_P_el', zeros)[i]
            return model.P_total[i] == total
        model.con_P_total = Constraint(model.t, rule=con_P_total)

    def solve(self, solver: str='gurobi', timelimit=60, mipgap=0.05, verbose=True):
        if self.objective_created is False:
            raise RuntimeError('No objective')

        solver = SolverFactory(solver)
        solver.options['timelimit'] = timelimit
        solver.options['mipgap'] = mipgap
        return solver.solve(self.model, tee=verbose)
    