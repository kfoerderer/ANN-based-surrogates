from pyomo.core.base import ConcreteModel

from ..simulation.simulationmodel import SimulationModel

class ConstraintFactory:

    factory_functions = {}

    def add_to_model(pyomo_model: ConcreteModel, name: str, simulation_model: SimulationModel, **kwargs):
        converter = ConstraintFactory.factory_functions[type(simulation_model)]
        converter(pyomo_model, name, simulation_model, **kwargs)

    def register_converter(cls, func):
        ConstraintFactory.factory_functions[cls] = func