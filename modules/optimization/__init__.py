def _initialize_factory():
    from .constraintfactory import ConstraintFactory
    from .factories import bess, chpp, hwt, evse
    from ..simulation.individual.bess import BESS
    from ..simulation.individual.chpp import CHPP
    from ..simulation.individual.hwt import HWT
    from ..simulation.individual.evse import EVSE

    ConstraintFactory.register_converter(BESS, bess.factory)
    ConstraintFactory.register_converter(CHPP, chpp.factory)
    ConstraintFactory.register_converter(HWT, hwt.factory)
    ConstraintFactory.register_converter(EVSE, evse.factory)

    from .factories import chpp_hwt, bess_chpp_hwt, hwt_gcb, holl, aggregated_bess
    from ..simulation.integrated.chpp_hwt import CHPP_HWT
    from ..simulation.integrated.bess_chpp_hwt import BESS_CHPP_HWT
    from ..simulation.integrated.hwt_gcb import HWT_GCB
    from ..simulation.integrated.holl import HoLL
    from ..simulation.integrated.aggregated_bess import AggregatedBESS

    ConstraintFactory.register_converter(CHPP_HWT, chpp_hwt.factory)
    ConstraintFactory.register_converter(BESS_CHPP_HWT, bess_chpp_hwt.factory)
    ConstraintFactory.register_converter(HWT_GCB, hwt_gcb.factory)
    ConstraintFactory.register_converter(HoLL, holl.factory)
    ConstraintFactory.register_converter(AggregatedBESS, aggregated_bess.factory)

_initialize_factory()