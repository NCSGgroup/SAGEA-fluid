from pysrc.Auxiliary.EnumClasses import LeakageMethod
from pysrc.Leakage.Addictive import Addictive
from pysrc.Leakage.BufferZone import BufferZone
from pysrc.Leakage.DataDriven import DataDriven
from pysrc.Leakage.ForwardModeling import ForwardModeling
from pysrc.Leakage.Iterative import Iterative
from pysrc.Leakage.Multiplicative import Multiplicative
from pysrc.Leakage.Scaling import Scaling
from pysrc.Leakage.ScalingGrid import ScalingGrid


def get_leakage_deductor(method: LeakageMethod):
    """
    :param method: LeakageType
    """

    if method == LeakageMethod.Multiplicative:
        leakage = Multiplicative()

    elif method == LeakageMethod.Addictive:
        leakage = Addictive()

    elif method == LeakageMethod.Scaling:
        leakage = Scaling()

    elif method == LeakageMethod.ScalingGrid:
        leakage = ScalingGrid()

    elif method == LeakageMethod.Iterative:
        leakage = Iterative()

    elif method == LeakageMethod.DataDriven:
        leakage = DataDriven()

    elif method == LeakageMethod.ForwardModeling:
        leakage = ForwardModeling()

    elif method == LeakageMethod.BufferZone:
        leakage = BufferZone()

    else:
        assert False, 'leakage false'

    return leakage
