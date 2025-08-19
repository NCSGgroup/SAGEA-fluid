from SaGEA.auxiliary.preference.EnumClasses import LeakageMethod
from SaGEA.post_processing.leakage.Addictive import Addictive
from SaGEA.post_processing.leakage.BufferZone import BufferZone
from SaGEA.post_processing.leakage.DataDriven import DataDriven
from SaGEA.post_processing.leakage.ForwardModeling import ForwardModeling
from SaGEA.post_processing.leakage.Iterative import Iterative
from SaGEA.post_processing.leakage.Multiplicative import Multiplicative
from SaGEA.post_processing.leakage.Scaling import Scaling
from SaGEA.post_processing.leakage.ScalingGrid import ScalingGrid


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
