from lib.SaGEA.auxiliary.preference.EnumClasses import LeakageMethod
from lib.SaGEA.post_processing.leakage.Addictive import Addictive
from lib.SaGEA.post_processing.leakage.BufferZone import BufferZone
from lib.SaGEA.post_processing.leakage.DataDriven import DataDriven
from lib.SaGEA.post_processing.leakage.ForwardModeling import ForwardModeling
from lib.SaGEA.post_processing.leakage.Iterative import Iterative
from lib.SaGEA.post_processing.leakage.Multiplicative import Multiplicative
from lib.SaGEA.post_processing.leakage.Scaling import Scaling
from lib.SaGEA.post_processing.leakage.ScalingGrid import ScalingGrid


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
