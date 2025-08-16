from abc import ABC, abstractmethod

from pysrc.Filter.Base import SHCFilter
from pysrc.BasisFunction.Harmonic import Harmonic


class Leakage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_to(self, gqij, get_grid=False):
        pass

    @abstractmethod
    def format(self):
        pass


def filter_grids(gqij, shc_filter: SHCFilter, harmonic: Harmonic):
    cqlm, sqlm = harmonic.analysis(gqij)

    cqlm_f, sqlm_f = shc_filter.apply_to(cqlm, sqlm)
    gqij_filtered = harmonic.synthesis(cqlm_f, sqlm_f)

    return gqij_filtered
