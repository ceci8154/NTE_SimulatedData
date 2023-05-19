""" Telescope

Implementing relevant functions of a telescope.

Note:
    This is kept rather simple for the moment. A possible extension would be to implement specific observatories incl.
    a 'get_efficiency()' function that respects mirror coatings (silver vs. aluminum)
"""
import math
from dataclasses import dataclass


@dataclass
class Telescope:
    """ Telescope class

    Attributes:
        d_primary (float): diameter of primary mirror [m]
        d_secondary (float): diameter of secondary mirror [m]
    """
    d_primary: float
    d_secondary: float = 0.

    @property
    def area(self) -> float:
        """ Effective collecting area

        Returns:
            effective collecting area of the telescope [m^2]
        """
        return (self.d_primary ** 2 - self.d_secondary ** 2) / 4. * math.pi
