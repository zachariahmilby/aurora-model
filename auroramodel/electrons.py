import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy import units as u, constants as c

from auroramodel.general import _package_directory


class ElectronProperties:
    """
    Properties of an electron population and methods to calculate its
    Maxwellian energy distribution.
    """
    def __init__(self, 
                 density: u.Quantity,
                 peak_energy: u.Quantity,
                 scale_height: u.Quantity, 
                 ratios: Iterable[float] = None):
        """
        Parameters
        ----------
        density : u.Quantity
            Electron number density.
        peak_energy : u.Quantity
            The peak energies of the electron kinetic energy distribution. If
            there are multiple populations, use multiple peak energies.
        scale_height : u.Quantity
            Scale height of the electrons in the plasma.
        ratios : iterable of floats, optional
            The relative fractions of the density allocated between the
            different energies.
        """
        self._density = density
        self._peak_energy = self._make_iterable_quantity(peak_energy)
        self._scale_height = scale_height
        if ratios is None:
            ratios = [1.0]
        self._ratios = np.array(ratios)

    def __str__(self):
        energy_label = 'energy'
        print_ratios = False
        if len(self._peak_energy) > 1:
            energy_label = 'energies'
            print_ratios = True
        string = 'Electron properties:\n'
        string += f'  Density: {self._density}\n'
        if print_ratios:
            string += f'  Density ratios: {self._ratios.tolist()}\n'
        if print_ratios:
            energy_value = (f'{self._peak_energy.value.tolist()} '
                            f'{self._peak_energy.unit}')
        else:
            energy_value = f'{self._peak_energy[0]}'
        string += f'  Peak {energy_label}: {energy_value}\n'
        string += f'  Gaussian scale height: {self._scale_height}'
        return string

    @staticmethod
    def _make_iterable_quantity(value: u.Quantity) -> u.Quantity:
        try:
            iter(value)
        except TypeError:
            value = [value.value] * value.unit
        return value

    @staticmethod
    def calculate_energy_domain() -> u.Quantity:
        """
        Calculate the discrete energy domain of the electrons.

        Returns
        -------
        u.Quantity
            The discrete energy domain of the electrons in [eV] (SI equivalent
            [J]).
        """
        # load wavelengths from one of the cross sections directly
        location = Path(_package_directory, 'anc', 'cross_sections')
        files = sorted(location.glob('*.dat'))
        wavelengths, _ = np.genfromtxt(files[0],
                                       unpack=True,
                                       delimiter=',',
                                       skip_header=True)
        return wavelengths * u.eV

    # noinspection PyUnresolvedReferences
    def calculate_energy_distribution(self) -> u.Quantity:
        """
        Calculate Maxwellian electron kinetic energy distribution centered at
        the peak energy/energies.

        Returns
        -------
        u.Quantity
            The electron kinetic energy distribution in [eV⁻¹ᐟ² g⁻¹ᐟ²]
            (SI equivalent [s m⁻¹ kg⁻¹]).
        """
        energy_domain = self.calculate_energy_domain()
        unit = 1 / (u.eV * u.g) ** (1/2)
        distribution = np.zeros(energy_domain.shape) * unit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for peak_energy, ratio in zip(self._peak_energy, self._ratios):
                if peak_energy != 0 * u.eV:
                    coeff = 4 / np.sqrt(2 * np.pi * c.m_e * peak_energy ** 3)
                    exp = np.exp((energy_domain/peak_energy))
                    distribution += ratio * coeff / exp * energy_domain
        return distribution

    def calculate_density(self,
                          distance: u.Quantity) -> u.Quantity:
        """
        Calculate electron density at some distance above or below the
        centrifugal equator.

        Parameters
        ----------
        distance : u.Quantity
            The height above or below the centrifugal equator. Probably in
            Jupiter radii, but can be any length unit.

        Returns
        -------
        The scaled electron density.
        """
        return self._density * np.exp(-(distance.si/self._scale_height.si)**2)

    @property
    def density(self) -> u.Quantity:
        return self._density * u.electron

    @property
    def ratios(self) -> np.ndarray:
        return self._ratios

    @property
    def eV(self) -> u.Quantity:  # noqa
        return self._peak_energy

    @property
    def scale_height(self) -> u.Quantity:
        return self._scale_height


"""Canonical electron properties appropriate to each satellite's orbit."""
_default_electron_properties: dict[str, ElectronProperties] = {
    'Io': ElectronProperties(density=2500*u.electron/u.cm**3,
                             peak_energy=5*u.eV,
                             scale_height=0.77*u.R_jup),
    'Europa': ElectronProperties(density=158*u.electron/u.cm**3,
                                 ratios=[0.95, 0.05],
                                 peak_energy=[20, 300]*u.eV,
                                 scale_height=1.70*u.R_jup),
    'Ganymede': ElectronProperties(density=20*u.electron/u.cm**3,
                                   peak_energy=100*u.eV,
                                   scale_height=2.83*u.R_jup),
    'Callisto': ElectronProperties(density=0.15*u.electron/u.cm**3,
                                   peak_energy=35*u.eV,
                                   scale_height=3.69*u.R_jup)
}

def get_canonical_electron_properties(name: str) -> ElectronProperties:
    """
    Get canonical electron properties for Io, Europa, Ganymede or Callisto.

    Parameters
    ----------
    name : str
        The name of the satellite. Either 'Io', 'Europa', 'Ganymede' or
        'Callisto'.

    Returns
    -------
    ElectronProperties
        The canonical electron properties for the named satellite.
    """
    return _default_electron_properties[name]
