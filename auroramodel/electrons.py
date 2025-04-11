import warnings

import numpy as np
from astropy import units as u, constants as c


class ElectronProperties:
    """
    Calculate an electron energy distribution for a given peak energy.
    """
    def __init__(self, 
                 n_e: u.Quantity, 
                 peak_energy: u.Quantity,
                 scale_height: u.Quantity, 
                 ratios: list[float] = None):
        """
        Parameters
        ----------
        n_e : u.Quantity
            Electron number density.
        peak_energy : u.Quantity
            The peak energies of the electron kinetic energy distribution. If
            there are multiple populations, use multiple peak energies.
        scale_height : u.Quantity
            Scale height of the electrons in the plasma.
        ratios : list[float]
            The relative fractions of the populations (if there are more than
            1).
        """
        if ratios is None:
            ratios = [1.0]
        self._n_e = n_e
        try:
            self._peak_energy = [e.to(u.eV).value for e in peak_energy] * u.eV
        except TypeError:
            self._peak_energy = [peak_energy.to(u.eV).value] * u.eV
        self._scale_height = scale_height
        self._ratios = ratios
        self._electron_energies = np.linspace(1, 1000, 9991) * u.eV
        self._differential = self._calculate_differential()
        self._energy_distribution = self._calculate_energy_distribution()

    # noinspection PyUnresolvedReferences
    def _calculate_energy_distribution(self) -> u.Quantity:
        """
        Calculate Maxwellian electron kinetic energy distribution centered at
        the peak energy/energies.
        """
        distribution = np.zeros(
            self._electron_energies.shape) * 1 / (u.eV * u.g) ** (1/2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for peak_energy, ratio in zip(self._peak_energy, self._ratios):
                if peak_energy != 0 * u.eV:
                    coeff = 4 / np.sqrt(2 * np.pi * c.m_e.to(u.g) * peak_energy**3)
                    exp = np.exp((self._electron_energies/peak_energy))
                    distribution += ratio * coeff / exp * self._electron_energies
        return distribution

    def _calculate_differential(self) -> u.Quantity:
        """
        Calculate energy spacing for integrations.
        """
        unit = self._electron_energies.unit
        return np.gradient(self._electron_energies.value) * unit

    @property
    def electron_energies(self) -> u.Quantity:
        return self._electron_energies

    @property
    def energy_distribution(self) -> u.Quantity:
        return self._energy_distribution

    @property
    def differential(self) -> u.Quantity:
        return self._differential

    @property
    def n0(self) -> u.Quantity:
        return self._n_e * u.electron

    def ne(self,
           z: u.Quantity):
        """
        Calculate electron density at some height `z` above or below the
        centrifugal equator.

        Parameters
        ----------
        z : u.Quantity
            The height above or below the centrifugal equator. Probably in
            Jupiter radii, but can be any length unit.

        Returns
        -------
        The scaled electron density.
        """
        if not isinstance(z, u.Quantity):
            z = z * u.R_jup
        scale = np.exp(-np.abs(z.si/self._scale_height.si))
        return self._n_e * scale

    @property
    def eV(self) -> u.Quantity:  # noqa
        return self._peak_energy

    @property
    def ratios(self) -> list[float]:
        return self._ratios

    @property
    def scale_height(self) -> u.Quantity:
        return self._scale_height


"""Canonical electron properties appropriate to each satellite's orbit."""
# noinspection PyUnresolvedReferences
default_electron_properties: dict[str, ElectronProperties] = {
    'Io': ElectronProperties(n_e=3000*u.electron/u.cm**3,
                             peak_energy=5*u.eV,
                             scale_height=0.77*u.R_jup),
    'Europa': ElectronProperties(n_e=160*u.electron/u.cm**3,
                                 peak_energy=[20, 250]*u.eV,
                                 ratios=[0.95, 0.05],
                                 scale_height=1.70*u.R_jup),
    'Ganymede': ElectronProperties(n_e=20*u.electron/u.cm**3,
                                   peak_energy=100*u.eV,
                                   scale_height=2.83*u.R_jup),
    'Callisto': ElectronProperties(n_e=0.15*u.electron/u.cm**3,
                                   peak_energy=35*u.eV,
                                   scale_height=3.69*u.R_jup)
}
