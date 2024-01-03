import warnings
from pathlib import Path

import astropy.constants as c
import astropy.units as u
import numpy as np
import pandas as pd

from auroramodel.general import _package_directory


class ElectronEnergyDistribution:
    """
    Calculate an electron energy distribution for a given peak energy.
    """
    def __init__(self, n_e: u.Quantity, peak_energies: [u.Quantity],
                 scale_height: u.Quantity, ratios=None):
        """
        Parameters
        ----------
        n_e : u.Quantity
            Electron number density.
        peak_energies : [u.Quantity]
            The peak energies of the electron energy distribution. If there
            are multiple populations, use multiple peak energies.
        ratios : [float]
            The ratios of the populations (if there are more than 1).
        """
        if ratios is None:
            ratios = [1.0]
        self._n_e = n_e
        try:
            iter(peak_energies)
        except TypeError:
            peak_energies = [peak_energies]
        self._peak_energies = peak_energies
        self._scale_height = scale_height
        self._ratios = ratios
        self._electron_energies = np.linspace(1, 1000, 9991) * u.eV
        self._differential = self._calculate_differential()
        self._energy_distribution = self._calculate_energy_distribution()

    # noinspection PyUnresolvedReferences
    def _calculate_energy_distribution(self) -> u.Quantity:
        """
        Calculate Maxwellian electron energy distribution centered at the peak
        energy.
        """
        distribution = np.zeros(
            self._electron_energies.shape) * 1 / (u.eV * u.g) ** (1/2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for peak_energy, ratio in zip(self._peak_energies, self._ratios):
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
        return self._n_e

    def ne(self, z: u.Quantity):
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
        return self._n_e * np.exp(-np.abs(z.si/self._scale_height.si))

    @property
    def eV(self) -> u.Quantity:  # noqa
        return self._peak_energies


class CrossSection:
    """
    Load a high-resolution excitation or emission cross section.
    """
    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename : str
            Name of the .dat file containing the cross section.
        """
        self._filename = filename
        self._parent_species, self._wavelength = self._parse_name()
        self._energy, self._cross_section = self._load_cross_section()

    def _parse_name(self):
        species, wavelength = self._filename.replace('.dat', '').split('_')
        return species, wavelength

    def _load_cross_section(self):
        """
        Load a cross section and its energies.
        """
        datapath = Path(_package_directory, 'cross_sections', self._filename)
        if not datapath.exists():
            raise FileNotFoundError("Cross section doesn't exist!")
        data = pd.read_csv(datapath)
        energy = data['energy_eV'].to_numpy() * u.eV
        cross_section = data['cross_section_cm2'].to_numpy() * u.cm**2
        return energy, cross_section

    # noinspection PyUnresolvedReferences
    def get_rate(
            self, electron_energy_distribution: ElectronEnergyDistribution):
        fev = electron_energy_distribution.energy_distribution
        dev = electron_energy_distribution.differential
        return np.sum(self._cross_section * fev * dev).to(u.cm**3 / u.s)

    @property
    def name(self) -> str:
        return f'{self._parent_species} {self._wavelength})'

    @property
    def energy(self) -> u.Quantity:
        return self._energy

    @property
    def cross_section(self) -> u.Quantity:
        return self._cross_section

    @property
    def wavelength(self) -> u.Quantity:
        return self._wavelength

    @property
    def parent_species(self) -> str:
        return self._parent_species


"""Electron energy distribution appropriate to each satellite's orbit."""
# noinspection PyUnresolvedReferences
electron_energy_distributions = {
    'Io': ElectronEnergyDistribution(
        n_e=3000*u.electron/u.cm**3, peak_energies=[5*u.eV],
        scale_height=0.74*c.R_jup),
    'Europa': ElectronEnergyDistribution(
        n_e=160*u.electron/u.cm**3, peak_energies=[20, 250]*u.eV,
        ratios=[0.95, 0.05],
        scale_height=1.68*c.R_jup),
    'Ganymede': ElectronEnergyDistribution(
        n_e=20*u.electron/u.cm**3, peak_energies=[100*u.eV],
        scale_height=2.78*c.R_jup),
    'Callisto': ElectronEnergyDistribution(
        n_e=0.15*u.electron/u.cm**3, peak_energies=[35*u.eV],
        scale_height=3.66*c.R_jup)
}


"""All available cross sections."""
cross_sections = {}
files = sorted(Path(_package_directory, 'cross_sections').glob('*.dat'))
for file in files:
    xs = CrossSection(file.name)
    cross_sections[f'{xs.parent_species}_{xs.wavelength}'] = xs


def save_rates(filepath: str or Path):
    """
    Wrapper function to calculate and save emission rate coefficients for each
    satellite.
    """

    wavelengths = np.unique(
        [xsec.wavelength for xsec in cross_sections.values()])
    parents = np.unique(
        [xsec.parent_species for xsec in cross_sections.values()])

    for satellite in electron_energy_distributions.keys():
        df = pd.DataFrame()
        df['wavelength_nm'] = [s.replace('nm', '') for s in wavelengths]
        energy_distribution = electron_energy_distributions[satellite]
        for parent in parents:
            rates = []
            for wavelength in wavelengths:
                try:
                    xc = cross_sections[f'{parent}_{wavelength}']
                    rates.append(xc.get_rate(energy_distribution).value)
                except KeyError:
                    rates.append('---')
            df[f'{parent}_cm3/s'] = rates
        savename = Path(
            filepath, f'{satellite.lower()}_emission_rate_coefficients.csv')
        df.to_csv(savename, index=False)
