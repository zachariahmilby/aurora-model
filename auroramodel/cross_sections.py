from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd

from auroramodel.electrons import ElectronProperties
from auroramodel.general import _package_directory


class CrossSection:
    """
    Load a high-resolution excitation or emission cross section.
    """
    def __init__(self,
                 filename: str):
        """
        Parameters
        ----------
        filename : str
            Name of the .dat file containing the cross section.
        """
        self._filename = filename
        self._parent_species, self._wavelength = self._parse_name()
        self._energy, self._cross_section = self._load_cross_section()

    def _parse_name(self) -> tuple[str, str]:
        replace = self._filename.replace('.dat', '').replace('nm', '')
        species, wavelength = replace.split('_')
        return species, wavelength

    def _load_cross_section(self):
        """
        Load a cross section and its energies.
        """
        datapath = Path(_package_directory, 'anc/cross_sections', self._filename)
        if not datapath.exists():
            raise FileNotFoundError("Cross section doesn't exist!")
        data = pd.read_csv(datapath)
        energy = data['energy_eV'].to_numpy() * u.eV
        cross_section = data['cross_section_cm2'].to_numpy() * u.cm**2
        return energy, cross_section

    # noinspection PyUnresolvedReferences
    def get_rate(self,
                 electron_properties: ElectronProperties) -> u.Quantity:
        fev = electron_properties.energy_distribution
        dev = electron_properties.differential
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
        return float(self._wavelength) * u.nm

    @property
    def parent_species(self) -> str:
        return self._parent_species


"""All available cross sections."""
cross_sections = {}
files = sorted(Path(_package_directory, 'anc', 'cross_sections').glob('*.dat'))
for file in files:
    xs = CrossSection(file.name)
    cross_sections[f'{xs.parent_species} {xs.wavelength}'] = xs
