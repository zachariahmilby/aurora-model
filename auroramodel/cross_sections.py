from pathlib import Path

import astropy.units as u
import numpy as np

from auroramodel.electrons import ElectronProperties
from auroramodel.general import _package_directory


class CrossSection:
    """
    Class to load an emission cross section and calculate an emission rate
    coefficient for a given electron energy distribution.
    """
    def __init__(self,
                 parent_species: str,
                 emitting_atom: str,
                 wavelength: u.Quantity):
        """
        Parameters
        ----------
        parent_species : str
            Chemical formula or symbol of the parent atom/molecule.
        emitting_atom : str
            Chemical formula or symbol of the emitting atom.
        wavelength : u.Quantity
            Wavelength of the emission cross section.
        """
        self._parent_species = parent_species
        self._emitting_atom = emitting_atom
        self._wavelength = wavelength.to(u.nm)
        self._energy, self._cross_section = self._load_cross_section()

    def _parse_filepath(self) -> Path:
        """
        Get the path to the .dat file containing the cross section.

        Returns
        -------
        Path
            The path to the .dat file containing the cross section.
        """
        wavelength = f'{self._wavelength}'.replace(' ', '')
        location = Path(_package_directory, 'anc', 'cross_sections')
        atom = self._emitting_atom
        filename = f'{atom}_{wavelength}_{self._parent_species}.dat'
        filepath = Path(location, filename)
        if not filepath.exists():
            msg = (f'No {self._emitting_atom} {self._wavelength} emission '
                   f'cross section for {self._parent_species} found.')
            raise FileNotFoundError(msg)
        return filepath

    def _load_cross_section(self) -> tuple[u.Quantity, u.Quantity]:
        """
        Load a cross section and its energies.

        Returns
        -------
        tuple[u.Quantity, u.Quantity]
            Emission energy in [eV] and cross section in [cm²].
        """
        filepath = self._parse_filepath()
        energy, cross_section = np.genfromtxt(filepath,
                                              unpack=True,
                                              delimiter=',',
                                              skip_header=True)
        energy = energy * u.eV
        cross_section = cross_section * u.cm**2
        return energy, cross_section

    def get_emission_rate_coefficient(
            self,
            electron_properties: ElectronProperties) -> u.Quantity:
        """
        Calculate emission rate coefficient in [ph cm³ electron⁻¹ s⁻¹].

        Parameters
        ----------
        electron_properties : ElectronProperties
            The properties of the electrons exciting the emission.

        Returns
        -------
        u.Quantity
            The emission rate coefficient in units of [ph cm³ s⁻¹].
        """

        fev = electron_properties.calculate_energy_distribution()
        dev = np.gradient(electron_properties.calculate_energy_domain())
        rate = np.sum(self._cross_section * fev * dev)
        return u.ph / u.electron * rate.to(u.cm**3 / u.s)

    @property
    def parent_species(self) -> str:
        return self._parent_species

    @property
    def emitting_atom(self) -> str:
        return self._emitting_atom

    @property
    def wavelength(self) -> u.Quantity:
        return self._wavelength

    @property
    def energy(self) -> u.Quantity:
        return self._energy

    @property
    def cross_section(self) -> u.Quantity:
        return self._cross_section


def get_all_cross_sections() -> dict[str, CrossSection]:
    """
    Get a dictionary containing all available cross sections.

    Returns
    -------
    dict[str, CrossSection]
        A dictionary containing all available cross sections as `CrossSection`
        objects.
    """
    cross_sections = {}
    location = Path(_package_directory, 'anc', 'cross_sections')
    files = sorted(location.glob('*.dat'))
    for file in files:
        emitting_atom, wavelength, parent_species = file.stem.split('_')
        wavelength = float(wavelength.replace('nm', '')) * u.nm
        cross_section = CrossSection(parent_species=parent_species,
                                     emitting_atom=emitting_atom,
                                     wavelength=wavelength)
        key = f'{parent_species} - {emitting_atom} {wavelength}'
        cross_sections[key] = cross_section
    return cross_sections
