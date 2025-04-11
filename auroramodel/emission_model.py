from pathlib import Path

import numpy as np
import pandas as pd

from auroramodel.cross_sections import cross_sections
from auroramodel.electrons import default_electron_properties, \
    ElectronProperties
from auroramodel.mcmc import MCMC
from auroramodel.observations import Observation


class EmissionModel:
    """
    Model target auroral emission.
    """
    def __init__(self,
                 target: str,
                 observations: list[Observation],
                 parent_species: list[str]):
        """
        Parameters
        ----------
        target : str
            The name of the target satellite.
        observations : list[Observation]
            A list of emission observations for fitting. Optional, since you
            can also just run the model in a theoretical context.
        parent_species : list[str]
            A list of the parent species you want to evaluate (or fit).
        """
        self._target = target
        self._observations = observations
        self._parent_species = parent_species

    def run(self,
            vary: str = 'atmosphere columns',
            electron_properties: ElectronProperties = None,
            scale_electron_density: bool = False,
            mcmc_out_directory: str or Path = None,
            iteration: int = None,
            count: int = None):
        """
        Run the emission model. Uses MCMC to estimate starting parameters, then
        uses least-squares minimization to calculate the best-fit parameters.
        Includes the ability to vary three different components: electron
        energy, electron density or column density.

        Parameters
        ----------
        vary : str
            Which component to vary. Options are 'atmosphere columns',
            'electron energy' or 'electron density'.
        electron_properties : ElectronProperties, optional
            Defines the electron properties used in the modeling. If you elect
            to vary electron energy or density, these properties will define
            the initial parameters.
        scale_electron_density : bool, optional
            Whether or not to apply the scale-heights in the electron
            properties to the electron densities.
        mcmc_out_directory : str, optional
            Output directory for MCMC graphics and CSVs if you want to save
            them.
        iteration : int
            Iteration number (if you want it reflected in the printed progress
            bar).
        count: int
            Total number of iterations (if you want it reflected in the printed
            progress bar).

        Returns
        -------

        """
        # determine what to vary
        options = ['atmosphere columns', 'electron energy', 'electron density']
        if vary not in options:
            msg = (f"Argument '{vary}' not valid. 'vary' must be one of "
                   f"'atmosphere columns', 'electron energy', or "
                   f"'electron density'.")
            raise ValueError(msg)

        if electron_properties is None:
            electron_properties = default_electron_properties[self._target]

        # get median parameters and initial uncertainty estimates with MCMC
        mcmc = MCMC(target=self._target,
                    observations=self._observations,
                    parent_species=self._parent_species)
        densities = mcmc.run(electron_properties=electron_properties,
                             scale_electron_density=scale_electron_density,
                             output_directory=mcmc_out_directory,
                             iteration=iteration,
                             count=count)

        return densities

def save_rates(filepath: str or Path) -> None:
    """
    Wrapper function to calculate and save emission rate coefficients for each
    satellite's canonical electron properties.

    Parameters
    ----------
    filepath : str or Path
        Where to save emission rate coefficient output CSV file.
    """

    wavelengths = np.unique(
        [xsec.wavelength for xsec in cross_sections.values()])
    parents = np.unique(
        [xsec.parent_species for xsec in cross_sections.values()])

    for satellite in default_electron_properties.keys():
        df = pd.DataFrame()
        df['wavelength_nm'] = [s.replace('nm', '') for s in wavelengths]
        electron_properties = default_electron_properties[satellite]
        for parent in parents:
            rates = []
            for wavelength in wavelengths:
                try:
                    xc = cross_sections[f'{parent}_{wavelength}']
                    rates.append(xc.get_rate(electron_properties).value)
                except KeyError:
                    rates.append('---')
            df[f'{parent}_cm3/s'] = rates
        savename = Path(
            filepath, f'{satellite.lower()}_emission_rate_coefficients.csv')
        df.to_csv(savename, index=False)
