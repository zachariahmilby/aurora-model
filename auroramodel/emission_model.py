from typing import Iterable

import astropy.units as u
import numpy as np
import pandas as pd
from lmfit import Model

from auroramodel.cross_sections import get_all_cross_sections
from auroramodel.electrons import (ElectronProperties,
                                   get_canonical_electron_properties)
from auroramodel.general import (available_constituents,
                                 wavelengths,
                                 emissions,
                                 _calculate_surface_brightness)
from auroramodel.observations import Observation


class EmissionModel:
    """
    Galilean satellite auroral emission model.
    """
    def __init__(self):
        self._available_constituents = available_constituents
        self._wavelengths = wavelengths
        self._emissions = emissions
        self._cross_sections = get_all_cross_sections()
        self._base_column_density = 1e14 * u.cm**-2

    @staticmethod
    def _parse_list_of_units(list_of_quantities: Iterable[u.Quantity],
                             unit: u.Unit) -> u.Quantity:
        """
        Convenience function to convert a list of Astropy Quantities into a
        single iterable quantity.

        Parameters
        ----------
        list_of_quantities : Iterable[u.Quantity]
            The list containing quantity objects.
        unit : u.Unit
            The desired unit for the output quantity.

        Returns
        -------
        u.Quantity
            The list as a single iterable quantity.
        """
        return [qty.to(unit).value for qty in list_of_quantities] * unit

    def run(self,
            electron_properties: ElectronProperties or str,
            column_densities: dict[str, u.Quantity],
            distance: u.Quantity = 0 * u.km) -> pd.DataFrame:
        """
        Evaluate the model for supplied column densities.

        Parameters
        ----------
        electron_properties : ElectronProperties or str
            Ambient electron properties. Can be specified exactly using the
            `ElectronProperties` class, or you can provide the string 'Io',
            'Europa', 'Ganymede' or 'Callisto' to get their current canonical
            values. To see these properties, use the
            `get_canonical_electron_properties` function.
        column_densities : dict[str, u.Quantity]
            A dictionary containing the column densities associated with each
            of the compositional species in [cm⁻²]. The keys should match one
            of the species returned by the method `available_constituents`.
        distance : u.Quantity
            Distance from the centrifugal equator if you want to scale the
            densities using the Gaussian scale height. Set to 0 km by default,
            meaning no scaling takes place.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the results of the model evaluation. Index
            labels are the emissions and column labels are the modeled
            atmospheric constituents. Note: the appropriate unit is stored in
            the property `unit`.
        """
        if isinstance(electron_properties, str):
            electron_properties = get_canonical_electron_properties(
                electron_properties)
        composition = list(column_densities.keys())
        column_densities = self._parse_list_of_units(
            column_densities.values(), u.cm**-2)

        n_atm = len(composition)
        n_lines = len(emissions)
        electron_density = electron_properties.calculate_density(distance)

        output = np.full((n_atm, n_lines), fill_value=np.nan)
        for i, species in enumerate(composition):
            for j, emission in enumerate(emissions):
                for item in [' nm', ' I' , '[', ']']:
                    emission = emission.replace(item, '')
                wavelength, atom = emission.split(' ')
                key = f'{species} - {atom} {wavelength} nm'
                if key not in self._cross_sections.keys():
                    continue
                else:
                    xs = self._cross_sections[key]
                    rate = xs.get_emission_rate_coefficient(
                        electron_properties)
                    brightness = _calculate_surface_brightness(
                        electron_density=electron_density,
                        column_density=column_densities[i],
                        rate=rate)
                    output[i, j] = brightness.value
        data = pd.DataFrame(data=output.T, columns=composition, index=emissions)
        data.unit = u.R
        return data

    @staticmethod
    def _get_measurements(observations: list[Observation]):
        measurements = np.full((len(wavelengths), 2), fill_value=np.nan)
        for i, wavelength in enumerate(wavelengths):
            for obs in observations:
                if obs.wavelength == wavelength:
                    measurements[i, 0] = obs.brightness.value
                    measurements[i, 1] = obs.uncertainty.value
        return measurements

    def _make_basemodel(self,
                        electron_properties: ElectronProperties,
                        distance: u.Quantity) -> np.ndarray:
        """
        Construct initial model of brightness using a column density of
        10¹⁴ cm⁻² for each constituent species.
        """
        n_constituents = len(self._available_constituents)
        column_densities = np.ones(n_constituents) * self._base_column_density
        items = zip(self._available_constituents, column_densities)
        column_densities = {i: j for i, j in items}
        base_model = self.run(electron_properties=electron_properties,
                              distance=distance,
                              column_densities=column_densities)
        return base_model.to_numpy()

    # noinspection PyPep8Naming
    @staticmethod
    def _fit_func(base_model: np.ndarray,
                  O: float,
                  O2: float,
                  H2O: float,
                  CO2: float,
                  SO2: float) -> np.ndarray:
        coeffs = np.array([O, O2, H2O, CO2, SO2])
        return np.nansum(coeffs * base_model, axis=1)

    def _fit_model(self,
                   composition: Iterable[str]) -> Model:
        """
        LMFit model for calculating best-fit atmospheric composition using
        minimization.

        Parameters
        ----------
        composition : Iterable[str]
            A list of atmospheric constituents to model.

        Returns
        -------
        Model
            An LMFit `Model` instance.
        """
        model = Model(self._fit_func, independent_vars=['base_model'])
        for constituent in self._available_constituents:
            if not constituent in composition:
                model.set_param_hint(constituent, value=0.0, vary=False)
            else:
                model.set_param_hint(constituent, min=0.0)
        return model

    def fit(self,
            observations: list[Observation],
            electron_properties: ElectronProperties or str,
            composition: Iterable[str],
            distance: u.Quantity = 0 * u.km) -> pd.DataFrame:
        """
        Fit the brightnesses for a given atmospheric composition, estimating
        uncertainties with Markov chain Monte Carlo.

        Parameters
        ----------
        observations : list[Observation]
            A list of observed brightnesses as `Observation` objects. Must at
            least include wavelength, brightness and brightness uncertainty.
        electron_properties : ElectronProperties or str
            Ambient electron properties. Can be specified exactly using the
            `ElectronProperties` class, or you can provide the string 'Io',
            'Europa', 'Ganymede' or 'Callisto' to get their current canonical
            values. To see these properties, use the
            `get_canonical_electron_properties` function.
        composition : Iterable[str]
            A list of atmospheric constituents to model. Must be in the same
            format as those listed in the property `available_constituents`.
        distance : u.Quantity
            Distance from the centrifugal equator if you want to scale the
            densities using the Gaussian scale height. Set to 0 km by default,
            meaning no scaling takes place.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the uncertainties estimated from the MCMC
            run. Index labels are the emissions and column labels are the
            modeled atmospheric constituents. Note: the appropriate unit is
            stored in the property `unit`.
        """
        print('Running auroral emission model...')
        model = self._fit_model(composition)
        measurements = self._get_measurements(observations)
        ind = np.where(np.isfinite(measurements[:, 0]))[0]
        base_model = self._make_basemodel(electron_properties, distance)
        params = dict()
        for constituent in self._available_constituents:
            if constituent in composition:
                params[constituent] = 1.0
            else:
                params[constituent] = 0.0
        params = model.make_params(**params)
        fit_kwargs = dict(data=measurements[ind, 0],
                      params=params,
                      weights=1 / measurements[ind, 1],
                      base_model=base_model[ind])

        # minimizer result
        print('   Calculating best-fit...')
        minimizer_fit = model.fit(**fit_kwargs, method='powell')

        # MCMC uncertainties
        print('   Estimating uncertainties...')
        run_mcmc_kwargs = dict(progress_kwargs=dict(leave=False))
        thin = 4
        burn = 1000
        discard = burn
        emcee_kws = dict(burn=burn, steps=5000, thin=thin,
                         run_mcmc_kwargs=run_mcmc_kwargs)
        emcee_fit = model.fit(**fit_kwargs,
                              method='emcee',
                              fit_kws=emcee_kws)

        best_fit = dict()
        magnitude = self._base_column_density.value
        for species in composition:
            density = minimizer_fit.params[species].value
            density = density * magnitude
            try:
                unc_minimizer = minimizer_fit.params[species].stderr
                unc_minimizer = unc_minimizer * magnitude
            except TypeError:
                unc_minimizer = np.nan
            unc_mcmc = emcee_fit.params[species].stderr * magnitude
            best_fit[species] = (density, unc_minimizer, unc_mcmc)
        index = ['value_cm-2',
                 'minimizer_uncertainty_cm-2',
                 'mcmc_uncertainty_cm-2']
        df = pd.DataFrame(best_fit, index=index).T

        return df

    @property
    def available_constituents(self) -> np.ndarray:
        """
        The names of atmospheric constituents currently available in the model.

        Returns
        -------
        np.ndarray
        """
        return self._available_constituents

    @property
    def wavelengths(self) -> u.Quantity:
        """
        The wavelengths of emissions currently modeled as an Astropy `Quantity`
        object.

        Returns
        -------
        u.Quantity
        """
        return self._wavelengths

    @property
    def emissions(self) -> np.ndarray:
        """
        The names of emissions currently modeled in spectroscopic notation,
        e.g., '630.0 nm [O I]'.

        Returns
        -------
        np.ndarray
        """
        return self._emissions


def get_emission_rates(
        electron_properties: ElectronProperties or str) -> pd.DataFrame:
    """
    Wrapper function to calculate emission rate coefficients for a given set
    of electron properties.

    Parameters
    ----------
    electron_properties : ElectronProperties or str
        Ambient electron properties. Can be specified exactly using the
        `ElectronProperties` class, or you can provide the string 'Io',
        'Europa', 'Ganymede' or 'Callisto' to get their current canonical
        values. To see these properties, use the
        `get_canonical_electron_properties` function.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the emission rate coefficients. Index labels are
        the emissions and column labels are the atmospheric constituents
        available in the model. Note: the appropriate unit is stored in the
        property `unit`.
    """
    if isinstance(electron_properties, str):
        electron_properties = get_canonical_electron_properties(
            electron_properties)
    unit = u.cm**3 * u.ph / u.electron / u.s
    cross_sections = get_all_cross_sections()
    rates = dict()
    for i, species in enumerate(available_constituents):
        constituent_rates = []
        for j, emission in enumerate(emissions):
            for item in [' nm', ' I', '[', ']']:
                emission = emission.replace(item, '')
            wavelength, atom = emission.split(' ')
            key = f'{species} - {atom} {wavelength} nm'
            if key not in cross_sections.keys():
                constituent_rates.append(np.nan)
            else:
                xs = cross_sections[key]
                rate = xs.get_emission_rate_coefficient(electron_properties)
                constituent_rates.append(rate.to(unit).value)
        rates[species] = np.array(constituent_rates)
    df = pd.DataFrame(rates, index=emissions)
    df.unit = unit
    return df
