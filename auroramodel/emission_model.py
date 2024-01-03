import warnings
from pathlib import Path
from datetime import datetime, timezone

import astropy.constants as c
import astropy.units as u
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from hiresaurora.general import FuzzyQuantity
from lmfit.models import GaussianModel

from auroramodel.cross_sections import cross_sections, \
    ElectronEnergyDistribution, electron_energy_distributions
from auroramodel.general import parent_species as default_parent_species, \
    wavelengths, emissions, _log, _write_log

halfsigma = 0.317310507863
corner_quantile = [0.5 - halfsigma, 0.5, 0.5 + halfsigma]
calc_quantiles = np.array(corner_quantile) * 100


# noinspection PyUnresolvedReferences
class Observation:
    """
    Class to hold observed aurora brightness and uncertainties.
    """
    def __init__(self, wavelength: float, brightness: u.Quantity,
                 uncertainty: u.Quantity, z: u.Quantity = 0 * c.R_jup,
                 time: str or Time = None):
        """
        Parameters
        ----------
        wavelength: float
            The emission wavelength in nanometers.
        brightness : u.Quantity
            The observed brightness in rayleighs.
        uncertainty : u.Quantity
            The uncertainty in the observed brightness in rayleighs.
        """
        self._wavelength = wavelength
        self._brightness = brightness
        self._uncertainty = uncertainty
        self._z = z
        if time is not None:
            try:
                self._time = Time(time).strftime('%Y-%m-%d %H:%M UTC')
            except ValueError:
                self._time = time
        else:
            self._time = None

    def __str__(self):
        return f'{self._wavelength} nm: {self._brightness} ± ' \
               f'{self._uncertainty}'

    @property
    def wavelength(self) -> u.Quantity:
        return self._wavelength * u.nm

    @property
    def brightness(self) -> u.Quantity:
        return self._brightness

    @property
    def uncertainty(self) -> u.Quantity:
        return self._uncertainty

    @property
    def z(self) -> u.Quantity:
        return self._z.to(u.R_jup)

    @property
    def time(self) -> str:
        return self._time


class _BestFit:
    """
    Class to retrieve best-fit information from an MCMC chain.
    """
    def __init__(self, chain: np.ndarray, base_column_density: u.Quantity):
        self._chain = chain
        self._gaussian_fit = self._fit_gaussian()
        self._base_column_density = base_column_density
        self._value, self._uncertainty = self._get_best_fit()
        self._upper_limit = self._get_column_density_upper_limit()

    def __str__(self):
        value = float(self._value)
        uncertainty = float(self._uncertainty)
        fuzz = FuzzyQuantity(value, uncertainty)
        fuzz_ul = FuzzyQuantity(value + uncertainty, uncertainty)
        return f'{fuzz.printable} (upper limit: {fuzz_ul.value_printable})'

    def _fit_gaussian(self):
        hist, edges = np.histogram(self._chain, bins=1000)
        centers = edges[:-1] + np.diff(edges)
        model = GaussianModel()
        model.set_param_hint('center', min=0)
        params = model.guess(hist, x=centers)
        result = model.fit(hist, params=params, x=centers)
        return result.params

    def _get_column_density(self) -> float:
        """
        Return the median of the chain as the best-fit column density.
        """
        return self._gaussian_fit['center'].value * self._base_column_density

    def _get_column_density_uncertainty(self) -> float:
        """
        Return the standard deviation of the chain as the uncertainty on the
        best-fit column density.
        """
        return self._gaussian_fit['sigma'].value * self._base_column_density

    def _get_best_fit(self) -> (float, float):
        """
        Return the best-fit and uncertainty with appropriate significant
        figures.
        """
        value = self._get_column_density()
        uncertainty = self._get_column_density_uncertainty()
        fuzz = FuzzyQuantity(value, uncertainty)
        return fuzz.value_formatted, fuzz.uncertainty_formatted

    def _get_column_density_upper_limit(self) -> float:
        """
        Return the upper limit as the +1-sigma percentile, formatted to match
        the significant figures of the uncertainty.
        """
        value = self._get_column_density()
        uncertainty = self._get_column_density_uncertainty()
        fuzz = FuzzyQuantity(value + uncertainty, uncertainty)
        return float(fuzz.value_formatted)

    @property
    def value(self) -> float:
        return float(self._value)

    @property
    def uncertainty(self) -> float:
        return float(self._uncertainty)

    @property
    def upper_limit(self) -> float:
        return self._upper_limit

    @property
    def base_column_density(self) -> u.Quantity:
        return self._base_column_density

    @property
    def result(self) -> str:
        return self.__str__()


class EmissionModel:
    """
    Model target auroral emission.
    """
    def __init__(self,
                 electron_energy_dist: str or ElectronEnergyDistribution,
                 observations: [Observation] = None,
                 parent_species: [str] = default_parent_species,
                 scale_electron_density: bool = False):
        """
        Parameters
        ----------
        electron_energy_dist : str or ElectronEnergyDistribution
            Target satellite ('Io', 'Europa', 'Ganymede' or 'Callisto').
            Setting this parameter as a string for the target satellite
            ('Io', 'Europa', 'Ganymede' or 'Callisto') will load a standard
            electron distribution appropriate for that satellite.

            Alternatively, you can specify an electron distribution generated
            using the `ElectronEnergyDistribution` class available from the
            `cross_sections` module.

        observations : [Observation]
            A list of emission observations for fitting. Optional, since you
            can also just run the model in a theoretical context.
        parent_species : [str]
            A list of the parent species you want to evaluate (or fit). Options
            currently available are O, O₂, H₂O, CO2 and S+.
        scale_electron_density : bool
            Whether or not to apply the scale-heights from Bagenal and Delamere
            (2011) to the electron densities.
        """
        if isinstance(electron_energy_dist, str):
            self._target = electron_energy_dist
            self._electron_energy_distribution = \
                electron_energy_distributions[self._target]
        else:
            self._target = None
            self._electron_energy_distribution = electron_energy_dist
        self._observations = observations
        self._base_column_density = 1e14 / u.cm**2
        self._scale_electron_density = scale_electron_density
        if scale_electron_density:
            self._z = self._observations[0].z
        else:
            self._z = 0 * u.R_jup
        self._n_dim = len(parent_species)
        self._n_walkers = int(self._n_dim * 10)
        self._n_emissions = len(wavelengths)
        self._parent_species = parent_species

    def _calculate_surface_brightness(
            self, column_density: u.Quantity, rate: u.Quantity) -> u.Quantity:
        """
        Calculate expected surface brightness for a given parent species column
        density and emission rate coefficient.

        Parameters
        ----------
        column_density : u.Quantity
            The parent species atmospheric column density in [1/cm²].
        rate : u.Quantity
            The emission rate coefficient in [cm³/s].


        Returns
        -------
        The surface brightness in [R].
        """
        conversion = 1 * u.ph / u.electron
        column_emission = (self._electron_energy_distribution.ne(z=self._z)
                           * column_density * rate * conversion)
        try:
            return (column_emission / (4 * np.pi * u.sr)).to(u.R)
        except u.core.UnitConversionError:
            return (column_emission * u.electron / (4 * np.pi * u.sr)).to(u.R)

    # noinspection PyUnresolvedReferences
    def eval(self, species: [str],
             column_densities: [u.Quantity]) -> u.Quantity:
        """
        Evaluate the model for supplied column densities.

        Parameters
        ----------
        species : [str]
            Species associated with the column densities.
        column_densities : [u.Quantity]
            The column densities associated with each of the species.
        """
        try:
            iter(column_densities)
        except TypeError:
            column_densities = [column_densities]
        model = np.full((self._n_dim, self._n_emissions), fill_value=np.nan)
        for i, parent in enumerate(species):
            for j, wavelength in enumerate(wavelengths):
                try:
                    xs = cross_sections[f'{parent}_{wavelength}nm']
                    rate = xs.get_rate(self._electron_energy_distribution)
                    brightness = self._calculate_surface_brightness(
                        column_density=column_densities[i], rate=rate)
                    model[i, j] = brightness.value
                except KeyError:
                    continue
        return model * u.R

    def _make_basemodel(self):
        """
        Construct initial model using a column density of 10¹⁴/cm² for each
        parent species.
        """
        species = self._parent_species
        column_densities = np.ones(len(species)) * self._base_column_density
        base_model = self.eval(species=self._parent_species,
                               column_densities=column_densities)
        return base_model.value

    def _get_observations(self):
        """
        Make an n×2 array of measured brightnesses and uncertainties for each
        observed wavelength.
        """
        measurements = np.full((len(wavelengths), 2), fill_value=np.nan)
        for i, wavelength in enumerate(wavelengths):
            for obs in self._observations:
                if obs.wavelength.value == wavelength:
                    measurements[i, 0] = obs.brightness.value
                    measurements[i, 1] = obs.uncertainty.value
        return measurements

    @staticmethod
    def _log_likelihood(theta, measurements, base_model):
        """
        Construct MCMC log likelihood.
        """
        full_model = base_model.T * theta
        chisq = (np.nansum(full_model, axis=1)
                 - measurements[:, 0])**2 / measurements[:, 1]**2
        return -0.5 * np.nansum(chisq)

    @staticmethod
    def _log_prior(theta: np.ndarray):
        """
        Construct MCMC log prior.
        """
        tmp = 1
        for i in range(theta.size):
            if 1e-6 < theta[i] < 1e6:
                continue
            else:
                tmp *= 0
        if tmp == 1:
            return 0.0
        else:
            return -np.inf

    def _log_probability(self, theta, measurements, base_model):
        """
        Construct MCMC log probability.
        """
        log_prior = self._log_prior(theta=theta)
        if not np.isfinite(log_prior):
            return -np.inf
        else:
            return log_prior + self._log_likelihood(
                theta, measurements, base_model)

    # noinspection PyTypeChecker
    def _run_mcmc(self, nsteps, nwalkers, progress: bool = True,
                  progress_kwargs: dict = None):
        """"
        Run MCMC to estimate best-fit atmosphere and uncertainties.
        """
        if nwalkers is None:
            nwalkers = self._n_walkers
        measurements = self._get_observations()
        base_model = self._make_basemodel()
        pos = 1 + np.random.randn(nwalkers, self._n_dim) / 10
        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers, ndim=self._n_dim,
            log_prob_fn=self._log_probability, threads=5,
            args=(measurements, base_model,))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            sampler.run_mcmc(pos, nsteps, progress=progress,
                             progress_kwargs=progress_kwargs)
        return sampler

    def _get_best_fit_atmosphere(self, chain: np.ndarray) -> [_BestFit]:
        """
        Take the output chain from the MCMC simulation and get the best-fit
        atmosphere, uncertainties and upper limits.
        """
        best_fits = []
        labels = [parent.replace('2', '₂') for parent in self._parent_species]
        print('\nBest-fit atmosphere:')
        for i in range(self._n_dim):
            best_fit = _BestFit(chain=chain[:, i],
                                base_column_density=self._base_column_density)
            print(f'   {labels[i]}: {best_fit.result}')
            best_fits.append(best_fit)
        return best_fits

    def _save_best_fit_column_density(
            self, samples: np.ndarray, output_directory: str or Path,
            prefix: str) -> None:
        """
        Save the best-fit results in a CSV.
        """
        magnitude = self._base_column_density.value
        quantiles = np.array(self._calculate_quantiles(samples)) * magnitude
        sigma_lowers = []
        sigma_uppers = []
        medians = []
        for quantile in quantiles:
            fuzz_lower = FuzzyQuantity(quantile[1], quantile[0])
            fuzz_upper = FuzzyQuantity(quantile[1], quantile[2])
            if (len(fuzz_lower.value_formatted)
                    > len(fuzz_upper.value_formatted)):
                median = fuzz_lower.value_formatted
            else:
                median = fuzz_upper.value_formatted
            ll = fuzz_lower.uncertainty_formatted
            ul = fuzz_upper.uncertainty_formatted
            sigma_lowers.append(ll)
            sigma_uppers.append(ul)
            medians.append(median)

        data = {
            'species': self._parent_species,
            'column_density_cm-2': medians,
            'column_density_sigma_lower_cm-2': sigma_lowers,
            'column_density_sigma_upper_cm-2': sigma_uppers,
        }
        df = pd.DataFrame(data=data)
        savename = Path(output_directory,
                        f'{prefix}best_fit_column_density.csv')
        df.to_csv(savename, index=False)

    def _save_autocorrelation(
            self, sampler: emcee.EnsembleSampler, discard: int, labels: [str],
            output_directory: str or Path, prefix: str):
        """
        Save plots of chain parameters so I can see if they've converged.
        """
        labels = [fr"{label.replace('₂', '$_2$')}" for label in labels]
        samples = sampler.get_chain() * self._base_column_density
        n = len(labels)
        fig, axes = plt.subplots(n, figsize=(4.5, 1 * n), sharex='all',
                                 layout='constrained', clear=True)
        log = True
        if n == 1:
            axes = [axes]
            log = False
        for i in range(n):
            axes[i].plot(samples[:, :, i], "k", alpha=0.25, linewidth=0.25)
            axes[i].axvspan(0, discard, color='grey', alpha=0.25, linewidth=0)
            axes[i].set_xlim(0, len(samples))
            if log:
                axes[i].set_yscale('log')
                ydiff = np.diff(np.log10(axes[i].get_ylim()))[0]
                fig.canvas.draw()
                if ydiff < 1:
                    axes[i].set_yscale('linear')
                    axes[i].set_ylim(bottom=0)
            axes[i].set_ylabel(labels[i])
        axes[-1].set_xlabel("Step Number")
        fig.supylabel(r'Column Density [$\mathrm{{cm^{{-2}}}}$]')
        savename = Path(output_directory, f'{prefix}mcmc_autocorrelation.jpg')
        plt.savefig(savename)
        plt.close(fig)

    @staticmethod
    def _calculate_quantiles(samples: np.ndarray,
                             quantiles: [float] = calc_quantiles) -> [[float]]:
        calculated_quantiles = []
        for i in range(samples.shape[1]):
            bounds = np.percentile(samples[:, i], quantiles)
            lower = bounds[1] - bounds[0]
            upper = bounds[2] - bounds[1]
            fuzz_lower = FuzzyQuantity(bounds[1], lower)
            fuzz_upper = FuzzyQuantity(bounds[1], upper)
            if (len(fuzz_lower.value_formatted)
                    > len(fuzz_upper.value_formatted)):
                median = float(fuzz_lower.value_formatted)
            else:
                median = float(fuzz_upper.value_formatted)
            ll = float(fuzz_lower.uncertainty_formatted)
            ul = float(fuzz_upper.uncertainty_formatted)
            calculated_quantiles.append([ll, median, ul])
        return calculated_quantiles

    def _make_corner_plot_titles(self, samples: np.ndarray, labels: [str]):
        titles = []
        quantiles = self._calculate_quantiles(samples=samples)
        magnitude = self._base_column_density.value
        for i in range(len(quantiles)):
            ll = quantiles[i][0] / magnitude
            median = quantiles[i][1] / magnitude
            ul = quantiles[i][2] / magnitude
            label = labels[i].replace('$', '')
            title = fr'$\mathrm{{{label}}} = '
            title += fr'\left({median}_{{-{ll}}}^{{+{ul}}}\right) '
            title += fr'\times 10^{{{np.log10(magnitude):.0f}}}'
            title += fr'\,\mathrm{{cm^{{-2}}}}$'
            titles.append(title)
        return titles

    def _save_corner_plot(self, samples: np.ndarray, labels: [str],
                          output_directory: str or Path, prefix: str):
        """
        Save MCMC results corner plot.
        """
        samples *= self._base_column_density.value
        labels = [label.replace('₂', r'$_2$') for label in labels]
        label_kwargs = dict(va='baseline', ha='center')
        titles = self._make_corner_plot_titles(samples=samples, labels=labels)
        fig = plt.figure(figsize=(6.5, 6.5 + 0.2 * (self._n_dim - 1)),
                         clear=True, layout='constrained')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            corner.corner(samples, labels=labels, label_kwargs=label_kwargs,
                          fig=fig, use_math_text=True, axes_scale='log',
                          quiet=True)
        axes = np.reshape(fig.axes, (self._n_dim, self._n_dim))
        for i in range(self._n_dim):
            for j in range(self._n_dim):
                if i == j:
                    axes[i, j].set_title(titles[i])
                if j <= i:
                    xdiff = np.diff(np.log10(axes[i, j].get_xlim()))[0]
                    if xdiff < 1:
                        axes[i, j].set_xscale('linear')
                    if i < self._n_dim - 1:
                        axes[i, j].xaxis.set_tick_params(
                            which='both', labelbottom=False)
                    else:
                        axes[i, j].xaxis.set_tick_params(
                            which='both', labelbottom=True)
                if j < i:
                    ydiff = np.diff(np.log10(axes[i, j].get_ylim()))[0]
                    if ydiff < 1:
                        axes[i, j].set_yscale('linear')
        for axis in axes.ravel():
            [i.set_rotation(0) for i in axis.get_xticklabels()]
            [i.set_rotation(0) for i in axis.get_xticklabels(minor=True)]
            [i.set_rotation(0) for i in axis.get_yticklabels()]
            [i.set_rotation(0) for i in axis.get_yticklabels(minor=True)]
        savename = Path(output_directory, f'{prefix}mcmc_fit_results.jpg')
        fig.canvas.draw()
        plt.savefig(savename)
        plt.close(fig)

    def _save_best_fit_emission(
            self, samples: np.ndarray, output_directory: str or Path,
            prefix: str):
        """
        Retrieve the maximum-likelihood model from the MCMC sampler and save it
        as a CSV.
        """
        quantiles = np.array(
            self._calculate_quantiles(samples)) * self._base_column_density
        model_lower = self.eval(species=self._parent_species,
                                column_densities=quantiles[:, 0])
        model_median = self.eval(species=self._parent_species,
                                 column_densities=quantiles[:, 1])
        model_upper = self.eval(species=self._parent_species,
                                column_densities=quantiles[:, 2])
        total_lower = np.nansum(model_median - model_lower, axis=0)
        total_median = np.nansum(model_median, axis=0)
        total_upper = np.nansum(model_median + model_upper, axis=0)
        savename = Path(
            output_directory, f'{prefix}best_fit_emission.csv')
        with open(savename, 'w') as file:
            file.write('transition,wavelength_nm')
            if self._observations is not None:
                file.write(',data_R,uncertainty_R')
            file.write(',lower_limit_R,median_R,upper_limit_R\n')
            for i in range(len(wavelengths)):
                wavelength = wavelengths[i]
                lower = total_lower[i].value
                median = total_median[i].value
                upper = total_upper[i].value
                parent = [j for j in emissions if str(wavelength) in j][0]
                if self._observations is not None:
                    observed = [
                        obs.brightness.value for obs in self._observations
                        if obs.wavelength.value == wavelength]
                    uncertainty = [
                        obs.uncertainty.value for obs in self._observations
                        if obs.wavelength.value == wavelength]
                    if len(observed) == 0:
                        observed = '---'
                        uncertainty = '---'
                    else:
                        observed = observed[0]
                        uncertainty = uncertainty[0]
                if median < 10:
                    fmt = '#.2g'
                else:
                    fmt = '.0f'
                file.write(f'{parent},{wavelength}')
                if self._observations is not None:
                    if observed != '---':
                        file.write(f',{observed:{fmt}},{uncertainty:{fmt}}')
                    else:
                        file.write(f',{observed},{uncertainty}')
                file.write(f',{lower:{fmt}},{median:{fmt}},{upper:{fmt}}\n')

    def run(self, output_directory: str or Path, prefix: str = None,
            nsteps: int = 5000, nwalkers: int = None, iteration: int = None,
            count: int = None):
        """
        Run the aurora model MCMC fit for the specified atmospheric constituent
        species.

        Parameters
        ----------
        output_directory : str or Path
            The directory where you want the results saved.
        prefix : str
            Add a file name here if you want it added to the beginning of the
            output file names.
        nsteps : int
            The number of steps in the MCMC simulation. Default is 5000.
        nwalkers : int
            The number of walkers in the MCMC simulation. Default is None,
            which will tell the sampler to use 10-times the number of species.
        iteration : int
            Iteration number (if you want it reflected in the printed progress
            bar).
        count: int
            Total number of iterations (if you want it reflected in the printed
            progress bar).
        """
        t0 = datetime.now(timezone.utc)
        log = []
        if prefix is not None:
            prefix = f'{prefix}_'
        else:
            prefix = ''
        labels = [parent.replace('2', '₂') for parent in self._parent_species]
        _log(log, '\nModeling aurora emission...')
        if self._target is not None:
            _log(log, f'   Target: {self._target}')
        else:
            _log(log, f'   Electron energy distribution: '
                      f'{self._electron_energy_distribution.eV}')
        if self._observations[0].time is not None:
            _log(log, f'   Time: {self._observations[0].time}')
        _log(log, f'   Atmospheric composition: {", ".join(labels)}')
        _log(log, f'   Distance from plasma sheet: {self._z.value} Rⱼ')
        rho = self._electron_energy_distribution.ne(z=self._z).value
        _log(log, f'   Plasma density: {rho:#.2g} e⁻/cm³')

        # run the full MCMC
        if (iteration is not None) and (count is not None):
            desc = f'   MCMC fit {iteration + 1}/{count}'
        else:
            desc = '   MCMC fit'
        progress_kwargs = dict(leave=False, desc=desc)
        sampler = self._run_mcmc(nsteps=nsteps, nwalkers=nwalkers,
                                 progress_kwargs=progress_kwargs)

        try:
            tau = np.max(sampler.get_autocorr_time())
            discard = int(4 * tau)
        except emcee.autocorr.AutocorrError:
            tau = int(nsteps / 4)
            discard = int(2 * tau)
        thin = int(tau / 2)
        samples = sampler.get_chain(discard=discard, thin=thin,
                                    flat=True)

        # calculate best fits and save results
        _log(log, 'Saving best-fit column densities...')
        self._save_best_fit_column_density(
            samples=samples, output_directory=output_directory, prefix=prefix)
        _log(log, 'Saving maximum likelihood emission model...')
        self._save_best_fit_emission(
            samples=samples, output_directory=output_directory, prefix=prefix)
        _log(log, 'Saving autocorrelation plots...')
        self._save_autocorrelation(
            sampler=sampler, discard=discard, labels=labels,
            output_directory=output_directory, prefix=prefix)
        _log(log, 'Saving corner plot...')
        self._save_corner_plot(
            samples=samples, labels=labels, output_directory=output_directory,
            prefix=prefix)

        # clear several lines from terminal output
        _log(log, f'Modeling complete, time elapsed '
                  f'{datetime.now(timezone.utc) - t0}.')
        _write_log(Path(output_directory, f'{prefix}log.txt'), log)
