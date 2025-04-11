import warnings
from pathlib import Path

import astropy.units as u
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from hiresaurora.general import FuzzyQuantity

from auroramodel.cross_sections import cross_sections
from auroramodel.electrons import ElectronProperties, \
    default_electron_properties
from auroramodel.general import parent_species as default_parent_species, \
    wavelengths, emissions, calculate_surface_brightness
from auroramodel.observations import Observation

halfsigma = 0.317310507863
calc_quantiles = np.array([0.5 - halfsigma, 0.5, 0.5 + halfsigma]) * 100


def _calculate_quantiles(maximum_likelihood_values: np.ndarray,
                         samples: np.ndarray,
                         magnitude: float,
                         quantiles: [float] = calc_quantiles) -> np.ndarray:
    """
    Array values are 16th, maximum likelihood and 84th percentile values
    rounded to appropriate significant figures.
    """
    calculated_quantiles = []
    samples = samples * magnitude
    for i in range(samples.shape[1]):
        bounds = np.percentile(samples[:, i], quantiles)
        lower, upper = np.diff(bounds)
        median = maximum_likelihood_values[i] * magnitude
        fuzz_lower = FuzzyQuantity(median, lower)
        fuzz_upper = FuzzyQuantity(median, upper)
        if (len(fuzz_lower.value_formatted)
                > len(fuzz_upper.value_formatted)):
            ml = str(fuzz_lower.value_formatted)
        else:
            ml = str(fuzz_upper.value_formatted)
        ll = str(fuzz_lower.uncertainty_formatted)
        ul = str(fuzz_upper.uncertainty_formatted)
        calculated_quantiles.append([ll, ml, ul])
    return np.array(calculated_quantiles)


class MCMC:
    """
    Explore atmosphere posterior parameter space using MCMC.
    """
    def __init__(self,
                 target: str = None,
                 observations: list[Observation] = None,
                 parent_species: list[str] = default_parent_species):
        """
        Parameters
        ----------
        observations : list[Observation]
            A list of emission observations for fitting. Optional, since you
            can also just run the model in a theoretical context.
        parent_species : list[str]
            A list of the parent species you want to evaluate (or fit).
        """
        self._target = target
        self._observations = observations
        self._base_column_density = 1e14 / u.cm**2
        self._n_emissions = len(wavelengths)
        self._parent_species = parent_species
        self._n_species = len(parent_species)

    # noinspection PyUnresolvedReferences
    def _eval_grid(self,
                   electron_properties: ElectronProperties,
                   z: u.Quantity,
                   column_densities: [u.Quantity]) -> u.Quantity:
        """
        Evaluate the model for supplied column densities.

        Parameters
        ----------
        electron_density : u.Quantity
            Number density of the electrons exciting the emission in [cm⁻³].
        column_densities : [u.Quantity]
            The column densities associated with each of the species in [cm⁻²].
        """
        try:
            iter(column_densities)
        except TypeError:
            column_densities = [column_densities]
        n_dim = len(self._parent_species)
        electron_density = electron_properties.ne(z=z)
        output = np.full((n_dim, self._n_emissions), fill_value=np.nan)
        for i, parent in enumerate(self._parent_species):
            for j, wavelength in enumerate(wavelengths):
                key = f'{parent} {wavelength.value} nm'
                if key not in cross_sections.keys():
                    continue
                else:
                    xs = cross_sections[key]
                    rate = xs.get_rate(electron_properties)
                    brightness = calculate_surface_brightness(
                        electron_density=electron_density,
                        column_density=column_densities[i],
                        rate=rate)
                    output[i, j] = brightness.value
        return output * u.R

    def _make_basemodel(self,
                        electron_properties: ElectronProperties,
                        z: u.Quantity) -> u.Quantity:
        """
        Construct initial model of brightness using a column density of
        10¹⁴ cm⁻² for each parent species.
        """
        column_densities = np.ones(self._n_species) * self._base_column_density
        base_model = self._eval_grid(electron_properties=electron_properties,
                                     z=z,
                                     column_densities=column_densities)
        return base_model

    def _get_observations(self) -> np.ndarray:
        """
        Make an n×2 array of measured brightnesses and uncertainties for each
        observed wavelength.
        """
        measurements = np.full((len(wavelengths), 2), fill_value=np.nan)
        for i, wavelength in enumerate(wavelengths):
            for obs in self._observations:
                if obs.wavelength == wavelength:
                    measurements[i, 0] = obs.brightness.value
                    measurements[i, 1] = obs.uncertainty.value
        return measurements

    @staticmethod
    def _log_likelihood(theta: np.ndarray,
                        measurements: np.ndarray,
                        base_model: np.ndarray) -> float:
        """
        Construct MCMC log likelihood.
        """
        full_model = base_model.T * theta
        chisq = (np.nansum(full_model, axis=1)
                 - measurements[:, 0])**2 / measurements[:, 1]**2
        return -0.5 * np.nansum(chisq)

    @staticmethod
    def _log_prior(theta: np.ndarray) -> float:
        """
        Construct MCMC log prior.
        """
        tmp = 1
        for i in theta:
            if 1e-6 <= i < 1e6:
                continue
            else:
                tmp *= 0
        if tmp == 1:
            return 0.0
        else:
            return -np.inf

    def _log_probability(self,
                         theta: np.ndarray,
                         measurements: np.ndarray,
                         base_model: np.ndarray) -> float:
        """
        Construct MCMC log probability.
        """
        log_prior = self._log_prior(theta=theta)
        log_likelihood = self._log_likelihood(theta, measurements, base_model)
        if not np.isfinite(log_prior):
            return -np.inf
        else:
            return log_prior + log_likelihood

    # noinspection PyTypeChecker
    def _run_mcmc(self,
                  electron_properties: ElectronProperties,
                  z: u.Quantity,
                  n_steps,
                  n_walkers,
                  progress: bool = True,
                  progress_kwargs: dict = None):
        """"
        Run MCMC to estimate posterior distributions of atmospheric species.
        """
        measurements = self._get_observations()
        base_model = self._make_basemodel(
            electron_properties=electron_properties,
            z=z)
        n_dim = len(self._parent_species)
        pos = 1 + np.random.randn(n_walkers, n_dim) / 100
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers,
                                        ndim=n_dim,
                                        log_prob_fn=self._log_probability,
                                        threads=10,
                                        args=(measurements, base_model.value))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            sampler.run_mcmc(pos,
                             n_steps,
                             progress=progress,
                             progress_kwargs=progress_kwargs)
        return sampler

    def _calculate_quantile_densities(
            self,
            samples: np.ndarray,
            truths: np.ndarray) -> tuple[dict, np.ndarray]:
        """
        Calculate the quintile species densities.
        """

        # quantiles
        magnitude = self._base_column_density.value
        quantiles = _calculate_quantiles(truths, samples, magnitude)
        sigma_lowers = []
        max_likelihoods = []
        sigma_uppers = []
        for q in quantiles:
            sigma_lowers.append(q[0])
            max_likelihoods.append(q[1])
            sigma_uppers.append(q[2])

        data = {
            'species': np.array(self._parent_species),
            'column_density_cm-2': np.array(max_likelihoods),
            'column_density_sigma_lower_cm-2': np.array(sigma_lowers),
            'column_density_sigma_upper_cm-2': np.array(sigma_uppers),
        }

        return data, quantiles

    @staticmethod
    def _save_quintile_densities(densities: dict,
                                 output_directory: str or Path,
                                 prefix: str):
        """
        Save the best-fit results in a CSV.
        """
        savename = Path(output_directory,
                        f'{prefix}best_fit_column_density.csv')
        with open(savename, 'w') as file:
            file.write(','.join(densities.keys()) + '\n')
            for i in range(densities['species'].shape[0]):
                vals = []
                for val in densities.values():
                    vals.append(val[i])
                file.write(','.join(vals) + '\n')

    def _save_autocorrelation(self,
                              sampler: emcee.EnsembleSampler,
                              discard: int,
                              labels: [str],
                              output_directory: str or Path,
                              prefix: str) -> None:
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

    def _make_corner_plot_titles(self,
                                 quantiles,
                                 labels: [str]) -> list[str]:
        titles = []
        magnitude = self._base_column_density.value
        for i in range(len(quantiles)):
            ll = float(quantiles[i][0]) / magnitude
            ml = float(quantiles[i][1]) / magnitude
            ul = float(quantiles[i][2]) / magnitude
            label = labels[i].replace('$', '')
            title = fr'$\mathrm{{{label}}} = '
            title += fr'\left({ml}_{{-{ll}}}^{{+{ul}}}\right) '
            title += fr'\times 10^{{{np.log10(magnitude):.0f}}}'
            title += fr'\,\mathrm{{cm^{{-2}}}}$'
            titles.append(title)
        return titles

    def _save_corner_plot(self,
                          quantiles: np.ndarray,
                          samples: np.ndarray,
                          truths: np.ndarray,
                          labels: [str],
                          output_directory: str or Path,
                          prefix: str):
        """
        Save MCMC results corner plot.
        """
        n_dim = len(self._parent_species)
        samples = samples * self._base_column_density.value
        labels = [label.replace('₂', r'$_2$') for label in labels]
        label_kwargs = dict(va='baseline', ha='center')
        titles = self._make_corner_plot_titles(quantiles=quantiles,
                                               labels=labels)
        fig = plt.figure(figsize=(6, 6 + 0.2 * (n_dim - 1)),
                         clear=True,
                         layout='constrained')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            corner.corner(samples,
                          truths=truths.astype(float),
                          truth_color='k',
                          labels=labels,
                          label_kwargs=label_kwargs,
                          fig=fig,
                          use_math_text=True,
                          axes_scale='log',
                          quiet=True,
                          alpha=0.25)
        n_dim = len(self._parent_species)
        axes: np.ndarray[plt.Axes] = np.reshape(fig.axes, (n_dim, n_dim))  # noqa
        for i in range(n_dim):
            for j in range(n_dim):
                if i == j:
                    axes[i, j].set_title(titles[i])
                if j <= i:
                    xdiff = np.diff(np.log10(axes[i, j].get_xlim()))[0]
                    if xdiff < 1:
                        axes[i, j].set_xscale('linear')
                    if i < n_dim - 1:
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

    def _save_quantile_emissions(self,
                                 electron_properties: ElectronProperties,
                                 z: u.Quantity,
                                 quantiles: np.ndarray,
                                 output_directory: str or Path,
                                 prefix: str):
        """
        Retrieve the median model from the MCMC sampler and save it
        as a CSV.
        """
        quantiles = quantiles.astype(float) * self._base_column_density.unit
        model_lower = self._eval_grid(electron_properties=electron_properties,
                                      z=z,
                                      column_densities=quantiles[:, 0])
        model_median = self._eval_grid(electron_properties=electron_properties,
                                       z=z,
                                       column_densities=quantiles[:, 1])
        model_upper = self._eval_grid(electron_properties=electron_properties,
                                      z=z,
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
                wavelength = wavelengths[i].value
                lower = total_lower[i].value
                median = total_median[i].value
                upper = total_upper[i].value
                parent = [j for j in emissions if str(wavelength) in j][0]
                if self._observations is not None:
                    observed = []
                    uncertainty = []
                    for obs in self._observations:
                        if obs.wavelength.value == wavelength:
                            fuzz = FuzzyQuantity(obs.brightness.value,
                                                 obs.uncertainty.value)
                            observed.append(fuzz.value_formatted)
                            uncertainty.append(fuzz.uncertainty_formatted)
                    if len(observed) == 0:
                        observed = '---'
                        uncertainty = '---'
                    else:
                        observed = observed[0]
                        uncertainty = uncertainty[0]
                        if uncertainty[0] == '1':
                            fmt_obs = '#.2g'
                        else:
                            fmt_obs = '#.1g'
                if median < 10:
                    fmt = '#.2g'
                else:
                    fmt = '.0f'
                file.write(f'{parent},{wavelength}')
                if self._observations is not None:
                    if observed != '---':
                        file.write(f',{float(observed)},'
                                   f'{float(uncertainty):{fmt_obs}}')
                    else:
                        file.write(f',{observed},{uncertainty}')
                file.write(f',{lower:{fmt}},{median:{fmt}},{upper:{fmt}}\n')

    def run(self,
            electron_properties: str or ElectronProperties,
            scale_electron_density: bool,
            output_directory: str or Path = None,
            prefix: str = None,
            n_steps: int = 10000,
            n_walkers: int = None,
            iteration: int = None,
            count: int = None,
            silent: bool = False) -> dict:
        """
        Run the aurora model MCMC for the specified atmospheric constituent
        species.

        Parameters
        ----------
        electron_properties : str or ElectronProperties
            Properties of the electrons exciting the emission. If you pass the
            string `"Io"`, `"Europa"`, `"Ganymede"` or `"Callisto"` it will use
            the canonical properties for each satellite.
        scale_electron_density : bool
            Whether or not to apply the scale-heights in the electron
            properties to the electron densities.
        output_directory : str or Path
            The directory where you want the results saved. If none specified
            and `save_outputs` is set to `True`, the results will be saved in
            the current working directory.
        prefix : str
            Add a file name here if you want it added to the beginning of the
            output file names.
        n_steps : int
            The number of steps in the MCMC simulation. Default is 10,000.
        n_walkers : int
            The number of walkers in the MCMC simulation. Default is None,
            which will tell the sampler to use 10-times the number of species.
        iteration : int
            Iteration number (if you want it reflected in the printed progress
            bar).
        count: int
            Total number of iterations (if you want it reflected in the printed
            progress bar).
        silent: bool
            Whether or not to suppress printing of the terminal output.
        """
        # parse electron properties
        if isinstance(electron_properties, str):
            electron_properties = default_electron_properties[self._target]

        # calculate electron density
        if scale_electron_density:
            z = self._observations[0].z
        else:
            z = 0 * u.R_jup

        # determine number of walkers
        if n_walkers is None:
            n_walkers = int(len(self._parent_species) * 10)

        # apply prefix
        if prefix is not None:
            prefix = f'{prefix}_'
        else:
            prefix = ''

        # add unicode subscript to species names
        labels = [parent.replace('2', '₂') for parent in self._parent_species]

        # run the full MCMC
        if (iteration is not None) and (count is not None):
            desc = f'   MCMC run {iteration + 1}/{count}'
        else:
            desc = '   MCMC run'
        progress_kwargs = dict(leave=False, desc=desc)
        if silent:
            progress_kwargs['disable'] = True
        sampler = self._run_mcmc(electron_properties=electron_properties,
                                 z=z,
                                 n_steps=n_steps,
                                 n_walkers=n_walkers,
                                 progress_kwargs=progress_kwargs)
        try:
            tau = np.max(sampler.get_autocorr_time())
            discard = int(4 * tau)
        except emcee.autocorr.AutocorrError:
            tau = int(n_steps / 4)
            discard = int(2 * tau)
        thin = int(tau / 2)
        samples = sampler.get_chain(discard=discard,
                                    thin=thin,
                                    flat=True)
        truths = np.median(sampler.flatchain, axis=0)

        # get column density quintiles
        densities, quantiles = self._calculate_quantile_densities(
            samples=samples,
            truths=truths)

        # save graphics and output CSVs if an output path provided
        if output_directory is not None:
            self._save_autocorrelation(sampler=sampler,
                                       discard=discard,
                                       labels=labels,
                                       output_directory=output_directory,
                                       prefix=prefix)
            self._save_quintile_densities(
                densities=densities,
                output_directory=output_directory,
                prefix=prefix)
            self._save_quantile_emissions(
                electron_properties=electron_properties,
                z=z,
                quantiles=quantiles,
                output_directory=output_directory,
                prefix=prefix)

            self._save_corner_plot(quantiles=quantiles,
                                   samples=samples,
                                   truths=quantiles[:, 1],
                                   labels=labels,
                                   output_directory=output_directory,
                                   prefix=prefix)

        return densities
