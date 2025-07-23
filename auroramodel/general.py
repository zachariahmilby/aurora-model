from pathlib import Path

import astropy.units as u
import numpy as np

_package_directory = Path(__file__).resolve().parent

rcparams = Path(_package_directory, 'anc', 'rcparams.mplstyle')

color_dict = {'red': '#D62728', 'orange': '#FF7F0E', 'yellow': '#FDB813',
              'green': '#2CA02C', 'blue': '#0079C1', 'violet': '#9467BD',
              'cyan': '#17BECF', 'magenta': '#D64ECF', 'brown': '#8C564B',
              'darkgrey': '#3F3F3F', 'grey': '#7F7F7F', 'lightgrey': '#BFBFBF'}

# atmospheric constituents currently available in the model
available_constituents = np.array(['O', 'O2', 'H2O', 'CO2', 'SO2'])

# wavelengths of the various transitions
wavelengths = u.Quantity([121.6,
                          130.4,
                          135.6,
                          297.2,
                          486.1,
                          557.7,
                          630.0,
                          636.4,
                          656.3,
                          777.4,
                          844.6], u.nm)

# emissions associated with each wavelength
emissions = np.array(['121.6 nm H I',
                      '130.4 nm O I',
                      '135.6 nm O I',
                      '297.2 nm [O I]',
                      '486.1 nm H I',
                      '557.7 nm [O I]',
                      '630.0 nm [O I]',
                      '636.4 nm [O I]',
                      '656.3 nm H I',
                      '777.4 nm O I',
                      '844.6 nm O I'])


def _calculate_surface_brightness(electron_density: u.Quantity,
                                  column_density: u.Quantity,
                                  rate: u.Quantity) -> u.Quantity:
    """
    Calculate expected surface brightness for a given parent species column
    density and emission rate coefficient.

    Parameters
    ----------
    electron_density : u.Quantity
        Number density of the electrons exciting the emission in [cm⁻³].
    column_density : u.Quantity
        The parent species atmospheric column density in [cm⁻²].
    rate : u.Quantity
        The emission rate coefficient for a particular parent species in
        [cm³/s].

    Returns
    -------
    The surface brightness in [R].
    """
    column_emission = electron_density * column_density * rate
    column_emission = column_emission
    try:
        brightness = (column_emission / (4 * np.pi * u.sr)).to(u.R)
    except u.core.UnitConversionError:
        brightness = (column_emission * u.electron / (4 * np.pi * u.sr)).to(u.R)
    return brightness
