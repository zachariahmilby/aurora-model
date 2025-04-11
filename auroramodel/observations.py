import numpy as np
from astropy import units as u
from astropy.time import Time


class Observation:
    """
    Class to hold observed aurora brightness and uncertainties.
    """
    def __init__(self,
                 wavelength: u.Quantity,
                 brightness: u.Quantity,
                 uncertainty: u.Quantity,
                 z: u.Quantity = 0 * u.R_jup,
                 time: str or Time = None,
                 systematic_uncertainty: float = 0.0):
        """
        Parameters
        ----------
        wavelength: u.Quantity
            The emission wavelength in nanometers.
        brightness : u.Quantity
            The observed brightness in rayleighs.
        uncertainty : u.Quantity
            The uncertainty in the observed brightness in rayleighs.
        z : u.Quantity, optional
            The distance of the target satellite from the plasma sheet
            centrifugal equator at the time of the observation.
        time : str or Time, optional
            The UTC time of the observation.
        systematic_uncertainty : float, optional
            Additional uncertainty as a fraction of the observed brightness.
            For example, passing 0.09 would mean a 9% systematic uncertainty.
        """
        self._wavelength = wavelength
        self._brightness = brightness
        self._uncertainty = np.sqrt(
            uncertainty**2 + (systematic_uncertainty*brightness)**2)
        self._z = z
        if time is not None:
            try:
                self._time = Time(time).strftime('%Y-%m-%d %H:%M UTC')
            except ValueError:
                self._time = time
        else:
            self._time = None

    def __str__(self):
        return f'{self._wavelength.value} nm: {self._brightness.value} ± ' \
               f'{self._uncertainty.value} R'

    @property
    def wavelength(self) -> u.Quantity:
        return self._wavelength

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
