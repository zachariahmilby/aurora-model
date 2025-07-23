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
                 distance: u.Quantity = 0 * u.km,
                 time: str or Time = None,
                 systematic_uncertainty: float or u.Quantity = 0.0):
        """
        Parameters
        ----------
        wavelength: u.Quantity
            The emission wavelength in nanometers.
        brightness : u.Quantity
            The observed brightness in rayleighs.
        uncertainty : u.Quantity
            The uncertainty in the observed brightness in rayleighs.
        distance : u.Quantity, optional
            The distance of the target satellite from the plasma sheet
            centrifugal equator at the time of the observation.
        time : str or Time, optional
            The UTC time of the observation.
        systematic_uncertainty : float or u.Quantity, optional
            Additional uncertainty as either a fraction of the observed
            brightness (if provided as a float) or in absolute units (if
            provided as an Astropy `Quantity`). For example, passing 0.09 would
            mean a 9% systematic uncertainty, while passing `100*u.R` would add
            100 rayleighs of systematic uncertainty.
        """
        self._wavelength = wavelength
        self._brightness = brightness
        self._uncertainty = self._process_systematic(uncertainty,
                                                     systematic_uncertainty)
        self._distance = distance
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

    def _process_systematic(self,
                            rand_unc: u.Quantity,
                            sys_unc: float or u.Quantity) -> u.Quantity:
        """
        Propagate systematic uncertainty, if any.

        Parameters
        ----------
        rand_unc : u.Quantity
            The uncertainty in the observed brightness in rayleighs.
        sys_unc : float or u.Quantity, optional
            Additional uncertainty as either a fraction of the observed
            brightness (if provided as a float) or in absolute units (if
            provided as an Astropy `Quantity`).

        Returns
        -------
        u.Quantity
            The propagated uncertainty combining systematic and random.
        """
        if isinstance(sys_unc, u.Quantity):
            return np.sqrt(rand_unc**2 + sys_unc**2)
        elif isinstance(sys_unc, float):
            return np.sqrt(rand_unc**2 + (sys_unc*self._brightness)**2)
        else:
            return rand_unc

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
    def distance(self) -> u.Quantity:
        return self._distance.to(u.R_jup)

    @property
    def time(self) -> str:
        return self._time
