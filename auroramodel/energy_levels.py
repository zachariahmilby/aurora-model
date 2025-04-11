import astropy.units as u
import numpy as np


class Transition:
    """
    Class to hold transition wavelength and probability information.
    """
    def __init__(self,
                 wavelength: u.Quantity,
                 einstein_a: u.Quantity):
        """
        Parameters
        ----------
        wavelength : u.Quantity
            Transition wavelength.
        einstein_a : u.Quantity
            Transition probability as an Einstein A coefficient (units of [Hz],
            probably).
        """
        self._wavelength = wavelength
        self._einstein_a = einstein_a

    @property
    def wavelength(self) -> u.Quantity:
        return self._wavelength

    @property
    def einstein_a(self) -> u.Quantity:
        return self._einstein_a


class EnergyLevel:
    """
    Class to hold energy level information for a given atom.
    """
    def __init__(self,
                 atom: str,
                 term: str,
                 transitions: [Transition]):
        """
        Parameters
        ----------
        atom : str
            The emitting atom, e.g., "H" or "O".
        term : str
            The term symbol for the given level, e.g., "1D" or "3p 3P".
        transitions : [Transition]
            A list of Transition objects, each of which contains a wavelength
            and Einstein A coefficient.
        """
        self._atom = atom
        self._term = term
        self._transitions = transitions
        self._wavelengths = self._get_wavelengths()
        self._relative_probabilities = self._calculate_relative_probabilities()

    def __str__(self):
        print_strs = [f'{self._atom}({self._term}) transitions:']
        for i in range(len(self._wavelengths)):
            print_strs .append(f'   {self._wavelengths[i]:.1f} '
                               f'({self._relative_probabilities[i]*100:.2f}%)')
        return '\n'.join(print_strs)

    def _calculate_relative_probabilities(self) -> np.ndarray:
        """
        Calculate the relative probability of each transition.
        """
        einstein_as = np.array(
            [t.einstein_a.si.value for t in self._transitions])
        return einstein_as / np.sum(einstein_as)

    def _get_wavelengths(self) -> [u.Quantity]:
        """
        Get the wavelengths of each transition.
        """
        return [t.wavelength.to(u.nm) for t in self._transitions]

    @property
    def wavelengths(self) -> [u.Quantity]:
        return self._wavelengths

    @property
    def probabilities(self) -> np.ndarray:
        return self._relative_probabilities


_transitions = {
    'O_130.4nm': Transition(wavelength=130.4*u.nm, einstein_a=5.643e8/u.s),
    'O_135.6nm': Transition(wavelength=135.6*u.nm, einstein_a=5.56e3/u.s),
    'O_297.2nm': Transition(wavelength=297.2*u.nm, einstein_a=7.54e-2/u.s),
    'O_557.7nm': Transition(wavelength=557.7*u.nm, einstein_a=1.26/u.s),
    'O_630.0nm': Transition(wavelength=630.0*u.nm, einstein_a=5.63e-3/u.s),
    'O_636.4nm': Transition(wavelength=636.4*u.nm, einstein_a=1.82e-3/u.s),
    'O_777.4nm': Transition(wavelength=777.4*u.nm, einstein_a=3.69e7/u.s),
    'O_844.6nm': Transition(wavelength=844.6*u.nm, einstein_a=3.22e7/u.s),
    'H_121.6nm': Transition(wavelength=121.6*u.nm, einstein_a=4.70e8/u.s),
    'H_486.1nm': Transition(wavelength=486.1*u.nm, einstein_a=8.42e-2/u.s),
    'H_656.3nm': Transition(wavelength=656.3*u.nm, einstein_a=4.41e-1/u.s),
}

energy_levels = {
    'O(3S)': EnergyLevel(atom='O', term='3S',
                         transitions=[_transitions['O_130.4nm']]),
    'O(5S)': EnergyLevel(atom='O', term='5S',
                         transitions=[_transitions['O_135.6nm']]),
    'O(1S)': EnergyLevel(atom='O', term='1S',
                         transitions=[_transitions['O_297.2nm'],
                                      _transitions['O_557.7nm']]),
    'O(1D)': EnergyLevel(atom='O', term='1D',
                         transitions=[_transitions['O_630.0nm'],
                                      _transitions['O_636.4nm']]),
    'O(3p5P)': EnergyLevel(atom='O', term='3p 5P',
                           transitions=[_transitions['O_777.4nm']]),
    'O(3p3P)': EnergyLevel(atom='O', term='3p 3P',
                           transitions=[_transitions['O_844.6nm']]),
    'H(Ly-alpha)': EnergyLevel(atom='H', term='Ly-alpha',
                               transitions=[_transitions['H_121.6nm']]),
    'H(H-beta)': EnergyLevel(atom='H', term='H-beta',
                             transitions=[_transitions['H_486.1nm']]),
    'H(H-alpha)': EnergyLevel(atom='H', term='H-alpha',
                              transitions=[_transitions['H_656.3nm']])
}
