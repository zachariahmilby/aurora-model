from pathlib import Path

import numpy as np

_package_directory = Path(__file__).resolve().parent

rcparams = Path(_package_directory, 'anc', 'rcparams.mplstyle')

color_dict = {'red': '#D62728', 'orange': '#FF7F0E', 'yellow': '#FDB813',
              'green': '#2CA02C', 'blue': '#0079C1', 'violet': '#9467BD',
              'cyan': '#17BECF', 'magenta': '#D64ECF', 'brown': '#8C564B',
              'darkgrey': '#3F3F3F', 'grey': '#7F7F7F', 'lightgrey': '#BFBFBF'}

parent_species = np.array(['O', 'O2', 'H2O', 'CO2'])

wavelengths = np.array([121.6, 130.4, 135.6, 297.2, 486.1, 557.7,
                        630.0, 636.4, 656.3, 777.4, 844.6])

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


def get_available_transitions():
    """
    Print available atomic/molecular parent species and wavelengths in the
    aurora model.
    """
    print('Available parent species and auroral emissions in model:')
    path = Path(_package_directory, 'cross_sections')
    for species in parent_species:
        files = sorted(path.glob(f'{species}_*.dat'))
        xs = [s.name.split('_')[1].replace('nm.dat', ' nm') for s in files]
        ems = []
        for sec in xs:
            for e in emissions:
                if sec in e:
                    ems.append(e)
        print(f"   Parent species: {species.replace('2', '₂')}")
        [print(f'      {emission}') for emission in ems]


def _log(log, string, silent: bool = False):
    log.append(string)
    if not silent:
        print(string)


def _write_log(path: Path, log: list):
    with open(Path(path), 'w') as file:
        file.write('\n'.join(log))
