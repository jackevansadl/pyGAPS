"""
This configuration file contains data required for testing
scientific pygaps functions on real or model data.
In this file there are:

    - references to sample isotherm files
    - pre-calculated values for each isotherm on various
      characterization tests.

'bet_area':             BET area value
's_bet_area':           BET area value in a selected range
'langmuir_area':        Langmuir area value
's_langmuir_area':      Langmuir area value in a selected range
't_area':               t-plot calculated area
't_pore_volume':        t-plot calculated volume
's_t_area':             t-plot calculated area in a selected range
'as_ref':               reference alpha-s isotherm
'as_area':              alpha-s calculated area
'as_pore_volume':       alpha-s calculated volume
's_as_area':            alpha-s calculated area in a selected range
'Khi_slope':            initial henry constant (slope method)
'Khi_virial':           initial henry constant (virial method)
'dr_volume':            Dubinin-Radushkevitch calculated micropore volume
'dr_potential':         Dubinin-Radushkevitch calculated surface potential
'da_volume':            Dubinin-Astakov calculated micropore volume
'da_potential':         Dubinin-Astakov calculated surface potential
'psd_meso_pore_size':   Primary pore size peak, mesopore range
'psd_micro_pore_size':  Primary pore size peak, micropore range
'psd_dft_pore_size':    Primary pore size peak, DFT range
"""

import os

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'docs', 'examples', 'data')

DATA_N77_PATH = os.path.join(DATA_PATH, 'characterisation')
DATA_IAST_PATH = os.path.join(DATA_PATH, 'iast')
DATA_ISOSTERIC_PATH = os.path.join(DATA_PATH, 'isosteric')
DATA_CALO_PATH = os.path.join(DATA_PATH, 'calorimetry')

DATA = {
    'MCM-41': {
        'file': 'MCM-41 N2 77.355.json',
        'bet_area': 400.0,
        's_bet_area': 350.0,
        'langmuir_area': 1450.0,
        's_langmuir_area': 500.0,
        't_area': 340.0,
        't_pore_volume': 0.28,
        's_t_area': 55.0,
        'as_ref': 'SiO2',
        'as_area': 270,
        'as_volume': 0.3,
        's_as_area': 360,
        'Khi_slope': 57000,
        'Khi_virial': 195000,
        'dr_volume': None,
        'dr_potential': None,
        'da_volume': None,
        'da_potential': None,
        'psd_meso_pore_size': 3.2,
        'psd_dft_pore_size': 3.2,
    },
    'NaY': {
        'file': 'NaY N2 77.355.json',
        'bet_area': 700.0,
        'langmuir_area': 1100.0,
        't_area': 200.0,
        't_pore_volume': 0.26,
        'Khi_slope': 1770000,
        'Khi_virial': 1260000,
        'dr_volume': None,
        'dr_potential': None,
        'da_volume': None,
        'da_potential': None,
    },
    'SiO2': {
        'file': 'SiO2 N2 77.355.json',
        'bet_area': 200.0,
        'langmuir_area': 850.0,
        't_area': 250.0,
        't_pore_volume': 0.0,
        'Khi_slope': 780,
        'Khi_virial': 249,
        'dr_volume': None,
        'dr_potential': None,
        'da_volume': None,
        'da_potential': None,
    },
    'Takeda 5A': {
        'file': 'Takeda 5A N2 77.355.json',
        'bet_area': 1075.0,
        'langmuir_area': 1600.0,
        't_area': 110.0,
        't_pore_volume': 0.43,
        'Khi_slope': 1600000,
        'Khi_virial': 4300000,
        'dr_volume': 0.484,
        'dr_potential': 5.84,
        'da_volume': 0.346,
        'da_potential': 7.071,
        'psd_micro_pore_size': 0.6,
        'psd_dft_pore_size': 0.5,
    },
    'UiO-66(Zr)': {
        'file': 'UiO-66(Zr) N2 77.355.json',
        'bet_area': 1250.0,
        'langmuir_area': 1350.0,
        't_pore_volume': 0.48,
        't_area': 17.0,
        'Khi_slope': 700000,
        'Khi_virial': 1350000,
        'dr_volume': None,
        'dr_potential': None,
        'da_volume': None,
        'da_potential': None,
        'psd_micro_pore_size': 0.7,
        'psd_dft_pore_size': 0.6,
    },

}

DATA_IAST = {
    'CH4': {
        'file': 'MOF-5(Zn) - IAST - CH4.json',
    },
    'C2H6': {
        'file': 'MOF-5(Zn) - IAST - C2H6.json',
    },
}

DATA_ISOSTERIC = {
    't1': {
        'file': 'BAX 1500 - Isosteric Heat - 298.json',
    },
    't2': {
        'file': 'BAX 1500 - Isosteric Heat - 323.json',
    },
    't3': {
        'file': 'BAX 1500 - Isosteric Heat - 348.json',
    },
}

DATA_CALO = {
    't1': {
        'file': 'HKUST-1(Cu) KRICT.json',
        'ienth': 27,
    },
    't2': {
        'file': 'Takeda 5A Test CO2.json',
        'ienth': 35,
    },
}
