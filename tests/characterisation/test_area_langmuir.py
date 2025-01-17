"""
This test module has tests relating to Langmuir area calculations.

All functions in /calculations/area_langmuir.py are tested here.
The purposes are:

    - testing the user-facing API function (area_langmuir)
    - testing individual low level functions against known results.

Functions are tested against pre-calculated values on real isotherms.
All pre-calculated data for characterisation can be found in the
/.conftest file together with the other isotherm parameters.
"""

import pytest
from matplotlib.testing.decorators import cleanup
from numpy import isclose
from numpy import linspace

import pygaps.characterisation.area_lang as al
import pygaps.parsing.json as pgpj
import pygaps.utilities.exceptions as pgEx

from .conftest import DATA
from .conftest import DATA_N77_PATH


@pytest.mark.characterisation
class TestAreaLangmuir():
    """Tests Langmuir surface area calculations."""
    def test_basic_functions(self):
        """Test basic functionality."""
        P = linspace(0, 1)
        L = linspace(0.3, 10)

        # arrays should be equal
        with pytest.raises(pgEx.ParameterError):
            al.area_langmuir_raw(P[1:], L, 1)

        # should not take less than 3 points
        with pytest.raises(pgEx.CalculationError):
            al.area_langmuir_raw(P[:2], L[:2], 1, p_limits=[-1, 10])

        # 3 will work
        al.area_langmuir_raw(P[:3], L[:3], 1, p_limits=[-1, 10])

        # test using autolimits
        al.area_langmuir_raw(P, L, 1)

    @pytest.mark.parametrize('sample', [sample for sample in DATA])
    def test_area_langmuir(self, sample):
        """Test calculation with several model isotherms."""
        sample = DATA[sample]
        # exclude datasets where it is not applicable
        if sample.get('langmuir_area', None):

            filepath = DATA_N77_PATH / sample['file']
            isotherm = pgpj.isotherm_from_json(filepath)

            area = al.area_langmuir(isotherm).get("area")

            err_relative = 0.1  # 10 percent
            err_absolute = 0.1  # 0.1 m2

            assert isclose(area, sample['langmuir_area'], err_relative, err_absolute)

    def test_area_langmuir_choice(self):
        """Test choice of points."""

        sample = DATA['MCM-41']
        filepath = DATA_N77_PATH / sample['file']
        isotherm = pgpj.isotherm_from_json(filepath)

        area = al.area_langmuir(isotherm, p_limits=[0.05, 0.30]).get("area")

        err_relative = 0.1  # 10 percent
        err_absolute = 0.1  # 0.1 m2

        assert isclose(area, sample['langmuir_area_s'], err_relative, err_absolute)

    def test_area_langmuir_branch(self):
        """Test branch to use."""

        sample = DATA['Takeda 5A']
        filepath = DATA_N77_PATH / sample['file']
        isotherm = pgpj.isotherm_from_json(filepath)

        area = al.area_langmuir(isotherm, branch="des").get("area")

        err_relative = 0.1  # 10 percent
        err_absolute = 0.1  # 0.1 m2
        assert isclose(area, sample['langmuir_area'], err_relative, err_absolute)

    @cleanup
    def test_area_langmuir_output(self):
        """Test verbosity."""
        sample = DATA['MCM-41']
        filepath = DATA_N77_PATH / sample['file']
        isotherm = pgpj.isotherm_from_json(filepath)
        al.area_langmuir(isotherm, verbose=True)
