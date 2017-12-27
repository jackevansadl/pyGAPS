"""
This test module has tests relating to the pointisotherm class
"""
import numpy
import pandas
import pytest
from matplotlib.testing.decorators import cleanup

import pygaps

from ..conftest import basic


@basic
class TestPointIsotherm(object):
    """
    Tests the pointisotherm class
    """
##########################

    def test_isotherm_create(self):
        "Checks isotherm can be created from basic data"
        isotherm_param = {
            'sample_name': 'carbon',
            'sample_batch': 'X1',
            'adsorbate': 'nitrogen',
            't_exp': 77,
        }

        isotherm_data = pandas.DataFrame({
            'pressure': [1, 2, 3, 4, 5, 3, 2],
            'loading': [1, 2, 3, 4, 5, 3, 2]
        })

        isotherm = pygaps.PointIsotherm(
            isotherm_data,
            loading_key='loading',
            pressure_key='pressure',
            **isotherm_param
        )

        return isotherm

    def test_isotherm_create_from_isotherm(self, basic_isotherm):
        "Checks isotherm can be created from isotherm"

        isotherm_data = pandas.DataFrame({
            'pressure': [1, 2, 3, 4, 5, 3, 2],
            'loading': [1, 2, 3, 4, 5, 3, 2]
        })

        # regular creation
        isotherm = pygaps.PointIsotherm.from_isotherm(
            basic_isotherm,
            isotherm_data,
        )

        return isotherm

    def test_isotherm_create_from_modelisotherm(self, basic_modelisotherm, basic_pointisotherm):
        "Checks isotherm can be created from isotherm"

        # regular creation
        isotherm = pygaps.PointIsotherm.from_modelisotherm(
            basic_modelisotherm,
            pressure_points=None
        )

        # Specifying points
        isotherm = pygaps.PointIsotherm.from_modelisotherm(
            basic_modelisotherm,
            pressure_points=[1, 2, 3, 4]
        )

        # Specifying isotherm
        isotherm = pygaps.PointIsotherm.from_modelisotherm(
            basic_modelisotherm,
            pressure_points=basic_pointisotherm
        )

        return isotherm

##########################
    def test_isotherm_ret_has_branch(self, basic_pointisotherm):
        """Checks that all the functions in pointIsotherm return their specified parameter"""

        # branch
        assert basic_pointisotherm.has_branch(branch='ads')
        assert basic_pointisotherm.has_branch(branch='des')

        return

    def test_isotherm_ret_data(self, basic_pointisotherm):
        """Checks that all the functions in pointIsotherm return their specified parameter"""

        other_key = "enthalpy"

        # all data
        assert basic_pointisotherm.data().equals(pandas.DataFrame({
            basic_pointisotherm.pressure_key: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.5, 2.5],
            basic_pointisotherm.loading_key: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.5, 2.5],
            other_key: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 4.0],
        }))

        # adsorption branch
        assert basic_pointisotherm.data(branch='ads').equals(pandas.DataFrame({
            basic_pointisotherm.pressure_key: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            basic_pointisotherm.loading_key: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            other_key: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }))

        # desorption branch
        assert basic_pointisotherm.data(branch='des').equals(pandas.DataFrame({
            basic_pointisotherm.pressure_key: [4.5, 2.5],
            basic_pointisotherm.loading_key: [4.5, 2.5],
            other_key: [4.0, 4.0],
        }, index=[6, 7]))

        return

    def test_isotherm_ret_pressure(self, basic_pointisotherm, use_adsorbate):
        """Checks that all the functions in pointIsotherm return their specified parameter"""

        # Regular return
        assert set(basic_pointisotherm.pressure()) == set(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.5, 2.5])

        # Branch specified
        assert set(basic_pointisotherm.pressure(
            branch='des')) == set([4.5, 2.5])

        # Unit specified
        assert set(basic_pointisotherm.pressure(branch='ads', pressure_unit='Pa')) == set(
            [100000, 200000, 300000, 400000, 500000, 600000])

        # Mode specified
        assert basic_pointisotherm.pressure(branch='ads', pressure_mode='relative')[
            0] == pytest.approx(0.12849, 0.001)

        # Mode and unit specified
        assert basic_pointisotherm.pressure(branch='ads',
                                            pressure_unit='Pa',
                                            pressure_mode='relative')[0] == pytest.approx(0.12849, 0.001)

        # Range specified
        assert set(basic_pointisotherm.pressure(branch='ads', min_range=2.3, max_range=5.0)) == set(
            [3.0, 4.0, 5.0])

        # Indexed option specified
        assert basic_pointisotherm.pressure(indexed=True).equals(pandas.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.5, 2.5]
        ))

        return

    def test_isotherm_ret_loading(self, basic_pointisotherm, use_sample, use_adsorbate):
        """Checks that all the functions in pointIsotherm return their specified parameter"""

        # Standard return
        assert set(basic_pointisotherm.loading()) == set(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.5, 2.5])

        # Branch specified
        assert set(basic_pointisotherm.loading(branch='ads')) == set(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Loading unit specified
        assert basic_pointisotherm.loading(branch='ads', loading_unit='mol')[
            0] == pytest.approx(0.001, 1e-5)

        # Loading basis specified
        assert basic_pointisotherm.loading(branch='ads', loading_basis='volume')[
            0] == pytest.approx(0.8764, 1e-3)

        # Adsorbent unit specified
        assert basic_pointisotherm.loading(branch='ads', adsorbent_unit='kg')[
            0] == pytest.approx(1000, 1e-3)

        # Adsorbent basis specified
        assert basic_pointisotherm.loading(branch='ads', adsorbent_basis='volume')[
            0] == pytest.approx(10, 1e-3)

        # All specified
        assert numpy.isclose(basic_pointisotherm.loading(branch='ads',
                                                         loading_unit='kg',
                                                         loading_basis='mass',
                                                         adsorbent_unit='m3',
                                                         adsorbent_basis='volume')[
            0], 280.1, 0.1, 0.1)

        # Range specified
        assert set(basic_pointisotherm.loading(branch='ads', min_range=2.3, max_range=5.0)) == set(
            [3.0, 4.0, 5.0])

        # Indexed option specified
        assert basic_pointisotherm.loading(indexed=True).equals(pandas.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.5, 2.5]
        ))

        return

    def test_isotherm_ret_other_data(self, basic_pointisotherm):
        """Checks that all the functions in pointIsotherm return their specified parameter"""

        other_key = "enthalpy"

        # Standard return
        assert set(basic_pointisotherm.other_data(other_key)) == set([
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 4.0])

        # Branch specified
        assert set(basic_pointisotherm.other_data(other_key, branch='ads')
                   ) == set([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

        # Range specified
        assert set(basic_pointisotherm.other_data(other_key, min_range=3, max_range=4.5)
                   ) == set([4.0, 4.0])

        # Indexed option specified
        assert basic_pointisotherm.other_data(other_key, indexed=True).equals(pandas.Series(
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 4.0]
        ))

        return

##########################
    def test_isotherm_ret_loading_at(self, basic_pointisotherm, use_sample, use_adsorbate):
        """Checks that all the functions in pointIsotherm return their specified parameter"""

        # Standard return
        loading = basic_pointisotherm.loading_at(1)
        assert numpy.isclose(loading, 1.0, 1e-5)

        # Branch specified
        loading_branch = basic_pointisotherm.loading_at(1, branch='ads')
        assert numpy.isclose(loading_branch, 1.0, 1e-5)

        # Pressure unit specified
        loading_punit = basic_pointisotherm.loading_at(
            100000, pressure_unit='Pa')
        assert numpy.isclose(loading_punit, 1.0, 1e-5)

        # Presure mode specified
        loading_mode = basic_pointisotherm.loading_at(
            0.5, pressure_mode='relative')
        assert numpy.isclose(loading_mode, 3.89137, 1e-5)

        # Loading unit specified
        loading_lunit = basic_pointisotherm.loading_at(1, loading_unit='mol')
        assert numpy.isclose(loading_lunit, 0.001, 1e-5)

        # Loading basis specified
        loading_lbasis = basic_pointisotherm.loading_at(
            1, loading_basis='volume')
        assert numpy.isclose(loading_lbasis, 0.8764, 1e-4, 1e-4)

        # Adsorbent unit specified
        loading_uads = basic_pointisotherm.loading_at(
            1, adsorbent_unit='kg')
        assert numpy.isclose(loading_uads, 1000, 1e-5)

        # Adsorbent basis specified
        loading_bads = basic_pointisotherm.loading_at(
            1, adsorbent_basis='volume')
        assert numpy.isclose(loading_bads, 10, 1e-5)

        # Everything specified
        loading_all = basic_pointisotherm.loading_at(0.5,
                                                     pressure_unit='Pa',
                                                     pressure_mode='relative',
                                                     loading_unit='kg',
                                                     loading_basis='mass',
                                                     adsorbent_unit='m3',
                                                     adsorbent_basis='volume')
        assert numpy.isclose(loading_all, 1090.11, 1e-2, 1e-2)

        # Interp_fill specified
        loading_fill = basic_pointisotherm.loading_at(
            10, interp_fill=(0, 20))
        assert numpy.isclose(loading_fill, 20.0, 1e-5)

        # Interp_type specified
        loading_type = basic_pointisotherm.loading_at(
            1, interpolation_type='slinear')
        assert loading_type == pytest.approx(1.0, 1e-5)

        return

    def test_isotherm_ret_pressure_at(self, basic_pointisotherm, use_sample, use_adsorbate):
        """Checks that all the functions in ModelIsotherm return their specified parameter"""

        # Standard return
        pressure = basic_pointisotherm.pressure_at(1)
        assert pressure == pytest.approx(1.0, 1e-5)

        # Branch specified
        pressure_branch = basic_pointisotherm.pressure_at(4.0, branch='des')
        assert pressure_branch == pytest.approx(4.0, 1e-5)

        # Pressure unit specified
        pressure_punit = basic_pointisotherm.pressure_at(
            1.0, pressure_unit='Pa')
        assert pressure_punit == pytest.approx(100000, 0.1)

        # Pressure mode specified
        pressure_mode = basic_pointisotherm.pressure_at(
            3.89137, pressure_mode='relative')
        assert pressure_mode == pytest.approx(0.5, 1e-5)

        # Loading unit specified
        pressure_lunit = basic_pointisotherm.pressure_at(
            0.001, loading_unit='mol')
        assert pressure_lunit == pytest.approx(1, 1e-5)

        # Loading basis specified
        pressure_lbasis = basic_pointisotherm.pressure_at(
            0.02808, loading_basis='mass', loading_unit='g')
        assert pressure_lbasis == pytest.approx(1, 1e-2)

        # Adsorbent unit specified
        pressure_bunit = basic_pointisotherm.pressure_at(
            1000, adsorbent_unit='kg')
        assert pressure_bunit == pytest.approx(1.0, 1e-5)

        # Adsorbent basis specified
        pressure_bads = basic_pointisotherm.pressure_at(
            10, adsorbent_basis='volume', adsorbent_unit='cm3')
        assert pressure_bads == pytest.approx(1.0, 1e-5)

        # Everything specified
        pressure_all = basic_pointisotherm.pressure_at(1.08948,
                                                       pressure_unit='Pa',
                                                       pressure_mode='relative',
                                                       loading_unit='g',
                                                       loading_basis='mass',
                                                       adsorbent_unit='cm3',
                                                       adsorbent_basis='volume')
        assert numpy.isclose(pressure_all, 0.5, 1e-2)

        return

    def test_isotherm_spreading_pressure_at(self, basic_pointisotherm, use_adsorbate):
        """Checks that all the functions in pointIsotherm return their specified parameter"""

        # Standard return
        spressure = basic_pointisotherm.spreading_pressure_at(1)
        assert spressure == pytest.approx(1.0, 1e-5)

        # Branch specified
        bpressure = basic_pointisotherm.spreading_pressure_at(1, branch='ads')
        assert bpressure == pytest.approx(1.0, 1e-5)

        # Pressure unit specified
        spressure_punit = basic_pointisotherm.spreading_pressure_at(
            100000, pressure_unit='Pa')
        assert spressure_punit == pytest.approx(1.0, 1e-5)

        # Pressure mode specified
        spressure_mode = basic_pointisotherm.spreading_pressure_at(
            0.5, pressure_mode='relative')
        assert spressure_mode == pytest.approx(3.89137, 1e-5)

        return

##########################

    @pytest.mark.parametrize('unit, multiplier', [
                            ('bar', 1),
                            ('Pa', 1e5),
        pytest.param("bad_unit", 1,
                                marks=pytest.mark.xfail),
    ])
    def test_isotherm_convert_pressure(self, basic_pointisotherm, isotherm_data, unit, multiplier):
        """Checks that the pressure conversion function work as expected"""

        # Do the conversion
        basic_pointisotherm.convert_pressure(unit_to=unit)

        # Convert initial data
        converted = isotherm_data[basic_pointisotherm.pressure_key] * multiplier
        iso_converted = basic_pointisotherm.pressure()

        # Check if one datapoint is now as expected
        assert iso_converted[0] == pytest.approx(converted[0], 0.01)

    @pytest.mark.parametrize('mode, multiplier', [
                            ('relative', 1 / 7.7827),
                            ('absolute', 1),
        pytest.param("bad_mode", 1,
                                marks=pytest.mark.xfail),
    ])
    def test_isotherm_convert_pressure_mode(self, basic_pointisotherm, use_adsorbate,
                                            isotherm_data, mode, multiplier):
        """Checks that the pressure mode conversion function work as expected"""

        # Do the conversion
        basic_pointisotherm.convert_pressure(mode_to=mode)

        # Convert initial data
        converted = isotherm_data[basic_pointisotherm.pressure_key] * multiplier
        iso_converted = basic_pointisotherm.pressure()

        # Check if one datapoint is now as expected
        assert iso_converted[0] == pytest.approx(converted[0], 0.01)

    @pytest.mark.parametrize('unit, multiplier', [
                            ('mmol', 1),
                            ('mol', 1e-3),
                            ('cm3(STP)', 22.414),
        pytest.param("bad_unit", 1,
                                marks=pytest.mark.xfail),
    ])
    def test_isotherm_convert_loading_unit(self, basic_pointisotherm, isotherm_data, unit, multiplier):
        """Checks that the loading conversion function work as expected"""

        # Do the conversion
        basic_pointisotherm.convert_loading(unit_to=unit)

        # Convert initial data
        converted = isotherm_data[basic_pointisotherm.loading_key] * multiplier
        iso_converted = basic_pointisotherm.loading()

        # Check if one datapoint is now as expected
        assert iso_converted[0] == pytest.approx(converted[0], 0.01)

    @pytest.mark.parametrize('basis, multiplier', [
                            ('molar', 1),
                            ('mass', 0.028),
        pytest.param("bad_mode", 1,
                                marks=pytest.mark.xfail),
    ])
    def test_isotherm_convert_loading_basis(self, basic_pointisotherm, use_sample,
                                            isotherm_data, basis, multiplier):
        """Checks that the loading basis conversion function work as expected"""

        # Do the conversion
        basic_pointisotherm.convert_loading(basis_to=basis)

        # Convert initial data
        converted = isotherm_data[basic_pointisotherm.loading_key] * multiplier
        iso_converted = basic_pointisotherm.loading()

        # Check if one datapoint is now as expected
        assert iso_converted[0] == pytest.approx(converted[0], 0.01)

    @pytest.mark.parametrize('unit, multiplier', [
                            ('g', 1),
                            ('kg', 1000),
        pytest.param("bad_unit", 1,
                                marks=pytest.mark.xfail),
    ])
    def test_isotherm_convert_adsorbent_unit(self, basic_pointisotherm, isotherm_data, unit, multiplier):
        """Checks that the loading conversion function work as expected"""

        # Do the conversion
        basic_pointisotherm.convert_adsorbent(unit_to=unit)

        # Convert initial data
        converted = isotherm_data[basic_pointisotherm.loading_key] * multiplier
        iso_converted = basic_pointisotherm.loading()

        # Check if one datapoint is now as expected
        assert iso_converted[0] == pytest.approx(converted[0], 0.01)

    @pytest.mark.parametrize('basis, multiplier', [
                            ('mass', 1),
                            ('volume', 10),
        pytest.param("bad_mode", 1,
                                marks=pytest.mark.xfail),
    ])
    def test_isotherm_convert_adsorbent_basis(self, basic_pointisotherm, use_sample,
                                              isotherm_data, basis, multiplier):
        """Checks that the loading basis conversion function work as expected"""

        # Do the conversion
        basic_pointisotherm.convert_adsorbent(basis_to=basis)

        # Convert initial data
        converted = isotherm_data[basic_pointisotherm.loading_key] * multiplier
        iso_converted = basic_pointisotherm.loading()

        # Check if one datapoint is now as expected
        assert iso_converted[0] == pytest.approx(converted[0], 0.01)

    @cleanup
    def test_isotherm_print_parameters(self, basic_pointisotherm):
        "Checks isotherm can print its own info"

        basic_pointisotherm.print_info(show=False)
