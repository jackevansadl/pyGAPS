"""Contains the Isotherm base class."""

import logging
import warnings

logger = logging.getLogger('pygaps')

from pygaps.core.adsorbate import Adsorbate
from pygaps.core.material import Material

from ..utilities.converter_mode import _LOADING_MODE
from ..utilities.converter_mode import _MATERIAL_MODE
from ..utilities.converter_mode import _PRESSURE_MODE
from ..utilities.converter_mode import c_temperature
from ..utilities.converter_unit import _PRESSURE_UNITS
from ..utilities.converter_unit import _TEMPERATURE_UNITS
from ..utilities.exceptions import ParameterError
from ..utilities.hashgen import isotherm_to_hash
from ..utilities.python_utilities import simplewarning

SHORTHANDS = {
    'm': "material",
    't': "temperature",
    'a': "adsorbate",
}


class BaseIsotherm():
    """
    Class which contains the general data for an isotherm, real or model.

    The isotherm class is the parent class that both PointIsotherm and
    ModelIsotherm inherit. It is designed to contain the information about
    an isotherm (such as material, adsorbate, data units etc.) but without
    any of the data itself.

    Think of this class as a extended python dictionary.

    Parameters
    ----------
    material : str
        Name of the material on which the isotherm is measured.
    adsorbate : str
        Isotherm adsorbate.
    temperature : float
        Isotherm temperature.

    Other Parameters
    ----------------
    material_basis : str, optional
        Whether the adsorption is read in terms of either 'per volume'
        'per molar amount' or 'per mass' of material.
    material_unit : str, optional
        Unit in which the material basis is expressed.
    loading_basis : str, optional
        Whether the adsorbed material is read in terms of either 'volume'
        'molar' or 'mass'.
    loading_unit : str, optional
        Unit in which the loading basis is expressed.
    pressure_mode : str, optional
        The pressure mode, either 'absolute' pressures or 'relative' in
        the form of p/p0.
    pressure_unit : str, optional
        Unit of pressure.


    Notes
    -----
    The class is also used to prevent duplication of code within the child
    classes, by calling the common inherited function before any other specific
    implementation additions.

    The minimum arguments required to instantiate the class are
    ``material``, ``temperature', ``adsorbate``.
    """

    # strictly required attributes
    _required_params = [
        'material',
        'temperature',
        'adsorbate',
    ]
    # unit-related attributes and their defaults
    _unit_params = {
        'pressure_mode': 'absolute',
        'pressure_unit': 'bar',
        'material_basis': 'mass',
        'material_unit': 'g',
        'loading_basis': 'molar',
        'loading_unit': 'mmol',
        'temperature_unit': 'K',
    }
    # other special reserved parameters
    # subclasses extend this
    _reserved_params = [
        "_material",
        "_adsorbate",
        "_temperature",
        "m",
        "t",
        "a",
    ]

    ##########################################################
    #   Instantiation and classmethods

    def __init__(self, **properties):
        """
        Instantiate is done by passing a dictionary with the parameters,
        as well as the info about units, modes and data columns.

        """
        # commonly used shorthands
        for shorthand in SHORTHANDS:
            data = properties.pop(shorthand, None)
            if data:
                properties[SHORTHANDS[shorthand]] = data

        # Must-have properties of the isotherm
        #
        # Basic checks
        if any(k not in properties for k in self._required_params):
            raise ParameterError(
                f"Isotherm MUST have the following properties:{self._required_params}"
            )

        # Isotherm temperature.
        self.temperature = float(properties.pop('temperature'))

        # Isotherm material
        self.material = properties.pop('material')

        # Isotherm adsorbate
        self.adsorbate = properties.pop('adsorbate')

        # Isotherm units
        #
        with simplewarning():
            for k in self._unit_params:
                if k not in properties:
                    warnings.warn(
                        f"WARNING: '{k}' was not specified , assumed as '{self._unit_params[k]}'"
                    )
                    properties[k] = self._unit_params[k]

        # TODO deprecation
        if self._unit_params['loading_basis'] == 'volume':
            self._unit_params['loading_basis'] = 'volume_gas'
            logger.warning(
                "Loading basis as 'volume' is unclear and deprecated. "
                "Assumed as 'volume_gas'."
            )

        self.pressure_mode = properties.pop('pressure_mode')
        self.pressure_unit = properties.pop('pressure_unit')
        if self.pressure_mode.startswith('relative'):
            self.pressure_unit = None
        self.material_basis = properties.pop('material_basis')
        self.material_unit = properties.pop('material_unit')
        self.loading_basis = properties.pop('loading_basis')
        self.loading_unit = properties.pop('loading_unit')
        self.temperature_unit = properties.pop('temperature_unit')

        # Check basis / mode
        if self.pressure_mode not in _PRESSURE_MODE:
            raise ParameterError(
                f"Mode selected for pressure ({self.pressure_mode}) is not an option. "
                f"See viable values: {_PRESSURE_MODE.keys()}"
            )

        if self.loading_basis not in _LOADING_MODE:
            raise ParameterError(
                f"Basis selected for loading ({self.loading_basis}) is not an option. "
                f"See viable values: {_LOADING_MODE.keys()}"
            )

        if self.material_basis not in _MATERIAL_MODE:
            raise ParameterError(
                f"Basis selected for material ({self.material_basis}) is not an option. "
                f"See viable values: {_MATERIAL_MODE.keys()}"
            )

        # Check units
        if self.pressure_mode == 'absolute' and self.pressure_unit not in _PRESSURE_UNITS:
            raise ParameterError(
                f"Unit selected for pressure ({self.pressure_unit}) is not an option. "
                f"See viable values: {_PRESSURE_UNITS.keys()}"
            )

        if self.loading_unit not in _LOADING_MODE[self.loading_basis]:
            raise ParameterError(
                f"Unit selected for loading ({self.loading_unit}) is not an option. "
                f"See viable values: {_LOADING_MODE[self.loading_basis].keys()}"
            )

        if self.material_unit not in _MATERIAL_MODE[self.material_basis]:
            raise ParameterError(
                f"Unit selected for material ({self.material_unit}) is not an option. "
                f"See viable values: {_MATERIAL_MODE[self.loading_basis].keys()}"
            )

        if self.temperature_unit not in _TEMPERATURE_UNITS:
            raise ParameterError(
                f"Unit selected for temperature ({self.temperature_unit}) is not an option. "
                f"See viable values: {_TEMPERATURE_UNITS.keys()}"
            )

        # Other named properties of the isotherm

        # Save the rest of the properties as metadata
        self.properties = properties

    ##########################################################
    #   Overloaded and own functions

    @property
    def iso_id(self):
        """Return an unique identifier of the isotherm."""
        return isotherm_to_hash(self)

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        try:
            self._material = Material.find(value)
        except ParameterError:
            self._material = Material(value)

    @property
    def adsorbate(self):
        return self._adsorbate

    @adsorbate.setter
    def adsorbate(self, value):
        try:
            self._adsorbate = Adsorbate.find(value)
        except ParameterError:
            self._adsorbate = Adsorbate(value)
            warnings.warn((
                "Specified adsorbate is not in internal list "
                "(or name cannot be resolved to an existing one). "
                "CoolProp backend disabled for this gas/vapour."
            ))

    @property
    def temperature(self):
        if self.temperature_unit == "K":
            return self._temperature
        return c_temperature(self._temperature, self.temperature_unit, "K")

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    def __eq__(self, other_isotherm):
        """
        Overload the equality operator of the isotherm.

        Since id's should be unique and representative of the
        data inside the isotherm, all we need to ensure equality
        is to compare the two hashes of the isotherms.
        """
        return self.iso_id == other_isotherm.iso_id

    def __repr__(self):
        """Print key isotherm parameters."""
        return f"<{type(self).__name__} {self.iso_id}>: '{self.adsorbate}' on '{self.material}' at {self.temperature} K"

    def __str__(self):
        """Print a short summary of all the isotherm parameters."""
        string = ""

        # Required
        string += f"Material: { str(self.material) }\n"
        string += f"Adsorbate: { str(self.adsorbate) }\n"
        string += f"Temperature: { str(self.temperature) }K\n"

        # Units/basis
        string += "Units: \n"
        string += f"\tUptake in: {self.loading_unit}/{self.material_unit}\n"
        if self.pressure_mode.startswith('relative'):
            string += "\tRelative pressure\n"
        else:
            string += f"\tPressure in: {self.pressure_unit}\n"

        string += "Other properties: \n"
        for prop in vars(self):
            if prop not in self._required_params + \
                    list(self._unit_params) + self._reserved_params:
                string += (f"\t{prop}: {str(getattr(self, prop))}\n")

        return string

    def to_dict(self):
        """
        Returns a dictionary of the isotherm class
        Is the same dictionary that was used to create it.

        Returns
        -------
        dict
            Dictionary of all parameters.
        """
        parameter_dict = vars(self).copy()

        # These line are here to ensure that material/adsorbate are copied as a string
        parameter_dict['adsorbate'] = str(parameter_dict.pop('_adsorbate'))
        parameter_dict['material'] = str(parameter_dict.pop('_material'))
        parameter_dict['temperature'] = parameter_dict.pop('_temperature')

        # Remove reserved parameters
        for param in self._reserved_params:
            parameter_dict.pop(param, None)

        # Add metadata
        parameter_dict.update(parameter_dict.pop('properties'))

        return parameter_dict

    def to_json(self, path=None, **kwargs):
        """
        Convert the isotherm to a JSON representation.

        Parameters
        ----------
        path
            File path or object. If not specified, the result is returned as a string.
        kwargs
            Custom arguments to be passed to "json.dump", like `indent`.

        Returns
        -------
        None or str
            If path is None, returns the resulting json format as a string.
            Otherwise returns None.
        """
        from ..parsing.json import isotherm_to_json
        return isotherm_to_json(self, path, **kwargs)

    def to_csv(self, path=None, separator=',', **kwargs):
        """
        Convert the isotherm to a CSV representation.

        Parameters
        ----------
        path
            File path or object. If not specified, the result is returned as a string.
        separator : str, optional
            Separator used int the csv file. Defaults to '',''.

        Returns
        -------
        None or str
            If path is None, returns the resulting json format as a string.
            Otherwise returns None.
        """
        from ..parsing.csv import isotherm_to_csv
        return isotherm_to_csv(self, path, separator, **kwargs)

    def to_xl(self, path, **kwargs):
        """
        Save the isotherm as an Excel file.

        Parameters
        ----------
        path
            Path where to save Excel file.

        """
        from ..parsing.excel import isotherm_to_xl
        return isotherm_to_xl(self, path, **kwargs)

    def to_aif(self, path=None, **kwargs):
        """
        Convert the isotherm to a AIF representation.

        Parameters
        ----------
        path
            File path or object. If not specified, the result is returned as a string.

        Returns
        -------
        None or str
            If path is None, returns the resulting json format as a string.
            Otherwise returns None.
        """
        from ..parsing.aif import isotherm_to_aif
        return isotherm_to_aif(self, path, **kwargs)

    def to_db(
        self,
        db_path=None,
        verbose=True,
        autoinsert_material=True,
        autoinsert_adsorbate=True,
        **kwargs
    ):
        """
        Upload the isotherm to an sqlite database.

        Parameters
        ----------
        db_path : str, None
            Path to the database. If none is specified, internal database is used.
        autoinsert_material: bool, True
            Whether to automatically insert an isotherm material if it is not found
            in the database.
        autoinsert_adsorbate: bool, True
            Whether to automatically insert an isotherm adsorbate if it is not found
            in the database.
        verbose : bool
            Extra information printed to console.

        """
        from ..parsing.sqlite import isotherm_to_db
        return isotherm_to_db(
            self,
            db_path=db_path,
            autoinsert_material=autoinsert_material,
            autoinsert_adsorbate=autoinsert_adsorbate,
            verbose=verbose,
            **kwargs
        )

    def convert_temperature(
        self,
        unit_to: str,
        verbose: bool = False,
    ):
        """
        Convert isotherm temperature from one unit to another.

        Parameters
        ----------
        unit_to : str
            The unit into which the internal temperature should be converted to.
        verbose : bool
            Print out steps taken.

        """
        self._temperature = c_temperature(self._temperature, self.temperature_unit, unit_to)
        self.temperature_unit = unit_to

        if verbose:
            logger.info(f"Changed temperature unit to '{unit_to}'.")

    # Figure out the adsorption and desorption branches
    @staticmethod
    def _splitdata(data, pressure_key):
        """
        Split isotherm data into an adsorption and desorption part and
        return a column which marks the transition between the two.
        """
        from ..utilities.math_utilities import split_ads_data
        return split_ads_data(data, pressure_key)
