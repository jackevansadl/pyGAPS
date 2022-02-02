"""
Parse to and from a CSV string/file format for isotherms.

The _parser_version variable documents any changes to the format,
and is used to check for any deprecations.

"""

from io import StringIO

import pandas

from pygaps import logger
from pygaps.core.baseisotherm import BaseIsotherm
from pygaps.core.modelisotherm import ModelIsotherm
from pygaps.core.pointisotherm import PointIsotherm
from pygaps.modelling import model_from_dict
from pygaps.utilities.exceptions import ParsingError
from pygaps.utilities.string_utilities import _from_bool
from pygaps.utilities.string_utilities import _from_list
from pygaps.utilities.string_utilities import _is_bool
from pygaps.utilities.string_utilities import _is_float
from pygaps.utilities.string_utilities import _is_list
from pygaps.utilities.string_utilities import _to_string

_parser_version = "3.0"


def isotherm_to_csv(isotherm, path=None, separator=','):
    """
    Convert isotherm into a CSV representation.

    If the path is specified, the isotherm is saved as a file,
    otherwise it is returned as a string.

    Parameters
    ----------
    isotherm : Isotherm
        Isotherm to be written to csv.
    path : str, None
        Path to the file to be written.
    separator : str, optional
        Separator used int the csv file. Defaults to '',''.

    Returns
    -------
    str: optional
        String representation of the CSV, if path not provided.

    """
    output = StringIO()

    iso_dict = isotherm.to_dict()
    iso_dict['file_version'] = _parser_version  # version

    # Parse material
    material = iso_dict['material']
    if isinstance(material, dict):
        iso_dict['material'] = material.pop('name')
        iso_dict.update({f"_material_{key}": val for key, val in material.items()})

    output.writelines([x + separator + _to_string(y) + '\n' for (x, y) in iso_dict.items()])

    if isinstance(isotherm, PointIsotherm):

        # We get data and replace adsorption terminology
        data = isotherm.data_raw.copy()
        data['branch'] = data['branch'].replace(0, 'ads').replace(1, 'des')

        output.write('data:[pressure,loading,branch,(otherdata)]\n')
        data.to_csv(output, sep=separator, index=False, header=True)

    elif isinstance(isotherm, ModelIsotherm):

        output.write('model:[name and parameters]\n')
        output.write(('name' + separator + isotherm.model.name + '\n'))
        output.write(('rmse' + separator + _to_string(isotherm.model.rmse) + '\n'))
        output.write(
            ('pressure range' + separator + _to_string(isotherm.model.pressure_range) + '\n')
        )
        output.write(
            ('loading range' + separator + _to_string(isotherm.model.loading_range) + '\n')
        )
        output.writelines([
            param + separator + str(isotherm.model.params[param]) + '\n'
            for param in isotherm.model.params
        ])

    if path:
        with open(path, mode='w', newline='\n', encoding='utf-8') as file:
            file.write(output.getvalue())
    else:
        return output.getvalue()


def isotherm_from_csv(str_or_path, separator=',', **isotherm_parameters):
    """
    Load an isotherm from a CSV file.

    Parameters
    ----------
    str_or_path : str
        The isotherm in a CSV string format or a path
        to where one can be read.
    separator : str, optional
        Separator used int the csv file. Defaults to `,`.
    isotherm_parameters :
        Any other options to be overridden in the isotherm creation.

    Returns
    -------
    Isotherm
        The isotherm contained in the csv string or file.

    """
    try:
        with open(str_or_path, encoding='utf-8') as f:
            raw_csv = StringIO(f.read())
    except OSError:
        try:
            raw_csv = StringIO(str_or_path)
        except Exception as e:
            raise ParsingError(
                "Could not parse CSV isotherm. "
                "The `str_or_path` is invalid or does not exist. "
            ) from e

    line = raw_csv.readline().rstrip()
    raw_dict = {}

    try:
        while not (line.startswith('data') or line.startswith('model') or line == ""):
            values = line.strip().split(sep=separator)

            if len(values) > 2:
                raise ParsingError(f"The isotherm metadata {values} contains more than two values.")
            key, val = values
            if not val:
                val = None
            elif _is_bool(val):
                val = _from_bool(val)
            elif val.isnumeric():
                val = int(val)
            elif _is_float(val):
                val = float(val)
            elif _is_list(val):
                val = _from_list(val)

            raw_dict[key] = val
            line = raw_csv.readline().rstrip()
    except Exception as e:
        raise ParsingError(
            "Could not parse CSV isotherm. "
            "The format may be wrong, check for errors."
        ) from e

    # version check
    version = raw_dict.pop("file_version", None)
    if not version or float(version) < float(_parser_version):
        logger.warning(
            f"The file version is {version} while the parser uses version {_parser_version}. "
            "Strange things might happen, so double check your data."
        )

    # check if material needs parsing
    material = {}
    for key, val in raw_dict.items():
        if key.startswith("_material_"):
            material[key.replace("_material_", "")] = val
    if material:
        for key in material.keys():
            raw_dict.pop("_material_" + key)
        material['name'] = raw_dict['material']
        raw_dict['material'] = material

    # Update dictionary with any user parameters
    raw_dict.update(isotherm_parameters)

    # Now read specific type of isotherm (Point, Model, Base)
    if line.startswith('data'):
        data = pandas.read_csv(raw_csv, sep=separator)

        # process isotherm branches if they exist
        if 'branch' in data.columns:
            data['branch'] = data['branch'].apply(lambda x: 0 if x == 'ads' else 1)
        else:
            raw_dict['branch'] = 'guess'

        isotherm = PointIsotherm(
            isotherm_data=data,
            pressure_key=data.columns[0],
            loading_key=data.columns[1],
            **raw_dict
        )

    elif line.startswith('model'):

        model = {}
        line = raw_csv.readline().rstrip()
        model['name'] = line.split(sep=separator)[1]
        line = raw_csv.readline().rstrip()
        model['rmse'] = line.split(sep=separator)[1]
        line = raw_csv.readline().rstrip()
        model['pressure_range'] = _from_list(line.split(sep=separator)[1])
        line = raw_csv.readline().rstrip()
        model['loading_range'] = _from_list(line.split(sep=separator)[1])
        line = raw_csv.readline().rstrip()
        model['parameters'] = {}
        while line != "":
            values = line.split(sep=separator)
            model['parameters'][values[0]] = float(values[1])
            line = raw_csv.readline().rstrip()

        isotherm = ModelIsotherm(
            model=model_from_dict(model),
            **raw_dict,
        )

    else:
        isotherm = BaseIsotherm(**raw_dict)

    return isotherm
