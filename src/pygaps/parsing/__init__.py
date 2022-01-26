# pylint: disable=W0614,W0611,W0622
# flake8: noqa
# isort:skip_file
from ..utilities.exceptions import ParsingError

from .csv import isotherm_from_csv
from .csv import isotherm_to_csv
from .aif import isotherm_from_aif
from .aif import isotherm_to_aif
from .excel import isotherm_from_xl
from .excel import isotherm_to_xl
from .isodb import isotherm_from_isodb
from .json import isotherm_from_json
from .json import isotherm_to_json
from .sqlite import isotherms_from_db
from .sqlite import isotherm_delete_db
from .sqlite import isotherm_to_db

_COMMERCIAL_FORMATS = {
    'bel': ('csv', 'xl', 'dat'),
    'mic': ('xl'),
    '3p': ('xl'),
}


def isotherm_from_commercial(path, manufacturer, fmt, **options):
    """
    Parse aa file generated by commercial apparatus.

    Parameters
    ----------
    path: str
        the location of the file.
    manufacturer : {'mic', 'bel', '3p'}
        Manufacturer of the apparatus.
    manufacturer : {'mic', 'bel', '3p'}
        The format of the import for the isotherm.

    Returns
    -------
    PointIsotherm
    """

    if manufacturer not in _COMMERCIAL_FORMATS.keys():
        raise ParsingError(f"Currently available manufacturers are {_COMMERCIAL_FORMATS.keys()}")

    if fmt not in _COMMERCIAL_FORMATS[manufacturer]:
        raise ParsingError(f"Currently available formats are {_COMMERCIAL_FORMATS[manufacturer]}")

    if manufacturer == 'mic' and fmt == 'xl':
        from .mic_excel import parse
    elif manufacturer == 'bel' and fmt == 'xl':
        from .bel_excel import parse
    elif manufacturer == 'bel' and fmt == 'csv':
        from .bel_csv import parse
    elif manufacturer == 'bel' and fmt == 'dat':
        from .bel_dat import parse
    elif manufacturer == '3p' and fmt == 'xl':
        from .trp_excel import parse
    elif manufacturer == 'qnt' and fmt == 'txt':
        from .qnt_txt import parse
    else:
        raise ParsingError("Something went wrong.")

    import pandas
    from pygaps.core.pointisotherm import PointIsotherm

    meta, data = parse(path, **options)
    data = pandas.DataFrame(data)

    meta['loading_key'] = 'loading'
    meta['pressure_key'] = 'pressure'

    # TODO pyGAPS does not yet handle saturation pressure recorded at each point
    # Therefore, we use the relative pressure column as our true pressure,
    # ignoring the absolute pressure column
    if 'pressure_relative' in data.columns:
        data['pressure'] = data['pressure_relative']
        data = data.drop('pressure_relative', axis=1)
        meta['pressure_mode'] = 'relative'
        meta['pressure_unit'] = None

    return PointIsotherm(isotherm_data=data, **meta)
