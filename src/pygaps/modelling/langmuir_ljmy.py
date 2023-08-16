"""Double Site Langmuir isotherm model."""

import numpy
from scipy.special import erf
from scipy import integrate
from scipy import optimize
from pygaps.utilities.exceptions import CalculationError

from pygaps.modelling.base_model import IsothermBaseModel


class Langmuir_LJMY(IsothermBaseModel):
    r"""
    Langmuir adsorption isotherm with step function.

    .. math::

        n

    References
    ----------
    .. [#] 

    """

    # Model parameters
    name = 'Langmuir_LJMY'
    formula = r"n(p) = n_{m_1}\frac{K_1 p}{1+K_1 p} +  n_{m_2}\frac{K_2 p}{1+K_2 p}"
    calculates = 'loading'
    param_names = ("n_m", "K", "n_m_step", "p_tr", "p_sigma")
    param_default_bounds = (
        (0., numpy.inf),
        (0., numpy.inf),
        (0., numpy.inf),
        (0., numpy.inf),
        (0., numpy.inf),
    )

    def loading(self, pressure):
        """
        Calculate loading at specified pressure.

        Parameters
        ----------
        pressure : float
            The pressure at which to calculate the loading.

        Returns
        -------
        float
            Loading at specified pressure.
        """
        Nm = self.params['n_m']
        K = self.params['K']
        Nm_step = self.params['n_m_step']
        P_tr = self.params['p_tr']
        P_sigma = self.params['p_sigma']


        return (Nm * K * pressure / (1.0 + K * pressure)) + Nm_step/2*(1+erf((pressure-P_tr)/(2**(1. / 3)*P_sigma)))

    def pressure(self, loading):
        """
        Calculate pressure at specified loading.

        For the Jensen-Seaton model, the pressure will
        be computed numerically as no analytical inversion is possible.

        Parameters
        ----------
        loading : float
            The loading at which to calculate the pressure.

        Returns
        -------
        float
            Pressure at specified loading.
        """
        def fun(x):
            return self.loading(x) - loading

        opt_res = optimize.root(fun, numpy.zeros_like(loading), method='hybr',  options={'maxfev' : int(1e6)})

        if not opt_res.success:
            raise CalculationError(f"Root finding for value {loading} failed.")

        return opt_res.x

    def spreading_pressure(self, pressure):
        r"""
        Calculate spreading pressure at specified gas pressure.

        Function that calculates spreading pressure by solving the
        following integral at each point i.

        .. math::

            \pi = \int_{0}^{p_i} \frac{n_i(p_i)}{p_i} dp_i

        The integral for the Jensen-Seaton model cannot be solved analytically
        and must be calculated numerically.

        Parameters
        ----------
        pressure : float
            The pressure at which to calculate the spreading pressure.

        Returns
        -------
        float
            Spreading pressure at specified pressure.
        """
        return integrate.quad(lambda x: self.loading(x) / x, 0, pressure)[0]

    def initial_guess(self, pressure, loading):
        """
        Return initial guess for fitting.

        Parameters
        ----------
        pressure : ndarray
            Pressure data.
        loading : ndarray
            Loading data.

        Returns
        -------
        dict
            Dictionary of initial guesses for the parameters.
        """
        saturation_loading, langmuir_k = super().initial_guess(pressure, loading)
        guess = {"K": saturation_loading * langmuir_k, "n_m" : saturation_loading, "n_m_step" : 1, "p_tr" : 1, "p_sigma" : 1}
        guess = self.initial_guess_bounds(guess)
        return guess


