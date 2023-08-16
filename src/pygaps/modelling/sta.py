"""Double Site Langmuir isotherm model."""

import numpy
from scipy import integrate
from scipy import optimize
from pygaps.utilities.exceptions import CalculationError

from pygaps.modelling.base_model import IsothermBaseModel


class STA(IsothermBaseModel):
    r"""
    Langmuir adsorption isotherm with step function.

    .. math::

        n

    References
    ----------
    .. [#] 

    """

    # Model parameters
    name = 'STA'
    formula = r"n(p) = {1-\sigma(p)}L_np(p) + \sigma(p) L_lp(p)"
    calculates = 'loading'
    param_names = ("n_np", "k_np", "n_lp", "k_lp", "s", "p_tr")
    param_default_bounds = (
        (0., numpy.inf),
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
        n_np = self.params['n_np']
        k_np = self.params['k_np']
        n_lp = self.params['n_lp']
        k_lp = self.params['k_lp']
        s = self.params['s']
        p_tr = self.params['p_tr']

        l_np = n_np * ((k_np*pressure)/(1+k_np*pressure))
        l_lp = n_lp * ((k_lp*pressure)/(1+k_lp*pressure))

        y1 = ((1+k_np*p_tr)/(1+k_np*pressure))**n_np
        y2 = ((1+k_lp*pressure)/(1+k_lp*p_tr))**n_lp
        y = y1*y2

        sigma = (y**s)/(1+y**s)

        return ((1-sigma)*l_np)+(sigma*l_lp)

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
        guess = {"k_lp": saturation_loading * langmuir_k, "n_lp" : saturation_loading, "k_np": saturation_loading * langmuir_k*10, "n_np" : saturation_loading*0.5, "s" : 10, "p_tr" : 1}
        guess = self.initial_guess_bounds(guess)
        return guess


