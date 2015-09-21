import numpy as np

from cosmolabe.units import units
import effective_shape_docstrings as docstrings


def external_docstring():
    """A decorator that allows the majority of the (ugly for humans) doc
    strings to be stored in another file.  These docs are meant to be
    read after being rendered from reST and LaTeX into html."""
    def add_docs(func):
        func.__doc__ = func.__doc__.format(getattr(docstrings, func.func_name))
        return func
    return add_docs


class EH98_effective_shape(object):
    """A class to calculate the effective shape transfer function as described
    in the article `Baryonic Features in the Matter Transfer Function` by
    Daniel Eisenstein and Wayne Hu (EH98_). Note that this **does** include the
    supression at small scales due to baryons but **does not** include
    oscillations.

    .. note::

        All equation numbers refer to those in EH98_.

    .. _EH98: http://adsabs.harvard.edu/abs/1998ApJ...496..605E
    """

    def __init__(self, cosmo_params):

        self.cosmo_params = cosmo_params
        self._OmegaM = cosmo_params['Omega_m']
        self._OmegaB = cosmo_params['Omega_b']
        self._OmegaC = cosmo_params['Omega_c']
        self._h = cosmo_params['h']
        Tcmb_mag = cosmo_params['T_cmb'].rescale('K').magnitude
        self.cu = units.CosmoUnits(self._h, 1.0)
        self._ThetaCMB = Tcmb_mag / 2.7 * self.cu.dimensionless

        self._Gamma0 = self._OmegaM * self._h
        self._omhh = self._OmegaM * self._h * self._h
        self._obhh = self._OmegaB * self._h * self._h
        self._f_b = self._OmegaB / self._OmegaM
        self._k_eq = self._return_k_eq()
        self._z_eq = self._return_z_eq()
        self._z_d = self._return_z_d()
        self._s = self._return_s()



    @external_docstring()
    def _return_z_eq(self):
        r"""Set the matter-radiation equality redshift (Eq. 2). {0}"""
        z_eq = 2.5e4 * self._omhh * self._ThetaCMB**(-4)
        return z_eq

    @external_docstring()
    def _return_k_eq(self):
        r"""Set horizon scale at matter-radiation equality (Eq. 3). {0}"""
        k_eq = 7.46e-2 * self._OmegaM * self._h * self._ThetaCMB**(-2)
        return k_eq * self.cu.h / self.cu.Mpc

    @external_docstring()
    def _return_z_d(self):
        r"""Set the drag epoch redshift (Eq. 4). {0}"""
        omhh = self._omhh
        b1 = 0.313 * omhh**(-0.419) * (1 + 0.607 * omhh**(0.674))
        b2 = 0.238 * omhh**(0.223)
        t3 = 1 + b1 * omhh**b2
        return 1291 * (omhh**0.251 / (1 + 0.659 * omhh**0.828)) * t3

    @external_docstring()
    def _return_R(self, z):
        r"""Return baryon to photon momentum density ratio (Eq. 5). {0}"""
        R = 31.5 * self._obhh * self._ThetaCMB**(-4.0) / (z/1.0e3)
        return R

    @external_docstring()
    def _return_s(self):
        r"""Set the sound horizon at the drag epoch (Eq. 6). {0}"""
        Req = self._return_R(self._z_eq)
        Rd = self._return_R(self._z_d)
        t1 = 2.0 / (3.0 * self._k_eq) * np.sqrt(6.0 / Req)
        t2 = np.sqrt(1.0 + Rd) + np.sqrt(Rd + Req)
        t3 = 1.0 + np.sqrt(Req)
        s = t1 * np.log(t2 / t3)
        return s.rescale('Mpc/hh')

    @external_docstring()
    def _return_q(self, k):
        r"""Return :math:`q \propto k / \keq` (Eq. 28). {0}"""
        k.units = 'hh/Mpc'
        q = k.magnitude * self._ThetaCMB**2 / self._Gamma0
        return q

    @external_docstring()
    def _return_q_eff(self, k):
        r"""Return :math:`\qeff \propto k / \keq` (Eq. 28 + 30). {0}"""
        Gamma_eff = self._return_Gamma_eff(k)
        k.units = 'hh/Mpc'
        q_eff = k.magnitude * self._ThetaCMB**2 / Gamma_eff
        return q_eff

    @external_docstring()
    def _return_Gamma_eff(self, k):
        r"""Return effective shape parameter (Eq. 30). {0}"""
        a1 = 0.328 * np.log(431.0 * self._omhh) * self._f_b
        a2 = 0.3089 * np.log(22.3 * self._omhh) * self._f_b**2
        ag = 1.0 - a1 + a2
        t1 = 1 + (0.43 * k * self._s)**4
        return self._Gamma0 * (ag + (1 - ag) / t1)

    def _preprocess_input_k(self, k_in):
        r"""Assert that ``k_in`` is a quantity (i.e. has units) and set them
        to :math:`h \, {\rm Mpc}^{-1}`."""
        if not hasattr(k_in, 'units'):
            raise ValueError('input k must have units')
        return k_in.rescale('hh/Mpc')

    @external_docstring()
    def T_fit(self, k_in, use_q_eff=True):
        r"""Return the effective shape transfer function (Eq. 29). {0}"""
        k = self._preprocess_input_k(k_in)
        if use_q_eff:
            q = self._return_q_eff(k)
        else:
            q = self._return_q(k)
        L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q * q)
