import numpy as np
from cosmolabe.units import units


class EH98(object):
    """A class to calculate the transfer function fits described in EH98_

    .. _EH98: http://adsabs.harvard.edu/abs/1998ApJ...496..605E
    """

    def __init__(self, cosmo_params):
        self.cosmo_params = cosmo_params
        self._attach_local_cosmo_parameter_names()

        self._z_eq = self._return_z_eq()
        self._k_eq = self._return_k_eq()
        self._z_d = self._return_z_d()
        self._s = self._return_s()
        self._s_fit = self._return_s_fit()
        self._k_silk = self._return_k_silk()

        self._alpha_c = self._return_alpha_c()
        self._beta_c = self._return_beta_c()
        self._alpha_b = self._return_alpha_b()
        self._beta_b = self._return_beta_b()
        self._beta_node = self._return_beta_node()


    def _attach_local_cosmo_parameter_names(self):
        """Create local versions of cosmology parameters."""
        self._Om = self.cosmo_params['Omega_m']
        self._Ob = self.cosmo_params['Omega_b']
        self._Oc = self.cosmo_params['Omega_c']
        self._h = self.cosmo_params['h']
        Tcmb_mag = self.cosmo_params['T_cmb'].rescale('K').magnitude
        self.cu = units.CosmoUnits(self._h, 1.0)
        self._ThetaCMB = Tcmb_mag / 2.7 * self.cu.dimensionless
        self._omhh = self._Om * self._h * self._h
        self._obhh = self._Ob * self._h * self._h
        self._Gamma = self._Om * self._h
        self._fb = self._Ob / self._Om
        self._fc = self._Oc / self._Om


    #=========================================
    # Transfer function parameters
    #=========================================

    def _return_z_eq(self):
        r"""Set the matter-radiation equality redshift (Eq. 2)."""
        z_eq = 2.5e4 * self._omhh * self._ThetaCMB**(-4) - 1
        return z_eq

    def _return_k_eq(self):
        r"""Set horizon scale at matter-radiation equality (Eq. 3)."""
        k_eq = 7.46e-2 * self._Om * self._h * self._ThetaCMB**(-2)
        return k_eq * self.cu.h / self.cu.Mpc

    def _return_z_d(self):
        r"""Set the drag epoch redshift (Eq. 4)."""
        omhh = self._omhh
        obhh = self._obhh
        b1 = 0.313 * omhh**(-0.419) * (1 + 0.607 * omhh**(0.674))
        b2 = 0.238 * omhh**(0.223)
        t3 = 1 + b1 * obhh**b2
        return 1291 * (omhh**0.251 / (1 + 0.659 * omhh**0.828)) * t3

    def _return_R(self, z):
        r"""Return baryon / photon momentum density ratio (Eq. 5).

        :param z: redshift
        :type z: ``float``
        """
        R = 31.5 * self._obhh * self._ThetaCMB**(-4) * 1.0e3 / (1 + z)
        return R

    def _return_s(self):
        r"""Set the sound horizon at the drag epoch (Eq. 6)."""
        Req = self._return_R(self._z_eq)
        Rd = self._return_R(self._z_d)
        t1 = 2.0 / (3.0 * self._k_eq) * np.sqrt(6.0 / Req)
        t2 = np.sqrt(1.0 + Rd) + np.sqrt(Rd + Req)
        t3 = 1.0 + np.sqrt(Req)
        s = t1 * np.log(t2 / t3)
        return s.rescale('Mpc/hh')

    def _return_s_fit(self):
        r"""Set a fit to the sound horizon at the drag epoch (Eq. 26)."""
        num = 44.5 * np.log(9.83 / self._omhh)
        den = np.sqrt(1 + 10 * self._obhh**(3./4))
        s = num / den * self.cu.Mpc
        return s.rescale('Mpc/hh')

    def _return_k_silk(self):
        r"""Return the silk damping scale (Eq. 7)."""
        t1 = 1.6 * self._obhh**0.52 * self._omhh**0.73
        t2 = 1 + (10.4 * self._omhh)**(-0.95)
        k_silk = t1 * t2 / self.cu.Mpc
        return k_silk.rescale('hh/Mpc')

    def _return_alpha_c(self):
        r"""Return cold dark matter suppression parameter (Eq. 11)."""
        a1 = (46.9 * self._omhh)**0.670 * (1 + (32.1 * self._omhh)**(-0.532))
        a2 = (12.0 * self._omhh)**0.424 * (1 + (45.0 * self._omhh)**(-0.582))
        fb3 = self._fb * self._fb * self._fb
        return a1**(-self._fb) * a2**(-fb3)

    def _return_beta_c(self):
        r"""Return cold dark matter log shift parameter (Eq. 12)."""
        b1 = 0.944 / (1 + (458 * self._omhh)**(-0.708))
        b2 = (0.395 * self._omhh)**(-0.0266)
        return 1.0 / (1 + b1 * (self._fc**b2 - 1))

    def _return_alpha_b(self):
        r"""Return baryon supression parameter (Eq. 14 + 15)."""
        Rd = self._return_R(self._z_d)
        y = (1 + self._z_eq) / (1 + self._z_d)
        sq1py = np.sqrt(1 + y)
        G = y * (-6 * sq1py + (2 + 3 * y) * np.log((sq1py + 1)/(sq1py - 1)))
        t1 = 2.07 * self._k_eq * self._s * (1 + Rd)**(-3.0/4)
        return t1 * G

    def _return_beta_b(self):
        r"""Return baryon envelope shift parameter (Eq. 24)."""
        t1 = 0.5 + self._fb
        t2 = (3 - 2 * self._fb) * np.sqrt((17.2 * self._omhh)**2 + 1)
        return t1 + t2

    def _return_beta_node(self):
        r"""Return baryon transfer function shift scale (Eq. 23)."""
        return 8.41 * self._omhh**0.435

    def _return_s_tilde(self, k):
        r"""Return shifted sound horizon (Eq. 22).

        :param k: wavenumber
        :type k: ``quantity``
        """
        ks = k * self._s
        return self._s / (1 + (self._beta_node / ks)**3)**(1./3)

    def _return_q(self, k):
        """Return :math:`q` using the traditional definition of the shape
        parameter :math:`\Gamma = \Om h` (Eq. 28).

        :param k: wavenumber
        :type k: ``quantity``
        """
        q = k / (self._k_eq * 13.41)
        return q.simplified

    def _return_q_eff(self, k):
        r"""Returns :math:`\qeff` calculated using the effective shape parameter
        :math:`\Gamma_{\rm eff}` (Eq. 28 + 30).

        :param k: wavenumber
        :type k: ``quantity``
        """
        Gamma_eff = self._return_Gamma_eff(k)
        k.units = 'hh/Mpc'
        q_eff = k.magnitude * self._ThetaCMB**2 / Gamma_eff
        return q_eff

    def _return_Gamma_eff(self, k):
        """Return effective shape parameter (Eq. 30).

        :param k: wavenumber
        :type k: ``quantity``
        """
        a1 = 0.328 * np.log(431.0 * self._omhh) * self._fb
        a2 = 0.3089 * np.log(22.3 * self._omhh) * self._fb**2
        ag = 1.0 - a1 + a2
        t1 = 1 + (0.43 * k * self._s)**4
        return self._Gamma * (ag + (1 - ag) / t1)

    def _preprocess_input_k(self, k_in):
        r"""Assert that ``k_in`` is a quantity (i.e. has units) and set them
        to :math:`h \, {\rm Mpc}^{-1}`.

        :param k_in: wavenumber
        :type k_in: ``quantity``
        """
        if not hasattr(k_in, 'units'):
            raise ValueError('input k must have units')
        return k_in.rescale('hh/Mpc')


    #=========================================
    # Transfer function fit pieces
    #=========================================

    def _Tb(self, k):
        """The baryonic part of the full transfer function fit (Eq. 21)."""
        q = self._return_q(k)
        ks = k * self._s
        t1 = self._T0(q, 1.0, 1.0) / (1 + (ks / 5.2)**2)
        t2 = self._alpha_b / (1 + (self._beta_b / ks)**3)
        t3 = np.exp(-(k / self._k_silk)**1.4)
        x = k * self._return_s_tilde(k)
        t4 = np.sin(x) / x
        return (t1 + t2 * t3) * t4

    def _Tc(self, k):
        """The CDM part of the of the full transfer function fit (Eq. 17)."""
        ks = k * self._s
        f = 1.0 / (1 + (ks / 5.4)**4)
        q = self._return_q(k)
        Tc = (
            f * self._T0(q, 1.0, self._beta_c) +
            (1.0-f) * self._T0(q, self._alpha_c, self._beta_c)
        )
        return Tc

    def _T0(self, q, alpha_c, beta_c):
        """The pressureless transfer function fit (Eq. 19)."""
        C = self._C(q, alpha_c)
        num = np.log(np.exp(1.0) + 1.8 * beta_c * q)
        return num / (num + C * q * q)

    def _C(self, q, alpha_c):
        """Parameter for pressureless transfer function fit (Eq. 20)."""
        C1 = 14.2 / alpha_c
        C2 = 386.0 / (1 + 69.9 * q**(1.08))
        return C1 + C2


    #=========================================
    # Transfer function fits
    #=========================================

    def T(self, k_in):
        """The full transfer function fit (Eq. 16)."""
        k = self._preprocess_input_k(k_in)
        Tb = self._Tb(k)
        Tc = self._Tc(k)
        T = self._fb * self._Tb(k) + self._fc * self._Tc(k)
        return T

    def T_zero_baryon(self, k_in):
        r"""Return the "zero baryon" transfer function fit (Eq. 29).

        An approximate form of the fit that treats all baryons as if they
        were cold dark matter (the zero baryon case).

        :parameter k_in: input wavenumbers
        :type k_in: ``quantity``
        """
        k = self._preprocess_input_k(k_in)
        q = self._return_q(k)
        L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q * q)

    def T_no_wiggles(self, k_in):
        r"""Return the "no wiggles" transfer function fit (Eq. 29 w/ 30+31).

        An approximate form of the fit that models the shape of the transfer
        function but not the oscillations due to baryons.

        :parameter k_in: input wavenumbers
        :type k_in: ``quantity``
        """
        k = self._preprocess_input_k(k_in)
        q = self._return_q_eff(k)
        L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q * q)
