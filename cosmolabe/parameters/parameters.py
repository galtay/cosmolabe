from cosmolabe.units import units

u = units.Units()

_PLANCK15_raw = {
    'Omega_b_h2': 0.02230,
    'Omega_c_h2': 0.1188,
    'tau': 0.066,
    'ln10^10As': 3.064,
    'ns': 0.9667,
    'h': 0.6774,
    'Omega_l': 0.6911,
    'Omega_m': 0.3089,
    'Omega_m_h2': 0.14170,
    'sigma_8': 0.8159,
    '10^9As': 2.142,
    'z_re': 8.8,
    'T_cmb': 2.7255 * u.K,
    'reference': ('http://adsabs.harvard.edu/abs/2015arXiv150201589P, '
                  'Table 4, last column, TT,TE,EE+lowP+lensing+ext'),
}

#: Planck 2015 parameters.
PLANCK15 = dict(_PLANCK15_raw)
PLANCK15.update({
    'Omega_b': _PLANCK15_raw['Omega_b_h2'] / _PLANCK15_raw['h']**2,
    'Omega_c': _PLANCK15_raw['Omega_c_h2'] / _PLANCK15_raw['h']**2,
    'As': _PLANCK15_raw['10^9As'] / 1.0e9,
})
