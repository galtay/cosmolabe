import numpy as np
import matplotlib.pyplot as plt

import cosmolabe as cl
from cosmolabe.parameters import PLANCK15
from cosmolabe.transfer_functions import EH98

plt.ion()

def trap( x, y ):
    """ Trapezoidal integration rule that preseves units. """

    dx = x[1:] - x[0:-1]
    I = 0.5 * ( y[0:-1] + y[1:] ) * dx
    I = np.sum(I)
    return I



co = cl.cosmology.Cosmology(PLANCK15)
sys.exit(1)

eh98 = EH98(PLANCK15)
cu = eh98.cu

H0 = 100.0 * PLANCK15['h'] * cu.km / cu.s / cu.Mpc
#k_arr = np.logspace(-3.0, 2.0, 100000) * cu.h / cu.Mpc
k_arr = np.logspace(-3.0, 2.0, 1000000) * cu.h / cu.Mpc
Tk = eh98.T(k_arr)

k_scl = (cl.pc.c * k_arr / H0).rescale('dimensionless')

k0 = 5.0e-2 / cu.Mpc
ns = PLANCK15['ns']
pp = 3.0 + ns
k_ratio = (k_arr / k0).rescale('dimensionless')

Dk2 = k_scl**(3.0+ns) * Tk * Tk

R8 = 8.0 * cu.Mpc / cu.h
x = k_arr * R8
j1 = (np.sin(x) - x * np.cos(x)) / (x*x)
W = 3 * j1 / x
W2 = W * W

plt.loglog(k_arr, Dk2 * W2, color='red', lw=2.0, ls='-',
           label='full fit')

plt.grid(which='major', ls='-', lw=1.0, color='grey', alpha=0.5)
plt.grid(which='minor', ls='-', lw=1.0, color='grey', alpha=0.5)

plt.xlabel(r'$k \; [h \, {\rm Mpc}^{-1}]$', fontsize=20)
plt.ylabel(r'$\Delta^2(k)$', fontsize=20)

#[3 j1(k*R) / (k*R)]^2
#j1(x) = (sin x - x cos x) / x^2







kw = np.array(1.0/R8)
lnkw = np.log( kw )

lnk_min = lnkw - 4.0
lnk_max = lnkw + 4.0

Nk = 8000
lnk = np.linspace(lnk_min, lnk_max, Nk)
kk = np.exp(1.0)**lnk * cu.h / cu.Mpc

x = kk * R8
j1 = (np.sin(x) - x * np.cos(x)) / (x*x)
W = 3 * j1 / x
W2 = W * W

k_scl = (cl.pc.c * kk / H0).rescale('dimensionless')
Tk = eh98.T(kk)
Dk2 = k_scl**(3.0+ns) * Tk * Tk

dsig2_dk = Dk2 * W2

#plt.legend(loc='best')
plt.tight_layout()
plt.show()
