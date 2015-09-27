import numpy as np
import matplotlib.pyplot as plt

import cosmolabe as cl
from cosmolabe.parameters import PLANCK15
from cosmolabe.transfer_functions import EH98

plt.ion()





eh98 = EH98(PLANCK15)
cu = eh98.cu

H0 = 100.0 * PLANCK15['h'] * cu.km / cu.s / cu.Mpc
k_arr = np.logspace(-3.0, 0.0, 1000) * cu.h / cu.Mpc
Tk = eh98.T(k_arr)

k_scl = (cl.pc.c * k_arr / H0).rescale('dimensionless')

k0 = 5.0e-2 / cu.Mpc
ns = PLANCK15['ns']
pp = 3.0 + ns
k_ratio = (k_arr / k0).rescale('dimensionless')

Dk2 = k_scl**(3.0+ns) * Tk * Tk

plt.loglog(k_arr, Dk2, color='red', lw=2.0, ls='-',
           label='full fit')

plt.grid(which='major', ls='-', lw=1.0, color='grey', alpha=0.5)
plt.grid(which='minor', ls='-', lw=1.0, color='grey', alpha=0.5)

plt.xlabel(r'$k \; [h \, {\rm Mpc}^{-1}]$', fontsize=20)
plt.ylabel(r'$\Delta^2(k)$', fontsize=20)








#plt.legend(loc='best')
plt.tight_layout()
plt.show()
