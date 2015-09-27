import numpy as np
import matplotlib.pyplot as plt

import cosmolabe as cl
from cosmolabe.parameters import PLANCK15
from cosmolabe.transfer_functions import EH98

plt.ion()

eh98_cosmo_params = {
    'Omega_m': 0.2, 'Omega_b': 0.1, 'Omega_c': 0.1, 'h': 0.5,
    'T_cmb': 2.728 * cl.u.K
}

def main():

    dat = np.loadtxt('trans.dat')

    eh98 = EH98(eh98_cosmo_params)
    k_arr = np.logspace(-3.0, 0.0, 1000) * eh98.cu.h / eh98.cu.Mpc

    Tk_no_wiggles = eh98.T_no_wiggles(k_arr)
    Tk_zero_baryon = eh98.T_zero_baryon(k_arr)
    Tk = eh98.T(k_arr)

    plt.loglog(k_arr, Tk_zero_baryon, color='green', lw=2.0, ls='-',
               label='zero baryon')

    plt.loglog(k_arr, Tk_no_wiggles, color='lime', lw=2.0, ls='-',
               label='no wiggles')

    plt.loglog(k_arr, np.abs(Tk), color='red', lw=2.0, ls='-',
               label='full fit')

    plt.loglog(dat[:,0], dat[:,1], color='blue', lw=1.0, ls='--',
               label='original')

    plt.grid(which='major', ls='-', lw=1.0, color='grey', alpha=0.5)
    plt.grid(which='minor', ls='-', lw=1.0, color='grey', alpha=0.5)

    plt.xlabel(r'$k \; [h \, {\rm Mpc}^{-1}]$', fontsize=20)
    plt.ylabel(r'$|T(k)|$', fontsize=20)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
