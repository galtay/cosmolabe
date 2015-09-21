_return_z_eq = r"""

.. pull-quote::

    `"In the usual cosmological paradigm, nonrelativistic particles
    (baryons, electrons, and CDM) dominate relativistic particles
    (photons and massless neutrinos) in density today.  However,
    because the density of these two classes of particles scale
    differently in time, at an earlier time, the reverse situation
    held.  The transition from a radiation-dominated universe to a
    matter-dominated one occurs roughly at`

.. math::

    \zeq = 2.50 \times 10^4
    \Om h^2 \ThetaCMB^{-4},

.. pull-quote::

    `the redshift where the two classes have equal density."`

    -- EH98
"""

_return_k_eq = r"""

.. pull-quote::

    `"As density perturbations behave differently in a radiation
    dominated universe versus a matter dominated one due to
    pressure support, the scale of the particle horizon at the
    equality epoch` :math:`\zeq`,

.. math::

    \keq = (2 \Om H_0^2 \zeq)^{1/2} =
    7.46 \times 10^{-2} \Om h^2 \ThetaCMB^{-2}
    \, {\rm Mpc}^{-1},

.. pull-quote::

    `is imprinted on the matter transfer function; in particular,
    perturbations on smaller scales are supressed in amplitude in
    comparison to those on large scales."`

    -- EH98
"""

_return_z_d = r"""

.. pull-quote::

    `"We thus define the drag epoch` :math:`z_d` `as the time at which
    the baryons are released from the Compton drag of the photons in
    terms of a weighted integral over the Thomson scattering rate
    (see` HS96_ `eqs. [C8], [E2]).  A fit to the numerical recombination
    results is`

.. math::

    \begin{aligned}
      z_d &= 1291 \frac{(\Om h^2)^{0.251}}
        {1 + 0.659(\Om h^2)^{0.828}}
        [1 + b_1 (\Ob h^2)^{b_2}] \\
      b_1 &= 0.313 (\Om h^2)^{-0.419}
        [1 + 0.607 (\Om h^2)^{0.674}] \\
      b_2 &= 0.238 (\Om h^2)^{0.223}
    \end{aligned}

.. pull-quote::

    `where we have reduced` :math:`z_d` `by a factor of 0.96 from`
    HS96_ `on phenomenological grounds.  For` :math:`\Ob h^2
    \lesssim 0.03` `, this epoch follows last scattering of the
    photons."`

    -- EH98

.. _HS96: http://adsabs.harvard.edu/abs/1996ApJ...471..542H
"""

_return_R = r"""

:param z: redshift
:type z: ``float``

.. pull-quote::

    `"Prior to` :math:`z_d` `, small-scale perturbations in the
    photon-baryon fluid propagate as acoustic waves.  The sound speed
    is` :math:`c_s = 1 / [3(1+R)]^{1/2}` `(in units where the speed of
    light is unity) where R is the ratio of the baryon to photon
    momentum density,`

.. math::

    R \equiv (3 \rho_b) / (4 \rho_{\gamma}) = 31.5 \Ob h^2
    \ThetaCMB^{-4} (z/10^3)^{-1}."

.. pull-quote::

    -- EH98
"""


_return_s = r"""

.. pull-quote::

    `"We define the sound horizon at the drag epoch as the co-moving
    distance a wave can travel prior to redshift`

.. math::

    \begin{aligned}
      s &= \int_0^{t(z_d)} c_s (1 + z) dt \\
        &= \frac{2}{3 \keq} \sqrt{\frac{6}{R_{\rm eq}}} \ln
        \frac{\sqrt{1 + R_d} + \sqrt{R_d + R_{\rm eq}}}
             {1 + \sqrt{R_{\rm eq}}}
    \end{aligned}

.. pull-quote::

    `where` :math:`R_d \equiv R(z_d)` `and` :math:`R_{\rm eq} \equiv
    R(\zeq)` `are the values of` :math:`R` `at the drag epoch
    of matter-radiation equality, respectively."`

    -- EH98

.. seealso::

   :py:func:`_return_R`, :py:func:`_return_k_eq`
"""

_return_q = r"""

Returns :math:`q` using the traditional definition of the shape
parameter :math:`\Gamma = \Om h`.

:param k: wavenumber
:type k: ``quantity``


.. pull-quote::

    `"Here the transfer function is parameterized by`
    :math:`q \propto k / \keq` `more commonly expressed as a
    shape parameter` :math:`\Gamma = \Om h` `, where`

.. math::

    q = \frac{k}{h \, {\rm Mpc^{-1}}}
    \frac{\ThetaCMB^2}{\Gamma}

.. pull-quote::

    -- EH98
"""

_return_q_eff = r"""

Returns :math:`\qeff` calculated using the effective shape parameter
:math:`\Gamma_{\rm eff}` defined in Eq. 30 of EH98_.

:param k: wavenumber
:type k: ``quantity``

.. math::

    q_{\rm eff} = \frac{k}{h \, {\rm Mpc^{-1}}}
    \frac{\ThetaCMB^2}{\Gamma_{\rm eff}}


.. seealso::

    :py:func:`_return_Gamma_eff`

.. _EH98: http://adsabs.harvard.edu/abs/1998ApJ...496..605E

"""

_return_Gamma_eff = r"""

:param k: wavenumber
:type k: ``quantity``

.. pull-quote::

    `"A reasonable fit to the nonoscillatory part of the transfer
    function can be written by rescaling` :math:`\Gamma_{\rm eff}(k)`
    `as one moves through the sound horizon`

.. math::

    \begin{aligned}
      \Gamma_{\rm eff}(k) &= \Om h \left[ \aG +
        \frac{1 - \aG} {1 + (0.43\,k\,s)^4} \right] \\
      \aG &= 1 - 0.328 \ln(431 \Om h^2) \fb +
                 0.380 \ln(22.3 \Om h^2) \left(\fb\right)^2"
    \end{aligned}

.. pull-quote::

    -- EH98
"""

T_fit = r"""

:parameter k_in: input wavenumbers
:type k_in: ``quantity``
:parameter use_q_eff: if ``True`` use the effective shape parameter as
  described in Eq. 30 (see :py:func:`_return_q_eff`).  Otherwise, the
  traditional value :math:`\Gamma = \Om h` is used (see :py:func:`_return_q`).
:type use_q_eff: ``bool``

.. pull-quote::

    `"A commonly used fitting formula to the zero-baryon limit
    was presented in` Bardeen96_ `(Eq. G3).  However this formula
    fits neither the exact small-scale solution of Section 2.2 nor
    does it have the quadratic deviation from unity required by the
    theory.  The latter is a fundamental requirement of causality
    (Zeldovich, 1965, Adv. Astron. Astrophys., 3, 241), in that one
    power of` :math:`k` `must arise from stress gradients generating
    bulk velocity and a second from velocity gradients generating
    density perturbations.  In fact the coefficient of this quadratic
    derivation can be calculated pertubatively if the stress gradients
    are dominated by the isotropic (pressure) term.
    The following functional form satisfies these criteria and is a
    better fit to the zero-baryon case extrapolated from trace
    baryon models calculated by CMBfast."`

.. math::

    \begin{aligned}
      T_0(q) &= \frac{L_0}{L_0 + C_0 q^2} \\
      L_0(q) &= \ln(2 \exp + 1.8 q) \\
      C_0(q) &= 14.2 + \frac{731}{1 + 62.5 q}
    \end{aligned}

.. pull-quote::

    -- EH98

.. _Bardeen96: http://adsabs.harvard.edu/abs/1986ApJ...304...15B
"""
