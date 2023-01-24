# The Cosmological Collider Flow

This folder contains the movies of the time evolution of the Cosmological Collider signals. We are interested in the three-point correlation function of the curvature perturbation in Fourier space, known as the bispectrum. Following standard conventions, we define the dimensionless shape function S such that 

$$
\langle \zeta_{\vec{k}_1} \zeta_{\vec{k}_2} \zeta_{\vec{k}_3}\rangle' = \frac{(2\pi)^4}{(k_1 k_2 k_3)^2} \Delta_\zeta^4 S(k_1, k_2, k_3),
$$

where $\Delta_\zeta^2 = \frac{k^3}{2\pi^2} \langle \zeta_{\vec{k}} \zeta_{-\vec{k}}\rangle'$ is the dimensionless power spectrum of the curvature perturbation. The movies below display the shape function as a function of time for the following Lagrangian

$$
\mathcal{L}/a^3 = -\frac{1}{2}(\partial_\mu \pi_c)^2 - \frac{1}{2}(\partial_\mu \sigma)^2 -\frac{1}{2}m^2\sigma^2 + \rho \dot{\pi}_c\sigma - \frac{1}{2}\alpha\dot{\pi}_c \sigma^2,
$$

both for $\rho/H = 0.1$ (weak mixing) and $\rho/H = 5$ (strong mixing). We also define the effective frequency of the cosmological collider signal $\mu_{\rm{eff}}^2 = m_{\rm{eff}}^2/H^2 - 9/4$ with $m_{\rm{eff}}^2 = m^2 + \rho^2$.

### Weak Mixing

For $\mu_{\rm{eff}} = 5, \rho/H = 0.1$

<p align="center">
  <img src="CosmologicalColliderFlow_WeakMixing.gif">
</p>

### Strong Mixing

For $\mu_{\rm{eff}} = 5, \rho/H = 5$

<p align="center">
  <img src="CosmologicalColliderFlow_StrongMixing.gif">
</p>

Note that we have normalized the shape functions to unity in the equilateral configuration. The measure of time is expressed in number of efolds.

