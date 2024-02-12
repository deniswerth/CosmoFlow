[![Python](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://python.org)
[![ArXiv](https://img.shields.io/badge/arXiv-2210...-yellowgreen.svg)](https://google.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)


# The Cosmological Flow

<p align="center">
  <img src="CosmoFlowLogo.jpg">
</p>

This repository contains details on CosmoFlow, a Pyhton package to automatically compute cosmological correlators. More details on the method and applications can be found in the paper:

[![ArXiv](https://img.shields.io/badge/arXiv-2302...-yellowgreen.svg)](https://arxiv.org/abs/2302.00655) [![ArXiv](https://img.shields.io/badge/arXiv-2312...-yellowgreen.svg)](https://arxiv.org/pdf/2312.06559.pdf) [![ArXiv](https://img.shields.io/badge/arXiv-2312...-yellowgreen.svg)]([https://arxiv.org/pdf/2312.06559.pdf](https://arxiv.org/pdf/2402.03693.pdf))


[Cosmological Flow of Primordial Correlators](https://arxiv.org/abs/2302.00655) [short paper]

[The Cosmological Flow: a Systematic Approach to Inflationary Correlators](https://arxiv.org/pdf/2312.06559.pdf) [long paper]

* [Denis Werth](mailto:werth@iap.fr) -- Sorbonne University, Institut d'Astrophysique de Paris (IAP)
* [Lucas Pinol](mailto:lucas.pinol@phys.ens.fr) -- Laboratoire de Physique de l'École Normale Supérieure (LPENS), ENS, CNRS, Université PSL, Sorbonne Université, Université Paris Cité
* [Sébastien Renaux-Petel](mailto:petel@iap.fr) -- CNRS, Institut d'Astrophysique de Paris (IAP)

The code is free of use but for any research done with it you are kindly asked to cite the code paper together with the two articles that presented the Cosmolofical Flow.

## Cosmological Correlators

The Cosmological Flow approach is based on computing cosmological correlators by solving differential equations in time governing their time evolution through the entirety of the spacetime during inflation, from their origin as quantum fluctuations in the deep past to the end of inflation. This method takes into account all physical effects at tree-level without approximation. Specifically, CosmoFlow computes the two- and three-point correlators of fields and/or conjugate momenta $X^{\mathsf{a}}$ in Fourier space

$$
\langle X^{\mathsf{a}}(\vec{k}_1) X^{\mathsf{b}}(\vec{k}_2)\rangle = (2\pi)^3 \delta^{(3)}(\vec{k}_1 + \vec{k}_2) \Sigma^{\mathsf{ab}}(k_1),
$$

$$
\langle X^{\mathsf{a}}(\vec{k}_1) X^{\mathsf{b}}(\vec{k}_2) X^{\mathsf{c}}(\vec{k}_3) \rangle = (2\pi)^3 \delta^{(3)}(\vec{k}_1 + \vec{k}_2+ \vec{k}_2) B^{\mathsf{abc}}(k_1, k_2, k_3),
$$

from any theory of inflationary fluctuations i.e. that includes an arbitrary number of degrees of freedom with any propagation speeds, couplings, and time-dependencies.

## Getting Started with a Tutorial Notebook

For the boundless possibilities that this approach provides, we have made our numerical code CosmoFlow available for the community. The code paper containing a detailed user guide and applications accompanied with readyto-use Jupyter Notebooks is:

[![ArXiv](https://img.shields.io/badge/arXiv-2302...-yellowgreen.svg)](google.com) [![ArXiv](https://img.shields.io/badge/arXiv-2312...-yellowgreen.svg)](google.com)

### Prerequisites

CosmoFlow can be used on any system that supports Python 3. The following modules/packages are required:

* Python (3)
* numpy
* scipy
* [tqdm](https://tqdm.github.io/) (for numerical integration visualisation)
* matplotlib

### Usage

The numerical routine CosmoFlow is composed of several modules:

* Parameters.py takes as inputs all the (time-dependent) couplings and background variables of the model the user wants to study, and generate continuous functions out of the given arrays.
* Theory.py defines the $u$ tensors that define the flow equations for the two- and three-point correlators. One needs to manually give the elements of the $\Delta, M, I, A, B, C$ and $D$ tensors that depend on the specific theory one considers.
* Solver.py contains the solver for the flow equation integration. It also defines the flow equations, the initial conditions for all two- and three-point correlators.

Follow the code paper for a step-by-step user guide.

### Tutorials

The code paper provides a lot of tutorial Jupyter Notebooks (in particular reproducing all the figures of the paper) to gain ease when using CosmoFlow. For those interested in learning more about this tool, do not hesitate to contact the developer.

## Extensions and Future Work

We have turned the Cosmological Flow approach into an open-source tool available for theorists, phenomenologists, and cosmologists. This tool is clearly just a first building-block for a far-reaching program of exploring the rich physics of inflation. A complete package would require several improvements: 

* Such tool should be easy to install and use, and easily scriptable. In particular, performing the Legendre transform to derive the Hamiltonian given a Lagrangian should be performed in an automatic manner. Second, we ideally want this tool to be fast, producing instantaneous results. 
* We plan to extend the Cosmological Flow approach to include spinning fields and to be able to compute the trispectrum, as well as compute loop-level diagrams.
* At the moment, the code is rather slow, especially when it comes to computing the full shape of the bispectrum i.e. scanning over all triangle configurations. Performing parameter constraints using numerical outputs given by CosmoFlow directly from a theory of inflationary fluctuations is at the moment out of reach. The main limitation is the integration time of the correlators inside the horizon, which grows exponentially with the number of subhorizon efolds. Scanning soft limits is therefore computationally very expensive. Such improvement calls for a better and more sophisticated numerical implementation than the one that was used so far. Ideally, the solver should be hard-coded in C++ with a Python wrapper as interface for flexibility.
* More ambitiously, we could include the CosmoFlow routine in the already-existing chain of late-time cosmological tools. This would automate the generation of theoretical primordial data that can be directly used for CMB or LSS observables.

## Licensing 

CosmoFlow is distributed under the MIT license. See [MIT License](https://en.wikipedia.org/wiki/MIT_License) for more details. If you want to use this free-of-use code for your own research, it would be appreciated that you cite the relevant two Cosmological Flow papers. 

## Author

* [Denis Werth](mailto:werth@iap.fr) -- Sorbonne University, Institut d'Astrophysique de Paris (IAP)

