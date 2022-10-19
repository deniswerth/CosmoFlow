[![Python](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://python.org)
[![ArXiv](https://img.shields.io/badge/arXiv-2210...-yellowgreen.svg)](https://google.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)



# The Cosmological Flow

* Paragraph of intro presenting the repository

* Link to arxiv paper.

## Getting Started

## Example Notebook

## Examples of Inflationary Correlators

* sentence saying that we provide several examples of inflationary correlators, look at the paper)
* Give a few examples of plots + minimal explanation

## Extensions and Future Work

We have turned the Cosmological Flow approach into an open-source tool available for cosmologists. This tool is clearly just a first building-block for a far-reaching program of exploring the rich physics of inflation. A complete package would require several improvements: 

* Such tool should be easy to install and use, and easily scriptable. In particular, performing the Legendre transform to derive the Hamiltonian given a Lagrangian should be performed in an automatic manner. Second, we ideally want this tool to be fast, producing instantaneous results. 
* At the moment, the code is rather slow, especially when it comes to computing the full shape of the bispectrum i.e. scanning over all triangle configurations. Performing parameter constraining using numerical outputs given by CosmoFlow directly from a theory of inflationary fluctuations is at the moment out of reach. The main limitation is the integration time of the correlators inside the horizon, which grows exponentially with the number of subhorizon efolds. Scanning the squeezed limit is therefore computationally very expensive. Such improvement calls for a better and more sophisticated numerical implementation that the one that was used so far. Ideally, the solver should be hard-coded in C++ with a Python wrapper as interface for flexibility.
* More ambitiously, we could include the CosmoFlow routine in the already-existing chain of late-time cosmological tools. This would automate the generation of theoretical primordial data that can be directly used for CMB or LSS observables.

## Licensing 

CosmoFlow is distributed under the MIT license. See [MIT License](https://en.wikipedia.org/wiki/MIT_License) for more details.

## Author

* [Denis Werth](mailto:werth@iap.fr) -- Sorbonne University, Institut d'Astrophysique de Paris (IAP)

