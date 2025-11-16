# flip-cpd: Flip Graph Search for Canonical Polyadic Decomposition

<!-- [![arXiv:2502.13266](https://img.shields.io/badge/arXiv-2502.13266-b31b1b.svg)](https://www.arxiv.org/pdf/2502.13266) -->
[![Live Demo](https://img.shields.io/badge/Live%20Demo-qdiag.xyz%2Fcpd-0b5fff?logo=google-chrome&logoColor=white)](https://qdiag.xyz/cpd/)

This repository provides tools for searching low-rank decompositions of arbitrary 3-way tensors, in particular encoding bilinear maps. If a bilinear map takes inputs with coordinates $A_i$, $B_j$ and produces outputs $C_k$, it can be written as
$$
C_k = \sum_{i,j} T_{ijk} A_i B_j
$$
for some tensor $T_{ijk}$. A rank-$r$ algorithm corresponds to a factorization
$$
T_{ijk} = \sum_{q=1}^r U_{qi} V_{qj} W_{qk},
$$
which for the matrix multiplication reduces the number of scalar multiplications to $r$. A famous example is the Strassen's scheme for $2 \times 2$ matrices:

<p align="center">
  <img
    src="docs/assets/fig1.png"
    alt="Strassen-like low-rank decomposition for 2×2 matrix multiplication (Fig. 1)"
    width="800"
  >
</p>

The main workflow in this project is to first search for such tensor decompositions over a small finite field, most commonly $\mathbb{F}_2$ or $\mathbb{F}_3$. We look for low-rank decompositions modulo a prime, then use Hensel lifting to pass to an extension field $\mathbb{F}_{p^k}$ with moderate $k$, and finally apply rational reconstruction to obtain coefficients over $\mathbb{Z}$ or $\mathbb{Q}$. For general matrix multiplication and structured matrix multiplication, `scripts/generator.py` can be used to generate the corresponding tensors (see `docs/generator.md`).

## Installation

The core tools are written in C++20 and have no external runtime dependencies beyond the standard library and threads. All third-party headers (`CLI11.hpp`, `cnpy.h`, `picojson.h`) are vendored in `third_party/`.

One way to build is via CMake:

```bash
cmake -S . -B build
cmake --build build
```

This will compile all binaries (`search2/3`, `lift2/3`, `select2/3`, `select_modular2/3`) and place them in the `bin/` directory. If you prefer not to use CMake, each tool can also be built directly from the corresponding `src/{name}.cpp` file with a single compiler invocation. A typical command for a field looks like

```bash
g++ -DMOD2 -Ofast -std=c++20 -march=native -Ithird_party -pthread src/search.cpp -o bin/search2
```

and analogously for `MOD3` and other sources (`lift.cpp`, `select.cpp`, `select_modular.cpp`).

For generating tensors and running the example notebooks you additionally need Python 3 with a few packages:

* `numpy` and `typer` for `scripts/generator.py`;
* `sympy` for verifying schemes in `examples/`.

## Usage

The C++ tools are organized around three main stages: modular search (`search2`/`search3`), lifting to characteristic zero (`lift2`/`lift3`), and selection of schemes (`select2`/`select3` and `select_modular2`/`select_modular3`). Given a tensor name from `data/tensors/`, a typical workflow is to run `search`, then `lift`, and finally `select` on the resulting schemes. An end-to-end example of this pipeline is provided in `examples/`, and `docs/cli.md` documents all command-line options.

`search`. These programs perform pool-based flip graph search for low-rank decompositions over a finite field. `search2` works over $\mathbb{F}_2$ and `search3` over $\mathbb{F}_3$. The same `2`/`3` suffix convention is used for the other tools. They take a tensor name (matching an entry in `data/tensors/`) and explore the flip graph.

`lift`. These tools take modular schemes found by `search2`/`search3` and perform Hensel lifting followed by rational reconstruction. Starting from schemes over $\mathbb{F}_2$ or $\mathbb{F}_3$, they lift to an extension field $\mathbb{F}_{p^k}$ and then reconstruct coefficients over $\mathbb{Z}$ or $\mathbb{Q}$. The resulting lifted and rational schemes are written to the corresponding directories under `data/`.

`select`. Once a collection of rational schemes is available, these programs read them, analyse possible recursions, and select Pareto-optimal schemes according to rank, number of recursion calls and number of additions. The selected schemes are written both in `.npy` format and in a human-readable `.txt` format.

`select_modular`. These variants run a similar selection and analysis procedure directly on modular schemes, without lifting.

## Output

Selected schemes are written to `data/schemes_selected/` in two formats: `.npy` files containing the triplet $(U, V, W)$ in a fixed binary layout, and `.txt` files with a human-readable description of the same scheme. See `examples/` and `docs/formats.md` for details. 

In addition, the repository includes a set of reference schemes from the paper in `data/schemes_paper/`, provided in both `.npy` and `.txt` form. These files can be loaded and analysed with the same tooling as newly generated schemes.


## Citation

If you use this code or the accompanying datasets in academic work, please cite:

```bibtex
@misc{khoruzhii_2025,
  title         = {Faster Algorithms for Structured Matrix Multiplication via Flip Graph Search},
  author        = {Kirill Khoruzhii and Patrick Gelß and Sebastian Pokutta},
  year          = {2025},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SC}
}
```
