# ENPOWER: exact power indices for plurality-voting ensembles

[![tests](https://github.com/iliesarpe/enpower/actions/workflows/tests.yml/badge.svg)](https://github.com/iliesarpe/enpower/actions/workflows/tests.yml)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](#citation)
[![arXiv](https://img.shields.io/badge/arXiv-2610.17980-b31b1b.svg)](https://arxiv.org/abs/2610.17980)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus)](cpp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Reproducing the paper](REPRODUCE.md)** | **[Implementation notes](docs/IMPLEMENTATION.md)** | **[Citation](#citation)**

Code of the paper **"Exact power indices for plurality-voting ensembles"** (Sarpe, Georgakopoulos and
Gionis, NeurIPS 2026).

Plurality-voting ensembles aggregate the predictions of multiple base classifiers casting hard votes and output the most
voted class, with ties broken uniformly at random. ENPOWER is a collection of algorithms that compute
the **exact** [Banzhaf](https://en.wikipedia.org/wiki/Banzhaf_power_index) and
[Shapley](https://en.wikipedia.org/wiki/Shapley_value) values of every base classifier, for characteristic function games
modeling plurality voting with random tie-breaking. The algorithms require
- linear time for binary classification, from closed-form solutions (ENPOWER<sub>B</sub>);
- low-degree polynomial time for multiclass classification, through ordinary generating functions
  computed with dynamic programming (ENPOWER<sub>MC</sub>-β, ENPOWER<sub>MC</sub>-φ and
  ENPOWER<sub>MC</sub>-φ++).

This repository contains the C++ implementation of ENPOWER and all the code of the experiments of the
paper.

## Citation

```bibtex
@article{sarpe2026enpower,
  title   = {Exact power indices for plurality-voting ensembles},
  author  = {Sarpe, Ilie and Georgakopoulos, Theofanis and Gionis, Aristides},
  journal = {arXiv preprint arXiv:2610.17980},
  year    = {2026}
}
```

## Simple example

```sh
make -C cpp
cpp/solver banzhaf --input cpp/examples/toy.txt
```

`cpp/examples/toy.txt` holds the votes of 5 base classifiers on 8 data points with 3 classes. The
command prints the Banzhaf values of the 5 base classifiers in the game over these 8 data points:

```json
{
  "mode": "banzhaf", "points": 8, "models": 5, "labels": 3, "profiles": 2,
  "game": "multiclass", "coefficients": "exact", "seconds": 0.0001,
  "values": [0.1927, 0.1536, 0.1328, 0.1536, 0.1328]
}
```

## Methods included

$n$ is the number of base classifiers, $\ell$ the number of classes, $|D|$ the number of data points
of the test set and $|U|$ the number of its unique vote profiles.

| Method | Index | Setting | Time | Command |
|---|---|---|---|---|
| ENPOWER<sub>B</sub> | Banzhaf, Shapley | binary | $O(\vert D \vert n)$ | `solver banzhaf --binary`, `solver shapley --binary` |
| ENPOWER<sub>MC</sub>-β | Banzhaf | multiclass | $O(\vert U \vert n \ell^2 + \vert D \vert n)$ | `solver banzhaf` |
| ENPOWER<sub>MC</sub>-φ | Shapley | multiclass | $O(\vert U \vert n^3 \ell + \vert D \vert n)$ | `solver shapley --baseline` |
| ENPOWER<sub>MC</sub>-φ++ | Shapley | multiclass | never worse than ENPOWER<sub>MC</sub>-φ (Appendix H) | `solver shapley --refined` |

## Installation

The C++ library uses only the standard library. Build it with any C++17 compiler:

```sh
make -C cpp            # builds cpp/solver
make -C cpp check      # builds cpp/check
cd cpp && ./check      # tests the algorithms against brute force
```

The experiments also require Python and LaTeX. Both are provided by the Apptainer image defined in
`container/`, which is built by `./reproduce.sh container` (see [REPRODUCE.md](REPRODUCE.md)).

## Usage

`cpp/solver` reads the votes of the base classifiers on a test set from a text file:
- a header line `points models labels`: the number of data points, of base classifiers and of classes;
- one line per data point: its true label, then the label predicted by each base classifier.

Labels are integers in `0 … labels-1`.

```
8 5 3
0 0 0 1 0 2
1 1 1 1 0 1
...
```

From a fitted scikit-learn ensemble, with labels encoded as integers:

```python
import numpy as np
preds = np.column_stack([est.predict(X) for est in ensemble.estimators_])   # data points x base classifiers
with open("votes.txt", "w") as f:
    f.write(f"{len(y)} {preds.shape[1]} {int(max(preds.max(), y.max())) + 1}\n")
    for label, row in zip(y, preds.astype(int)):
        f.write(f"{label} {' '.join(map(str, row))}\n")
```

Then compute the Banzhaf or Shapley values of every base classifier in the game over the test set:

```sh
cpp/solver banzhaf --input votes.txt              # ENPOWER_MC-β
cpp/solver shapley --input votes.txt --refined    # ENPOWER_MC-φ++
cpp/solver shapley --input votes.txt --binary     # ENPOWER_B (binary classification only)
```

Each command prints one JSON object. Its `values` field holds one value per base classifier, in the
order of the columns of the input, and `profiles` is the number of unique vote profiles.
`shapley` reports the values of each algorithm it runs under its own key: `baseline` for
ENPOWER<sub>MC</sub>-φ, `refined` for ENPOWER<sub>MC</sub>-φ++ and `binary` for ENPOWER<sub>B</sub>.
Without flags it runs both ENPOWER<sub>MC</sub>-φ and ENPOWER<sub>MC</sub>-φ++. `--binary` computes
the values of the binary classification game, so use it only on data with two classes; there it
returns the same values as the multiclass algorithms. The comment at the top of `cpp/main.cpp` lists
every mode and flag. `python cpp/tools/enpower.py dataset` converts a stored `*_predictions.npz` to
this format.

## Reproducing the paper

```sh
./reproduce.sh all
```

This builds the image, downloads the datasets, runs every experiment and writes all figures and
tables of the paper to `results/`. [REPRODUCE.md](REPRODUCE.md) explains each step, which command
produces which figure or table, and how to run on a SLURM cluster. Implementation details are in
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## License

MIT, see [LICENSE](LICENSE). `python/shapley/` corresponds to the library of Rozemberczki et al., used for
the Troupe baselines, under its own MIT license.
