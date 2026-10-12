# Implementation of the C++ library

**[Back to the README](../README.md)** | **[Reproducing the paper](../REPRODUCE.md)**

This document describes the C++ implementation of ENPOWER. The notation follows the paper: n is the
number of base classifiers, ℓ the number of classes, n<sub>1</sub> the number of base classifiers
that predict the correct class of a data point, and N = n − n<sub>1</sub>. In the code, ℓ is `L`.

```
cpp/libs/utils.h            numerics: binomial coefficients (exact and in log scale), binomial pmf,
                            the wide integers WideUInt<W> and their exact division
cpp/libs/gameSolver.h       Solver<T>: the entry points over a dataset
cpp/libs/gameSolver.tpp     one entry point per (power index, game)
cpp/libs/shapleyMC.{h,tpp}  ENPOWER_MC-φ and ENPOWER_MC-φ++ on one vote profile
cpp/libs/banzhafMC.{h,tpp}  ENPOWER_MC-β on one vote profile
cpp/libs/shapleyBinary.h    the closed form of the Shapley values of ENPOWER_B
cpp/libs/baselines.h        the baselines WMG and ISCG
cpp/libs/sampling.h         the baselines MC-β and MC-φ
cpp/main.cpp                cpp/solver: one mode per computation, JSON output on stdout
cpp/check.cpp               cpp/check: tests of all algorithms against brute force, and a benchmark
```

The library implements both power indices for both games:

|         | Binary classification (ENPOWER<sub>B</sub>) | Multiclass classification |
|---|---|---|
| Banzhaf | `BanzhafBinary_`, closed form | `BanzhafMC_`, ENPOWER<sub>MC</sub>-β |
| Shapley | `ShapleyBinary_`, closed form | `ShapleyMC_`, ENPOWER<sub>MC</sub>-φ and ENPOWER<sub>MC</sub>-φ++ |

On data with two classes, the binary and the multiclass algorithms return the same values;
`cpp/check` tests this.

The power indices are computed once per unique *vote profile*: the number n<sub>1</sub> of votes for
the correct class, together with the sorted numbers of votes of the incorrect classes. By symmetry,
the values of the base classifiers on a data point depend only on its vote profile. The values of a
vote profile are therefore computed once and reused for every data point with that profile.

## Multiclass Shapley values

`Solver<T>::getShapleyMC(refined, stats)` computes the exact Shapley values of the base classifiers
in the multiclass game over the whole dataset.

* `refined = false` is ENPOWER<sub>MC</sub>-φ. The matrix of the coefficients of `P_h(t,u)` has ℓ
  rows and N + 1 columns, and the loops over `k` and `m` run up to ℓ − 1 and N.
* `refined = true` is ENPOWER<sub>MC</sub>-φ++ (Appendix H of the paper), with the following bounds:
  - the matrix has `K(h) + 1` rows and `Phi(h-1) + 1` columns;
  - the loop over `h` stops at `H^* = min{n_1, n_(1)}`;
  - the loop over `k` stops at `K(h)`, or at `K_{-j}(h)` after `RemoveClass`;
  - the loops over `m` stop at `min{mbar(k,h), Phi(h)}` and `min{mbar_{-j}(k,h), Phi(h)}`.

  `K` and `Phi` are computed in a single pass over the vote counts, in `O(L + H^*)` time and without
  sorting.

  Every coefficient of `P_h(t,u)` with `m < kh` is zero, so row `k` is stored from its first column
  that may be non-zero: `P(k,m)` holds `[t^k u^{m+kh}] P_h(t,u)`. Since `Phi(h) = Phi(h-1) + K(h)`,
  this removes `K(h)` columns from every row, and it removes the factor `u^h` from `AddClass` and
  `RemoveClass`, which receive the offset as the argument `tieShift`.

## Multiclass Banzhaf values

`Solver<T>::getBanzhaf(false)` computes the exact Banzhaf values of the base classifiers in the
multiclass game (ENPOWER<sub>MC</sub>-β). The computation has the structure of the Shapley
computation, with every coalition weighted by `2^-(n-1)`. There is one iteration for each
`h in [1, H^*]`, over the polynomial

    P_h(t) = prod_{j >= 2} (A_j(h) + B_j(h) t),
    A_j(h) = sum_{i < h} C(n_j, i),   B_j(h) = C(n_j, h),

whose coefficient of `t^k` is the number of coalitions in which exactly `k` incorrect classes tie with
the correct class at `h` votes. The value of a base classifier that predicts the correct class is
computed from the coefficients of `P_h`. The value of a base classifier that predicts an incorrect
class with `n_j` votes is computed from the coefficients of `P_h / (A_j(h) + B_j(h) t)`. Since the
polynomial has one variable and there is no loop over `m`, a vote profile requires `O(n_1 L^2)` time,
instead of the `O(L N^3)` time of ENPOWER<sub>MC</sub>-φ.

## Binary classification

`Solver<T>::getShapleyBinary()` computes the Shapley values of ENPOWER<sub>B</sub> in closed form. In
the binary game, a vote profile is determined by the number `n_+` of correct votes. The value is a
single sum over the number of votes at which the two classes tie, and each term is obtained from the
previous one by a ratio, so each unique `n_+` requires `O(n)` time. `getBanzhaf(true)` computes the
Banzhaf values of ENPOWER<sub>B</sub> from the binomial pmf of the ties.

## Exact arithmetic

The coefficients of the two dynamic programs count coalitions. They are therefore integers of up to N
bits, and `RemoveClass` divides one polynomial by another. In floating-point arithmetic, each
division amplifies the error of its input, and the error compounds over the iterations of the loop
over `h`. For this reason, the coefficients are computed with exact integer arithmetic.

`shapmc::payoutsFor` and `bzmc::payoutsFor` select the narrowest exact type for each vote profile:
- `__int128` for N ≤ 126;
- `utilities::WideUInt<4>` (256 bits) for N ≤ 255;
- `utilities::WideUInt<8>` (512 bits) for N ≤ 511;
- `utilities::WideUInt<16>` (1024 bits) for N ≤ 1023.

Since the Banzhaf computation divides by a polynomial, the wide integers also require an exact
division. `utilities::divExact` factors the divisor into `2^s` and an odd factor, and multiplies by
the inverse of the odd factor modulo `2^(64 W)`, computed with three steps of Newton's iteration.

Only the final weighted sum is computed in floating-point arithmetic (`long double`, with the weights
computed through logarithms), so the rounding error does not compound. The flag `--float` computes
the coefficients in `long double` instead, for comparison with the exact computation.

With exact coefficients, `AddClass` skips the entries of `P_h` that are zero. In floating-point
arithmetic these entries are not exactly zero and cannot be skipped.

## Tests

`cpp/check` tests, with exact coefficients, that:

1. the identities on `K`, `Phi`, `N^*` and `H^*` hold;
2. every coefficient outside `[0, K(h)] x [0, Phi(h)]` is zero, the matrix of ENPOWER<sub>MC</sub>-φ++
   agrees with the matrix of ENPOWER<sub>MC</sub>-φ inside it, and the corner `(K(h), Phi(h))` is
   non-zero;
3. no term dropped by `mbar(k,h)`, `mbar_{-j}(k,h)`, `K(h)` or `K_{-j}(h)` is non-zero, on `P_h` and
   on its quotient by `A_j(h;u) + B_j(h;u) t`;
4. no coefficient with `k >= 1` is non-zero after an iteration `h > H^*`;
5. ENPOWER<sub>MC</sub>-φ, ENPOWER<sub>MC</sub>-φ++ and a brute-force computation of the definition
   of the Shapley value agree;
6. `WideUInt<4>` and `WideUInt<8>` agree with `__int128` on every vote profile;
7. the Banzhaf values of ENPOWER<sub>MC</sub>-β agree with a brute-force computation of the
   definition, and their wide coefficients agree with the 128-bit ones;
8. the closed form of the Shapley values of ENPOWER<sub>B</sub> agrees with ENPOWER<sub>MC</sub>-φ on
   the two-class vote profile `(n_1, [n - n_1])`;
9. the baseline WMG agrees with its closed forms and with a brute-force computation of its
   definition, and the baseline ISCG agrees with a brute-force computation of its definition.

`cpp/check` tests every vote profile up to `--nmax` base classifiers and `--Lmax` classes (defaults 8
and 5), then `--trials` random vote profiles up to `--rnmax` base classifiers and `--rLmax` classes
(defaults 400, 14 and 6), then every binary vote profile up to `--bnmax` base classifiers (default
120). The exit code is 0 if and only if all tests pass.

`cpp/check --bench [--benchN n --benchL L]` reports, for each vote profile, the number of
multiply-accumulate operations of ENPOWER<sub>MC</sub>-φ and of ENPOWER<sub>MC</sub>-φ++, together
with the factor `L (n/N^*)^3` of the analysis in Appendix H.

## Limitations

* The exact coefficients support N ≤ 1023, the width of `shapmc::Widest`; `payoutsFor` throws an
  exception for larger N. A larger `W` in `WideUInt<W>` increases this limit; the two functions
  `payoutsFor` are the only places that select the type.
* `utilities::binom_exact` saturates beyond `2^127` and reports the saturation. It is used only by the
  128-bit computation, which `payoutsFor` selects only for N ≤ 126.
* ENPOWER<sub>MC</sub>-β has no counterpart of ENPOWER<sub>MC</sub>-φ++. Of the bounds of
  ENPOWER<sub>MC</sub>-φ++, only the restriction to `h ≤ H^*` applies to a polynomial in one variable.
* `solver shapley --binary` and `solver banzhaf --binary` compute the values of the binary
  classification game and must be used only on data with two classes. On data with more classes,
  their values differ from those of the multiclass game.
