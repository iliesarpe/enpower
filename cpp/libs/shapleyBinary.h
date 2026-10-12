#ifndef SHAPLEY_BINARY_H
#define SHAPLEY_BINARY_H

#include <cmath>
#include "utils.h"

/**
* Exact Shapley values of the base classifiers of a plurality-voting ensemble,
* binary classification: the closed form of ENPOWER_B (Thm 4.2 of the paper).
* get_shapley and compute_shap_summation of python/indices_binary.py are the
* reference implementation in Python.
*
* The binary game of Eq. (2) is the multiclass game of Eq. (3) with \ell = 2:
* the value of a coalition is 1 when its correct classifiers outnumber its
* incorrect ones, 1/2 when the two numbers are equal, and 0 otherwise; the
* value of the empty coalition is 0. The value of a classifier therefore
* depends on the vote profile of a data point only through the number n_+ of
* correct classifiers. The two values of a vote profile are a single sum over
* h, the number of votes at which the two labels tie, in which each term is
* obtained from the previous one by a ratio.
*
* The computation over a dataset requires O(n) time per distinct n_+ (Thm 4.3).
*/
namespace shapbin {

using Real = long double;

// The two Shapley values of a vote profile: the value of a correct classifier,
// and the absolute value of the (negative) value of an incorrect classifier
struct Payouts {
   Real correct   = 0;
   Real incorrect = 0;
};

namespace detail {

   // Beta(n - 2x + 1, 2x) C(y, x), the first term of the sum when the two
   // labels do not tie in the whole ensemble. It is 0 for x = 0, in which case
   // the sum is empty.
   inline Real openingTerm(int n, int x, int y)
   {
      if (x <= 0) return 0.0L;
      const long double lb = std::lgamma(static_cast<long double>(n - 2 * x) + 1.0L)
                           + std::lgamma(static_cast<long double>(2 * x))
                           - std::lgamma(static_cast<long double>(n) + 1.0L)
                           + utilities::log_binom(y, x);
      return std::exp(lb);
   }

   /**
   * The sum of compute_shap_summation over the coalitions in which the two
   * labels tie at h votes each, for h in [1, b], with first term a. The term of
   * h + 1 is the term of h times
   *
   *     2 (n1 - h)(n2 - h)(2h + 1) / ((h + 1)(n - 2h - 1)(n - 2h - 2)),
   *
   * so the sum requires one iteration per h. c is the contribution of the
   * coalitions in which the correct label already has the most votes without
   * the classifier; it applies only to a correct classifier.
   */
   inline Real summation(int n, Real a, int b, int n1, int n2, bool isCorrect)
   {
      if (n1 == 0) return 0.0L;
      Real acc = 0;
      if (b > 0 && n > 2) {
         Real weight  = static_cast<Real>(3 * n1 + n2 - 4);
         Real current = static_cast<Real>(n2)
                      / (static_cast<Real>(n) * (n - 1) * (n - 2));
         acc = current * weight;
         for (int h = 1; h < b; ++h) {
            const Real den = static_cast<Real>(h + 1)
                           * static_cast<Real>(n - 2 * h - 1)
                           * static_cast<Real>(n - 2 * h - 2);
            if (den == 0) break;              // h is beyond the range of the sum
            const Real num = 2.0L * static_cast<Real>(n1 - h)
                                  * static_cast<Real>(n2 - h)
                                  * static_cast<Real>(2 * h + 1);
            current *= num / den;
            weight  -= 4.0L;
            acc     += current * weight;
         }
      }
      const Real c = isCorrect ? Real(1) / static_cast<Real>(n) : Real(0);
      return c + 0.5L * (a + acc);
   }

} // namespace detail

/**
* The Shapley values of the vote profile of a data point, for an ensemble of n
* classifiers of which nPlus are correct on the data point.
*/
inline Payouts payoutsFor(int n, int nPlus)
{
   const int nMinus = n - nPlus;

   Real aCorrect = 0, aWrong = 0;
   int  bCorrect = 0, bWrong = 0;

   if (nMinus < nPlus) {                 // the correct label has the most votes
      bCorrect = nMinus;
      aWrong   = detail::openingTerm(n, nMinus, nPlus);
      bWrong   = nMinus - 1;
   } else if (nMinus > nPlus) {
      aCorrect = detail::openingTerm(n, nPlus, nMinus);
      bCorrect = nPlus - 1;
      bWrong   = nPlus;
   } else {                              // the two labels tie in the whole ensemble, v(N) = 1/2
      aCorrect = Real(1) / static_cast<Real>(n);
      aWrong   = aCorrect;
      bCorrect = nPlus - 1;
      bWrong   = nMinus - 1;
   }

   Payouts out;
   out.correct   = detail::summation(n, aCorrect, bCorrect, nPlus, nMinus, true);
   out.incorrect = detail::summation(n, aWrong,   bWrong,   nMinus, nPlus, false);
   return out;
}

} // namespace shapbin

#endif //SHAPLEY_BINARY_H
