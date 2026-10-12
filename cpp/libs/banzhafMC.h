#ifndef BANZHAF_MC_H
#define BANZHAF_MC_H

#include <map>
#include <vector>
#include "shapleyMC.h"

/**
* Exact Banzhaf values of the base classifiers of a plurality-voting ensemble,
* multiclass classification: ENPOWER_MC-beta (Thm 4.6 and Alg. 3 of the paper).
* python/banzhaf_multiclass.py is the reference implementation in Python.
*
* As for the Shapley values, the values depend only on the vote profile of a
* data point. The loop over h in [1, n_1] ranges over the number of votes at
* which the correct label ties, and a dynamic program computes the polynomial
*
*     P_h(t) = prod_{j >= 2} (A_j(h) + B_j(h) t),
*     A_j(h) = sum_{i < h} C(n_j, i),   B_j(h) = C(n_j, h),
*
* whose coefficient of t^k is the number of coalitions in which exactly k
* incorrect labels tie with the correct label at h votes. The value of a
* classifier predicting the correct label is computed from the coefficients of
* P_h, and the value of a classifier predicting an incorrect label with n_j
* votes from the coefficients of P_h / (A_j(h) + B_j(h) t), the counterpart of
* RemoveClass.
*
* The coefficients of P_h are numbers of coalitions, bounded by 2^N with
* N = n - n_1, and are computed as exact integers of the narrowest type that
* represents them. Only the weights of the two sums are computed in floating
* point, through logarithms, as in shapmc.
*
* The division by A_j(h) + B_j(h) t involves the L coefficients of a polynomial
* in one variable, instead of the (k, m) matrix of the Shapley computation, so
* its floating-point error is smaller. The coefficients are nevertheless exact,
* as in shapmc; Coefficients::Float is used only for comparison.
*/
namespace bzmc {

using shapmc::Coefficients;
using shapmc::Profile;
using shapmc::Real;

// The Banzhaf values of one vote profile: the value of a classifier predicting
// the correct label, and the value of a classifier predicting an incorrect
// label with c votes, for each c
struct Payouts {
   Real correct = 0;
   std::map<int, Real> incorrect;
};

// ---------------------------------------------------------------------------
// The exact division by A_j(h) + B_j(h) t, for each coefficient type. The wide
// integers implement only exact division, since this is the only division the
// library requires and its quotient is always an integer.
// ---------------------------------------------------------------------------
template <typename C>
struct Arith {
   static C div(const C& a, const C& b) { return a / b; }
};

template <int W>
struct Arith<utilities::WideUInt<W>> {
   static utilities::WideUInt<W> div(const utilities::WideUInt<W>& a,
                                     const utilities::WideUInt<W>& b) {
      return utilities::divExact(a, b);
   }
};

// The weights of [t^k] P_h(t) in the values of a correct and of an incorrect
// classifier (Thm 4.6), the counterparts of f_1(k,h,m) and f_j(k,h,m) of Thm 4.8
Real gCorrect  (int k, int h, const Profile& p);
Real gIncorrect(int k, int h, int c, const Profile& p);

// The Banzhaf values of one vote profile, with the coefficients of P_h of type C
template <typename C> Payouts payoutsDP(const Profile& p);

// The Banzhaf values of one vote profile, computed with the narrowest integer
// type that represents its coefficients exactly, as in shapmc::payoutsFor
Payouts payoutsFor(const Profile& p, Coefficients coef = Coefficients::Exact);

// The same values from the definition of the Banzhaf value, for testing:
// beta_i = 2^-(n-1) sum_{S subset N - i} v(S + i) - v(S), with the coalitions S
// enumerated by their vote profile in prod_j (n_j + 1) steps.
Payouts payoutsBruteForce(const Profile& p);

} // namespace bzmc

#include "banzhafMC.tpp"

#endif //BANZHAF_MC_H
