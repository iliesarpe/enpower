#ifndef SHAPLEY_MC_H
#define SHAPLEY_MC_H

#include <vector>
#include <map>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "utils.h"

/**
* Exact Shapley values of the base classifiers of a plurality-voting ensemble,
* multiclass classification.
*
* Two implementations of the identities of Thm 4.8 of the paper:
*
*   baseline: ENPOWER_MC-phi (Alg. 4). The matrix of the coefficients of
*             P_h(t,u) has \ell rows and n - n_1 + 1 columns, and the loop over h
*             runs up to n_1.
*   refined : ENPOWER_MC-phi++ (Appendix H). The matrix has K(h) + 1 rows and
*             Phi(h) + 1 columns, the loop over h runs up to H^*, and the two
*             loops over m stop at \bar m(k,h) and \bar m_{-j}(k,h).
*
* Both take as input the vote profile of one data point and do not enumerate
* coalitions.
*/
namespace shapmc {

using Real = long double;

// ---------------------------------------------------------------------------
// Vote profile of one data point
// ---------------------------------------------------------------------------
struct Profile {
   int n  = 0;              // n      : classifiers of the ensemble
   int L  = 0;              // \ell   : labels of the task
   int M1 = 0;              // n_1    : classifiers predicting the correct label
   std::vector<int> inc;    // n_j, j in [2, \ell]: zero counts dropped, sorted

   int N() const {          // N = n - n_1 : classifiers predicting an incorrect label
      int s = 0;
      for (int c : inc) s += c;
      return s;
   }
   int maxInc() const {     // n_(1) : largest number of votes of an incorrect label
      int mx = 0;
      for (int c : inc) mx = std::max(mx, c);
      return mx;
   }
};

// ---------------------------------------------------------------------------
// K(h), Phi(h), H^* and N^* of Def. H.1, computed by bucket sort of the counts
// ---------------------------------------------------------------------------
struct ProfileFns {
   int  Hstar = 0;                 // H^* = min{n_1, n_(1)}
   long long Nstar = 0;            // N^* = Phi(H^*)
   std::vector<int> K;             // K[h]   for h in [0, H^*], K[0] unused
   std::vector<long long> Phi;     // Phi[h] for h in [0, H^*], Phi[0] = 0

   // K_{-j}(h) = K(h) - 1[n_j >= h], for the label of c votes
   int Kminus(int h, int c) const { return K[h] - (c >= h ? 1 : 0); }
};

ProfileFns profileFunctions(const Profile& p);

// \bar m(k,h) and \bar m_{-j}(k,h) of Appendix H.1. Both use the bound Phi(h-1)
// on the votes of the labels that do not tie, which is tighter than (L-1)(h-1)
// unless every incorrect label has at least h - 1 votes.
inline long long mBar(int k, int h, const Profile& p, const ProfileFns& f) {
   return std::min<long long>(p.N(), static_cast<long long>(k) + f.Phi[h - 1]);
}
inline long long mBarMinus(int k, int h, int c, const Profile& p, const ProfileFns& f) {
   long long spread = static_cast<long long>(k) + f.Phi[h - 1] - std::min(c, h - 1);
   return std::min<long long>(p.N() - c, spread);
}

// ---------------------------------------------------------------------------
// Instrumentation, shared by the two implementations
// ---------------------------------------------------------------------------
struct Stats {
   long long entriesVisited = 0;   // accesses to the entries of P_h
   long long ops            = 0;   // multiply-accumulate operations, the cost model of Thm H.7
   long long termsSummed    = 0;   // terms of the two identities of Thm 4.8
   long long matrixEntries  = 0;   // entries of P_h, summed over the loop over h
   long long iterationsH    = 0;   // iterations of the loop over h
};

// The Shapley values of one vote profile: the value of a classifier predicting
// the correct label, and the value of a classifier predicting an incorrect
// label with c votes, for each c
struct Payouts {
   Real correct = 0;
   std::map<int, Real> incorrect;
};

// ---------------------------------------------------------------------------
// The coefficients of P_h(t,u) count coalitions, so they are integers of up to
// N = n - n_1 bits, and RemoveClass divides one polynomial by another. In
// floating-point arithmetic the error of each division amplifies the error of
// its input, and it compounds over the loop over h. The coefficients are
// therefore exact integers: __int128 for N <= 126, and a wide integer type for
// larger N. The floating-point type is used only for comparison (--float).
// ---------------------------------------------------------------------------
template <typename C>
struct CoefTraits {
   static C binom(int a, int b) { return static_cast<C>(utilities::binom(a, b)); }
};

template <>
struct CoefTraits<__int128> {
   static __int128 binom(int a, int b) { return utilities::binom_exact(a, b); }
};

template <int W>
struct CoefTraits<utilities::WideUInt<W>> {
   static const utilities::WideUInt<W>& binom(int a, int b) {
      return utilities::binom_wide<W>(a, b);
   }
};

// The coefficients are bounded by 2^N, so 64 W bits represent them exactly for
// N <= 64 W - 1, that is, N <= 255, 511 and 1023.
using Wide    = utilities::WideUInt<4>;
using Wider   = utilities::WideUInt<8>;
using Widest  = utilities::WideUInt<16>;
constexpr int kMaxExactN = 1023;   // for larger N the coefficients would overflow

// Arithmetic of the coefficients, as chosen by the dataset entry point
enum class Coefficients { Exact, Float };

// The Shapley values of one vote profile, computed with the narrowest integer
// type that represents its coefficients exactly (128 bits for N <= 126). This
// is the only function that selects the algorithm and the coefficient type.
Payouts payoutsFor(const Profile& p, bool refined, Stats* st, Coefficients coef);

// ---------------------------------------------------------------------------
// The matrix of the coefficients of P_h(t,u): P(k,m) = [t^k u^m] P_h(t,u)
// ---------------------------------------------------------------------------
template <typename C>
struct PolyMatrix {
   int rows = 0, cols = 0;
   std::vector<C> a;

   PolyMatrix() = default;
   PolyMatrix(int r, int c) : rows(r), cols(c), a(static_cast<size_t>(r) * c, C(0)) {}

   C&       operator()(int k, int m)       { return a[static_cast<size_t>(k) * cols + m]; }
   const C& operator()(int k, int m) const { return a[static_cast<size_t>(k) * cols + m]; }
};

// AddClass and RemoveClass of Alg. 6 and 7, on the sub-matrix [0,maxK] x [0,maxM].
// tieShift is the column offset of the term of a label tied at h votes. It is
// h on the matrix of the coefficients [t^k u^m] P_h(t,u), and 0 on the shifted
// matrix of ENPOWER_MC-phi++, whose row k starts at column m = kh, since the
// coefficients with m < kh are zero (Lemma H.2 (i)).
template <typename C> void addClass   (PolyMatrix<C>& P, int c, int h, int maxK, int maxM, int tieShift, Stats* st);
template <typename C> void removeClass(PolyMatrix<C>& P, int c, int h, int maxK, int maxM, int tieShift, Stats* st);

// ComputeCoeffsDP (Alg. 5) of ENPOWER_MC-phi, and its counterpart in
// ENPOWER_MC-phi++ with the smaller matrix
template <typename C> PolyMatrix<C> coeffsBaseline(const Profile& p, int h, Stats* st);
template <typename C> PolyMatrix<C> coeffsRefined (const Profile& p, int h, const ProfileFns& f, Stats* st);

// f_1(k,h,m) and f_j(k,h,m) of Thm 4.8
Real fCorrect  (int k, int h, int m, const Profile& p);
Real fIncorrect(int k, int h, int m, int c, const Profile& p);

// The Shapley values of one vote profile. Alg. 4 runs the loop over h up to n_1.
// With stopEarly, the loop stops once h exceeds the votes of every incorrect
// label, which is the restriction to H^* of Cor. H.4 (i); without it, the loop
// follows Alg. 4 exactly, as required to compare the two algorithms.
template <typename C> Payouts payoutsBaseline(const Profile& p, Stats* st, bool stopEarly = true);
template <typename C> Payouts payoutsRefined (const Profile& p, Stats* st);

// The same values from the definition of the Shapley value, for testing. The
// coalitions are enumerated by their vote profile, in prod_j (n_j + 1) steps.
Payouts payoutsBruteForce(const Profile& p);

// The vote profile of one data point, with the correct label first
template <typename T>
Profile makeProfile(const std::vector<T>& preds, T label, int labels);

} // namespace shapmc

#include "shapleyMC.tpp"

#endif //SHAPLEY_MC_H
