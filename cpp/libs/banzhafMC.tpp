#include <cmath>
#include <limits>

namespace bzmc {

namespace detail {
   using shapmc::detail::fromLog;
   using shapmc::detail::logC;

   inline long double logHalves(int n) {           // log 2^-(n-1), the Banzhaf weight
      return -static_cast<long double>(n - 1) * std::log(2.0L);
   }
}

// ---------------------------------------------------------------------------
// The two weights of Thm 4.6, the Banzhaf counterparts of f_1 and f_j. Every
// coalition has the same weight 2^-(n-1), so the weights count coalitions: for
// a tie at h votes, C(n_1 - 1, h) and C(n_1 - 1, h - 1) for a classifier
// predicting the correct label, and C(c - 1, h) and C(c - 1, h - 1), times the
// C(n_1, h) coalitions in which the correct label has h votes, for a classifier
// predicting an incorrect label with c votes. The factor 2^-(n-1) is computed
// in log scale, so the weights remain in the range of long double for every n.
// ---------------------------------------------------------------------------
inline Real gCorrect(int k, int h, const Profile& p)
{
   if (p.M1 <= 0) return 0.0L;
   const long double base = detail::logHalves(p.n);
   const Real t1 = detail::fromLog(detail::logC(p.M1 - 1, h)     + base);
   const Real t2 = detail::fromLog(detail::logC(p.M1 - 1, h - 1) + base);
   return (static_cast<Real>(k) * t1 + t2) / static_cast<Real>(k + 1);
}

inline Real gIncorrect(int k, int h, int c, const Profile& p)
{
   if (c <= 0 || p.M1 <= 0) return 0.0L;
   const long double base = detail::logHalves(p.n) + detail::logC(p.M1, h);
   const Real t1 = detail::fromLog(detail::logC(c - 1, h)     + base);
   const Real t2 = detail::fromLog(detail::logC(c - 1, h - 1) + base);
   return (static_cast<Real>(k + 1) * t1 + t2)
        / (static_cast<Real>(k + 1) * static_cast<Real>(k + 2));
}

// ---------------------------------------------------------------------------
// The Banzhaf values of one vote profile
// ---------------------------------------------------------------------------
template <typename C>
Payouts payoutsDP(const Profile& p)
{
   Payouts out;
   out.correct = detail::fromLog(detail::logHalves(p.n));  // the marginal contribution to the empty coalition
   for (int c : p.inc) out.incorrect[c] = 0;
   if (p.M1 <= 0) return out;             // no correct classifier: every marginal contribution is zero

   // the distinct numbers of votes of the incorrect labels, with multiplicity:
   // labels with the same number of votes have the same value, by symmetry
   std::vector<int> vals, mult;
   for (int c : p.inc) {
      if (!vals.empty() && vals.back() == c) ++mult.back();
      else { vals.push_back(c); mult.push_back(1); }
   }
   const int La = static_cast<int>(p.inc.size());
   const int D  = static_cast<int>(vals.size());

   // A_j(h) = sum_{i < h} C(n_j, i) and B_j(h) = C(n_j, h), one per distinct count
   std::vector<C> A(D, C(1)), B(D, C(0));
   for (int d = 0; d < D; ++d) B[d] = shapmc::CoefTraits<C>::binom(vals[d], 1);

   std::vector<C> P(La + 1, C(0)), Q(La, C(0));

   // for h > H^* = min{n_1, n_(1)} no incorrect label has h votes, both
   // C(n_j - 1, h) and C(n_j - 1, h - 1) are zero, and so is every term
   const int Hstar = std::min(p.M1, p.maxInc());

   for (int h = 1; h <= Hstar; ++h) {
      // P_h(t) = prod_j (A_j + B_j t), one factor per label
      std::fill(P.begin(), P.end(), C(0));
      P[0] = C(1);
      int deg = 0;
      for (int d = 0; d < D; ++d) {
         for (int r = 0; r < mult[d]; ++r) {
            for (int k = deg + 1; k >= 1; --k) P[k] = P[k] * A[d] + P[k - 1] * B[d];
            P[0] = P[0] * A[d];
            ++deg;
         }
      }

      // a correct classifier added to a coalition in which k incorrect labels tie
      // with the correct label
      for (int k = 1; k <= La; ++k) {
         if (P[k] == C(0)) continue;
         out.correct += gCorrect(k, h, p) * static_cast<Real>(P[k]);
      }

      // the value of a classifier predicting an incorrect label is computed
      // from the quotient of P_h by the factor of its label, below
      for (int d = 0; d < D; ++d) {
         if (vals[d] < h) continue;      // C(n_j - 1, h) and C(n_j - 1, h - 1) are both zero
         Q[0] = Arith<C>::div(P[0], A[d]);
         for (int k = 1; k < La; ++k)
            Q[k] = Arith<C>::div(P[k] - B[d] * Q[k - 1], A[d]);

         Real v = 0;
         for (int k = 0; k < La; ++k) {
            if (Q[k] == C(0)) continue;
            v -= gIncorrect(k, h, vals[d], p) * static_cast<Real>(Q[k]);
         }
         out.incorrect[vals[d]] += v;
      }

      for (int d = 0; d < D; ++d) {      // A_j(h + 1) = A_j(h) + C(n_j, h)
         A[d] = A[d] + B[d];
         B[d] = shapmc::CoefTraits<C>::binom(vals[d], h + 1);
      }
   }
   return out;
}

// ---------------------------------------------------------------------------
// The definition of the Banzhaf value, for testing
// ---------------------------------------------------------------------------
inline Payouts payoutsBruteForce(const Profile& p)
{
   std::vector<int> cnt;
   cnt.push_back(p.M1);
   for (int c : p.inc) cnt.push_back(c);
   const int T = static_cast<int>(cnt.size());

   // v(S) of Eq. (3), from the votes of the coalition S for each label
   auto value = [&](const std::vector<int>& s, int extra) -> Real {
      int mx = -1, ties = 0, tot = 0;
      for (int q = 0; q < T; ++q) {
         const int vq = s[q] + (q == extra ? 1 : 0);
         tot += vq;
         if (vq > mx)       { mx = vq; ties = 1; }
         else if (vq == mx) { ++ties; }
      }
      if (tot == 0) return 0.0L;                   // v(empty) = 0
      const int v0 = s[0] + (extra == 0 ? 1 : 0);
      if (v0 != mx) return 0.0L;                   // the correct label does not receive the most votes
      return Real(1) / static_cast<Real>(ties);
   };

   Payouts out;
   for (int t = 0; t < T; ++t) {
      if (cnt[t] == 0) continue;
      std::vector<int> lim(T), s(T, 0);
      for (int q = 0; q < T; ++q) lim[q] = cnt[q] - (q == t ? 1 : 0);

      Real acc = 0;
      while (true) {
         const Real dv = value(s, t) - value(s, -1);
         if (dv != 0) {
            // the coalitions with these votes, all of the same weight
            long double l = detail::logHalves(p.n);
            for (int q = 0; q < T; ++q) l += detail::logC(lim[q], s[q]);
            acc += dv * detail::fromLog(l);
         }
         int q = 0;
         while (q < T && s[q] == lim[q]) { s[q] = 0; ++q; }
         if (q == T) break;
         ++s[q];
      }
      if (t == 0) out.correct = acc;
      else        out.incorrect[cnt[t]] = acc;
   }
   return out;
}

// ---------------------------------------------------------------------------
// Selection of the coefficient type, as in shapmc::payoutsFor
// ---------------------------------------------------------------------------
inline Payouts payoutsFor(const Profile& p, Coefficients coef)
{
   if (coef == Coefficients::Float) return payoutsDP<Real>(p);
   if (p.N() <= 126)                return payoutsDP<__int128>(p);
   if (p.N() <= 255)                return payoutsDP<shapmc::Wide>(p);
   if (p.N() <= 511)                return payoutsDP<shapmc::Wider>(p);
   if (p.N() > shapmc::kMaxExactN)
      throw std::overflow_error("bzmc::payoutsFor: N = " + std::to_string(p.N())
                                + " exceeds the exact width of " + std::to_string(shapmc::kMaxExactN));
   return payoutsDP<shapmc::Widest>(p);
}

} // namespace bzmc
