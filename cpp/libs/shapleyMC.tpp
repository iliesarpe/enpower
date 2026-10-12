#include <cmath>
#include <limits>

namespace shapmc {

namespace detail {
   inline long double logC(int a, int b) { return utilities::log_binom(a, b); }

   // log eta(s) = -log n - log C(n-1, s-1), and -infinity outside [1, n]
   inline long double logEta(int s, int n) {
      if (s < 1 || s > n) return -std::numeric_limits<long double>::infinity();
      return -std::log(static_cast<long double>(n)) - logC(n - 1, s - 1);
   }

   inline Real fromLog(long double l) {
      return (l == -std::numeric_limits<long double>::infinity()) ? 0.0L : std::exp(l);
   }
}

// ---------------------------------------------------------------------------
// Profile functions
// ---------------------------------------------------------------------------
inline ProfileFns profileFunctions(const Profile& p)
{
   ProfileFns f;
   f.Hstar = std::min(p.M1, p.maxInc());
   f.K.assign(f.Hstar + 1, 0);
   f.Phi.assign(f.Hstar + 1, 0);
   if (f.Hstar == 0) return f;

   // one scan of the counts fills the buckets and the tail count
   std::vector<int> bucket(f.Hstar, 0);
   int tail = 0;
   for (int c : p.inc) {
      if (c >= f.Hstar) ++tail;
      else              ++bucket[c];
   }
   // K(h) = c_>= + sum_{v=h}^{H*-1} c[v], by a scan of the buckets downwards
   int run = tail;
   for (int h = f.Hstar; h >= 1; --h) {
      if (h < f.Hstar) run += bucket[h];
      f.K[h] = run;
   }
   // Phi is the sequence of the partial sums of K, since Phi(0) = 0
   for (int h = 1; h <= f.Hstar; ++h) f.Phi[h] = f.Phi[h - 1] + f.K[h];
   f.Nstar = f.Phi[f.Hstar];
   return f;
}

// ---------------------------------------------------------------------------
// The two subroutines over the coefficient matrix
// ---------------------------------------------------------------------------
template <typename C>
void addClass(PolyMatrix<C>& P, int c, int h, int maxK, int maxM, int tieShift, Stats* st)
{
   const C bh = (c >= h) ? CoefTraits<C>::binom(c, h) : C(0);
   for (int k = maxK; k >= 0; --k) {
      for (int m = maxM; m >= 0; --m) {
         if (st) ++st->entriesVisited;
         const C v = P(k, m);
         if (v == C(0)) continue;
         if (c >= h) { P(k + 1, m + tieShift) += v * bh; if (st) ++st->ops; }  // the label is tied at h votes
         const int qmax = std::min(c, h - 1);
         for (int q = qmax; q >= 1; --q) {                  // the label receives q < h votes
            P(k, m + q) += v * CoefTraits<C>::binom(c, q);
            if (st) ++st->ops;
         }
      }
   }
}

template <typename C>
void removeClass(PolyMatrix<C>& P, int c, int h, int maxK, int maxM, int tieShift, Stats* st)
{
   const int qmax = std::min(c, h - 1);
   const C bh = (c >= h) ? CoefTraits<C>::binom(c, h) : C(0);
   for (int k = 0; k <= maxK; ++k) {
      for (int m = 0; m <= maxM; ++m) {
         if (st) ++st->entriesVisited;
         C v = P(k, m);
         if (c >= h && k >= 1 && m >= tieShift) { v -= P(k - 1, m - tieShift) * bh; if (st) ++st->ops; }
         for (int q = 1; q <= qmax; ++q)
            if (m >= q) { v -= P(k, m - q) * CoefTraits<C>::binom(c, q); if (st) ++st->ops; }
         P(k, m) = v;
      }
   }
}

// ---------------------------------------------------------------------------
// The coefficients of P_h(t,u)
// ---------------------------------------------------------------------------
template <typename C>
PolyMatrix<C> coeffsBaseline(const Profile& p, int h, Stats* st)
{
   PolyMatrix<C> P(p.L, p.N() + 1);
   P(0, 0) = C(1);
   int curK = 0, curM = 0;
   for (int c : p.inc) {
      addClass(P, c, h, curK, curM, h, st);
      ++curK;                       // one row per processed label
      curM += c;                    // every vote of the label is a new column
   }
   return P;
}

// The shifted matrix of ENPOWER_MC-phi++: row k holds the coefficients of
// P_h(t,u) from column kh onwards, that is, P(k,m) = [t^k u^{m+kh}] P_h(t,u).
// Every coefficient with m < kh is zero (Lemma H.2 (i)), so the matrix has
// K(h) fewer columns, the difference between Phi(h) and Phi(h-1), and the
// factor u^h of B_j(h;u) is removed.
template <typename C>
PolyMatrix<C> coeffsRefined(const Profile& p, int h, const ProfileFns& f, Stats* st)
{
   PolyMatrix<C> P(f.K[h] + 1, static_cast<int>(f.Phi[h - 1]) + 1);
   P(0, 0) = C(1);
   int curK = 0, curM = 0;
   for (int c : p.inc) {
      addClass(P, c, h, curK, curM, 0, st);
      if (c >= h) ++curK;           // only a label with at least h votes increases the degree in t
      curM += std::min(c, h - 1);   // after the shift, the degree in u grows by that of A_j(h;u)
   }
   return P;
}

// ---------------------------------------------------------------------------
// The two coefficients f_1 and f_j of Thm 4.8
// ---------------------------------------------------------------------------
inline Real fCorrect(int k, int h, int m, const Profile& p)
{
   if (p.M1 <= 0) return 0.0L;
   const Real t1 = detail::fromLog(detail::logC(p.M1 - 1, h)     + detail::logEta(m + h + 1, p.n));
   const Real t2 = detail::fromLog(detail::logC(p.M1 - 1, h - 1) + detail::logEta(m + h,     p.n));
   return (static_cast<Real>(k) * t1 + t2) / static_cast<Real>(k + 1);
}

inline Real fIncorrect(int k, int h, int m, int c, const Profile& p)
{
   if (c <= 0) return 0.0L;
   const long double lb = detail::logC(p.M1, h);
   const Real t1 = detail::fromLog(lb + detail::logC(c - 1, h)     + detail::logEta(m + 2 * h + 1, p.n));
   const Real t2 = detail::fromLog(lb + detail::logC(c - 1, h - 1) + detail::logEta(m + 2 * h,     p.n));
   return t1 / static_cast<Real>(k + 2)
        + t2 / (static_cast<Real>(k + 1) * static_cast<Real>(k + 2));
}

// ---------------------------------------------------------------------------
// The Shapley values of one vote profile, ENPOWER_MC-phi (Alg. 4)
// ---------------------------------------------------------------------------
template <typename C>
Payouts payoutsBaseline(const Profile& p, Stats* st, bool stopEarly)
{
   Payouts out;
   out.correct = Real(1) / static_cast<Real>(p.n);
   for (int c : p.inc) out.incorrect[c] = 0;

   const int N = p.N();
   const int maxInc = p.maxInc();

   for (int h = 1; h <= p.M1; ++h) {
      if (stopEarly && h > maxInc) break;     // no incorrect label reaches h votes
      if (st) ++st->iterationsH;

      PolyMatrix<C> P = coeffsBaseline<C>(p, h, st);
      if (st) st->matrixEntries += static_cast<long long>(P.rows) * P.cols;

      for (int k = 1; k <= p.L - 1; ++k) {
         for (int m = k * h; m <= N; ++m) {
            if (st) { ++st->termsSummed; ++st->entriesVisited; ++st->ops; }
            out.correct += fCorrect(k, h, m, p) * static_cast<Real>(P(k, m));
         }
      }

      for (std::map<int, Real>::iterator it = out.incorrect.begin(); it != out.incorrect.end(); ++it) {
         const int c = it->first;
         if (c < h) continue;
         removeClass(P, c, h, p.L - 1, N, h, st);
         Real v = 0;
         for (int k = 0; k <= p.L - 2; ++k) {
            for (int m = k * h; m <= N - c; ++m) {
               if (st) { ++st->termsSummed; ++st->entriesVisited; ++st->ops; }
               v -= fIncorrect(k, h, m, c, p) * static_cast<Real>(P(k, m));
            }
         }
         it->second += v;
         addClass(P, c, h, p.L - 2, N - c, h, st);
      }
   }
   return out;
}

// ---------------------------------------------------------------------------
// The Shapley values of one vote profile, ENPOWER_MC-phi++ (Appendix H.1)
// ---------------------------------------------------------------------------
template <typename C>
Payouts payoutsRefined(const Profile& p, Stats* st)
{
   Payouts out;
   out.correct = Real(1) / static_cast<Real>(p.n);
   for (int c : p.inc) out.incorrect[c] = 0;

   const ProfileFns f = profileFunctions(p);

   for (int h = 1; h <= f.Hstar; ++h) {                          // Cor. H.4 (i)
      if (st) ++st->iterationsH;

      const int      Kh    = f.K[h];
      const long long Phih = f.Phi[h];
      const int      width = static_cast<int>(f.Phi[h - 1]);     // the shifted matrix, Lemma H.2 (i)

      PolyMatrix<C> P = coeffsRefined<C>(p, h, f, st);           // Lemma H.5 (iii), (iv)
      if (st) st->matrixEntries += static_cast<long long>(P.rows) * P.cols;

      for (int k = 1; k <= Kh; ++k) {                            // Lemma H.5 (iii)
         const long long base = static_cast<long long>(k) * h;   // row k starts at m = kh
         const long long mmax = std::min(mBar(k, h, p, f), Phih);  // Lemma H.2 (i)
         for (long long m = base; m <= mmax; ++m) {
            if (st) { ++st->termsSummed; ++st->entriesVisited; ++st->ops; }
            out.correct += fCorrect(k, h, static_cast<int>(m), p)
                         * static_cast<Real>(P(k, static_cast<int>(m - base)));
         }
      }

      for (std::map<int, Real>::iterator it = out.incorrect.begin(); it != out.incorrect.end(); ++it) {
         const int c = it->first;
         if (c < h) continue;
         const int Kmj = f.Kminus(h, c);                         // Lemma H.5 (v)
         removeClass(P, c, h, Kh, width, 0, st);
         Real v = 0;
         for (int k = 0; k <= Kmj; ++k) {
            const long long base = static_cast<long long>(k) * h;
            const long long mmax = std::min(mBarMinus(k, h, c, p, f), Phih);  // Lemma H.2 (ii)
            for (long long m = base; m <= mmax; ++m) {
               if (st) { ++st->termsSummed; ++st->entriesVisited; ++st->ops; }
               v -= fIncorrect(k, h, static_cast<int>(m), c, p)
                  * static_cast<Real>(P(k, static_cast<int>(m - base)));
            }
         }
         it->second += v;
         addClass(P, c, h, Kmj, width - std::min(c, h - 1), 0, st);
      }
   }
   return out;
}

// ---------------------------------------------------------------------------
// The definition of the Shapley value, for testing
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
            int a = 0;
            long double l = 0;
            for (int q = 0; q < T; ++q) { a += s[q]; l += detail::logC(lim[q], s[q]); }
            l += -std::log(static_cast<long double>(p.n)) - detail::logC(p.n - 1, a);
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
template <typename T>
Profile makeProfile(const std::vector<T>& preds, T label, int labels)
{
   Profile p;
   p.n = static_cast<int>(preds.size());
   p.L = labels;
   std::vector<int> counts(labels, 0);
   for (size_t i = 0; i < preds.size(); ++i) ++counts[static_cast<size_t>(preds[i])];
   p.M1 = counts[static_cast<size_t>(label)];
   for (int j = 0; j < labels; ++j)
      if (j != static_cast<int>(label) && counts[j] > 0) p.inc.push_back(counts[j]);
   std::sort(p.inc.begin(), p.inc.end());
   return p;
}

// ---------------------------------------------------------------------------
// Selection of the algorithm and of the coefficient type
// ---------------------------------------------------------------------------
inline Payouts payoutsFor(const Profile& p, bool refined, Stats* st, Coefficients coef)
{
   if (coef == Coefficients::Float)
      return refined ? payoutsRefined <Real>(p, st) : payoutsBaseline<Real>(p, st);

   if (p.N() <= 126)
      return refined ? payoutsRefined <__int128>(p, st) : payoutsBaseline<__int128>(p, st);
   if (p.N() <= 255)
      return refined ? payoutsRefined <Wide>(p, st) : payoutsBaseline<Wide>(p, st);
   if (p.N() <= 511)
      return refined ? payoutsRefined <Wider>(p, st) : payoutsBaseline<Wider>(p, st);
   if (p.N() > kMaxExactN)
      throw std::overflow_error("shapmc::payoutsFor: N = " + std::to_string(p.N())
                                + " exceeds the exact width of " + std::to_string(kMaxExactN));
   return refined ? payoutsRefined <Widest>(p, st) : payoutsBaseline<Widest>(p, st);
}

} // namespace shapmc
