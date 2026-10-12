#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include "shapleyMC.h"
#include "gameSolver.h"

// ---------------------------------------------------------------------------
// Tests of the algorithms of the library, in particular of ENPOWER_MC-phi++
// (Appendix H of the paper).
//
// For every vote profile, with the exact coefficients of P_h(t,u), it tests:
//
//   (1) the identities of Lemma H.5 on K(h), Phi(h), N^* and H^*;
//   (2) that the coefficients outside [0,K(h)] x [0,Phi(h)] are zero, that the
//       matrix of ENPOWER_MC-phi++ agrees with the matrix of ENPOWER_MC-phi
//       inside it, and that the coefficient (K(h), Phi(h)) is non-zero;
//   (3) that no term removed by \bar m(k,h) or by \bar m_{-j}(k,h) is non-zero;
//   (4) that no coefficient with k >= 1 is non-zero after an iteration h > H^*;
//   (5) that ENPOWER_MC-phi, ENPOWER_MC-phi++ and the brute-force computation
//       of the definition agree;
//   (6) that the wide integers agree with the 128-bit integers;
//   and the bounds of Lemma H.6 on sum_h K(h) Phi(h)^2.
//
// The coefficients are exact 128-bit integers, so (2), (3) and (4) are tested
// exactly.
//
// The other power indices of the library are tested on the same profiles:
//
//   (7) the Banzhaf values of ENPOWER_MC-beta against the definition, and its
//       wide coefficients against its 128-bit coefficients;
//   (8) the closed form of the Shapley values of ENPOWER_B against
//       ENPOWER_MC-phi on data with two labels, which define the same game.
//
// The two exact baselines of the binary classification experiments:
//
//   (9)  the dynamic program of WMG against the closed forms of its game,
//        C(c-1, q-1)/2^(c-1) and 1/c (when c >= q), and against the definitions
//        of both power indices for small n;
//   (10) ISCG against the definition of the Shapley value of the induced
//        subgraph game, on random datasets.
//
// The exit code is 0 if and only if all tests pass.
// ---------------------------------------------------------------------------

using Coef = __int128;
using shapmc::Profile;
using shapmc::ProfileFns;
using shapmc::Real;

namespace {

struct Report {
   long long profiles      = 0;
   long long bruteChecked  = 0;   // profiles small enough for the brute-force computation
   long long identities    = 0;   // violations of (1)
   long long window        = 0;   // violations of (2)
   long long corner        = 0;
   long long droppedCorr   = 0;   // violations of (3), first identity
   long long droppedIncorr = 0;   // violations of (3), second identity
   long long aboveHstar    = 0;   // violations of (4)
   long long valuesRefined = 0;   // violations of (5)
   long long valuesBrute   = 0;
   long long valuesWide    = 0;   // violations of (6)
   long long telescoping   = 0;   // violations of the bounds of Lemma H.6
   long long banzhafBrute  = 0;   // violations of (7)
   long long banzhafWide   = 0;
   long long binaryProfiles = 0;  // vote profiles of the binary sweep
   long long binaryShapley = 0;   // violations of (8)
   long long wmgProfiles   = 0;   // profiles (n, c) of (9)
   long long wmgClosed     = 0;   // violations of (9), closed forms
   long long wmgBrute      = 0;   // violations of (9), definitions
   long long iscgDatasets  = 0;   // datasets of (10)
   long long iscgBrute     = 0;   // violations of (10)
   long long entriesBase   = 0;
   long long entriesRef    = 0;
   long long allocBase     = 0;
   long long allocRef      = 0;

   long long total() const {
      return identities + window + corner + droppedCorr + droppedIncorr
           + aboveHstar + valuesRefined + valuesBrute + valuesWide + telescoping
           + banzhafBrute + banzhafWide + binaryShapley
           + wmgClosed + wmgBrute + iscgBrute;
   }
};

bool close(Real a, Real b, Real tol)
{
   const Real d = std::fabsl(a - b);
   const Real s = std::max<Real>(1.0L, std::max(std::fabsl(a), std::fabsl(b)));
   return d <= tol * s;
}

std::string profileString(const Profile& p)
{
   std::string s = "n=" + std::to_string(p.n) + " L=" + std::to_string(p.L)
                 + " n_1=" + std::to_string(p.M1) + " inc=[";
   for (size_t i = 0; i < p.inc.size(); ++i) {
      s += std::to_string(p.inc[i]);
      if (i + 1 < p.inc.size()) s += ",";
   }
   return s + "]";
}

void fail(Report& r, long long& counter, const Profile& p, const std::string& what)
{
   ++counter;
   if (r.total() <= 20) std::cout << "  FAIL  " << profileString(p) << " : " << what << "\n";
}

// ---------------------------------------------------------------------------
void auditProfile(const Profile& p, Report& r)
{
   ++r.profiles;
   const ProfileFns f = shapmc::profileFunctions(p);
   const int N = p.N();

   // (1) the identities of Lemma H.5 ----------------------------------------
   for (int h = 1; h <= f.Hstar; ++h) {
      int kDirect = 0;
      long long phiDirect = 0;
      for (int c : p.inc) {
         if (c >= h) ++kDirect;
         phiDirect += std::min(c, h);
      }
      if (f.K[h] != kDirect)
         fail(r, r.identities, p, "K(" + std::to_string(h) + ") from the buckets differs from the definition");
      if (f.Phi[h] != phiDirect)
         fail(r, r.identities, p, "Phi(" + std::to_string(h) + ") differs from the definition");
      if (f.K[h] != f.Phi[h] - f.Phi[h - 1])
         fail(r, r.identities, p, "K(h) is not the discrete derivative of Phi");
      if (f.K[h] > p.L - 1)
         fail(r, r.identities, p, "K(h) exceeds L-1");
   }
   long long nstarDirect = 0;
   for (int c : p.inc) nstarDirect += std::min(c, p.M1);
   if (f.Nstar != nstarDirect)
      fail(r, r.identities, p, "N^* differs from sum_j min{n_j, n_1}");
   if (f.Nstar > std::min<long long>(N, static_cast<long long>(p.L - 1) * p.M1))
      fail(r, r.identities, p, "N^* exceeds min{N, (L-1) n_1}");
   if (f.Hstar > f.Nstar && f.Nstar >= 0 && f.Hstar > 0)
      fail(r, r.identities, p, "H^* exceeds N^*");

   // (2), (3) the support of the coefficients --------------------------------
   for (int h = 1; h <= f.Hstar; ++h) {
      shapmc::PolyMatrix<Coef> full = shapmc::coeffsBaseline<Coef>(p, h, nullptr);
      shapmc::PolyMatrix<Coef> ref  = shapmc::coeffsRefined <Coef>(p, h, f, nullptr);

      for (int k = 0; k < full.rows; ++k) {
         for (int m = 0; m < full.cols; ++m) {
            const long long ms = static_cast<long long>(m) - static_cast<long long>(k) * h;
            if (ms < 0) {                      // the support of Lemma H.2
               if (full(k, m) != 0)
                  fail(r, r.window, p, "a non-zero coefficient lies below m = k h, h="
                                        + std::to_string(h) + " k=" + std::to_string(k) + " m=" + std::to_string(m));
               continue;
            }
            const bool stored = (k <= f.K[h] && m <= f.Phi[h] && ms < ref.cols);
            if (stored) {
               if (ref(k, static_cast<int>(ms)) != full(k, m))
                  fail(r, r.window, p, "the refined matrix differs from the baseline one inside the window");
            } else if (full(k, m) != 0) {
               fail(r, r.window, p, "a non-zero coefficient lies outside the shifted window, h="
                                     + std::to_string(h) + " k=" + std::to_string(k) + " m=" + std::to_string(m));
            }
         }
      }
      if (full(f.K[h], static_cast<int>(f.Phi[h])) <= 0)
         fail(r, r.corner, p, "the corner (K(h), Phi(h)) is zero, h=" + std::to_string(h));

      // first identity: the terms removed by the bounds of the loops over k and m
      for (int k = 1; k <= p.L - 1; ++k) {
         const long long mmax = std::min(shapmc::mBar(k, h, p, f), f.Phi[h]);
         for (int m = 0; m <= N; ++m) {
            const bool kept = (k <= f.K[h] && m >= k * h && m <= mmax);
            if (!kept && full(k, m) != 0)
               fail(r, r.droppedCorr, p, "a dropped term of the first identity is non-zero, h="
                                          + std::to_string(h) + " k=" + std::to_string(k) + " m=" + std::to_string(m));
         }
      }

      // second identity: the same, on the quotient by A_j(h;u) + B_j(h;u) t
      int last = -1;
      for (int c : p.inc) {
         if (c == last || c < h) continue;
         last = c;
         shapmc::PolyMatrix<Coef> quo = full;
         shapmc::removeClass(quo, c, h, p.L - 1, N, h, nullptr);
         const int kmj = f.Kminus(h, c);
         for (int k = 0; k <= p.L - 2; ++k) {
            const long long mmax = std::min(shapmc::mBarMinus(k, h, c, p, f), f.Phi[h]);
            for (int m = 0; m <= N - c; ++m) {
               const bool kept = (k <= kmj && m >= k * h && m <= mmax);
               if (!kept && quo(k, m) != 0)
                  fail(r, r.droppedIncorr, p, "a dropped term of the second identity is non-zero, h="
                                               + std::to_string(h) + " c=" + std::to_string(c)
                                               + " k=" + std::to_string(k) + " m=" + std::to_string(m));
            }
         }
      }
   }

   // (4) the iterations above H^* -------------------------------------------
   for (int h = f.Hstar + 1; h <= p.M1; ++h) {
      shapmc::PolyMatrix<Coef> full = shapmc::coeffsBaseline<Coef>(p, h, nullptr);
      for (int k = 1; k < full.rows; ++k)
         for (int m = 0; m < full.cols; ++m)
            if (full(k, m) != 0)
               fail(r, r.aboveHstar, p, "a non-zero coefficient with k >= 1 survives h > H^*, h=" + std::to_string(h));
   }

   // (5) the values ---------------------------------------------------------
   shapmc::Stats sb, sr;
   const shapmc::Payouts base  = shapmc::payoutsBaseline<Coef>(p, &sb);
   const shapmc::Payouts refd  = shapmc::payoutsRefined <Coef>(p, &sr);

   // the brute-force computation enumerates prod_j (n_j + 1) vote profiles per label
   double bruteCost = static_cast<double>(p.M1 + 1);
   for (int c : p.inc) bruteCost *= static_cast<double>(c + 1);
   bruteCost *= static_cast<double>(p.inc.size() + 1);
   const bool doBrute = (bruteCost <= 2e7);
   const shapmc::Payouts brute = doBrute ? shapmc::payoutsBruteForce(p) : shapmc::Payouts();
   if (doBrute) ++r.bruteChecked;

   r.entriesBase += sb.entriesVisited;
   r.entriesRef  += sr.entriesVisited;
   r.allocBase   += sb.matrixEntries;
   r.allocRef    += sr.matrixEntries;

   if (p.M1 > 0) {
      if (!close(base.correct, refd.correct, 1e-15L))
         fail(r, r.valuesRefined, p, "the refined value of a correct classifier differs from the baseline one");
      if (doBrute && !close(base.correct, brute.correct, 1e-12L))
         fail(r, r.valuesBrute, p, "the value of a correct classifier differs from the brute force one");
   }
   for (std::map<int, Real>::const_iterator it = base.incorrect.begin(); it != base.incorrect.end(); ++it) {
      const int c = it->first;
      if (!close(it->second, refd.incorrect.at(c), 1e-15L))
         fail(r, r.valuesRefined, p, "the refined value of a classifier of " + std::to_string(c) + " votes differs");
      if (doBrute && !close(it->second, brute.incorrect.at(c), 1e-12L))
         fail(r, r.valuesBrute, p, "the value of a classifier of " + std::to_string(c) + " votes differs from brute force");
   }

   // (6) the wide integers, used when the coefficients exceed 128 bits, against
   // the 128-bit integers on a profile where all three types are exact. Both
   // widths are tested, since payoutsFor selects between them by N.
   const shapmc::Payouts wide  = shapmc::payoutsRefined<shapmc::Wide >(p, nullptr);
   const shapmc::Payouts wider = shapmc::payoutsRefined<shapmc::Wider>(p, nullptr);
   if (p.M1 > 0 && (!close(refd.correct, wide.correct, 1e-15L)
                 || !close(refd.correct, wider.correct, 1e-15L)))
      fail(r, r.valuesWide, p, "the wide value of a correct classifier differs from the 128 bit one");
   for (std::map<int, Real>::const_iterator it = refd.incorrect.begin(); it != refd.incorrect.end(); ++it)
      if (!close(it->second, wide.incorrect.at(it->first), 1e-15L)
       || !close(it->second, wider.incorrect.at(it->first), 1e-15L))
         fail(r, r.valuesWide, p, "the wide value of a classifier of " + std::to_string(it->first) + " votes differs");

   // (7) the multiclass Banzhaf values ---------------------------------------
   // The dynamic program of bzmc against the definition of the Banzhaf value,
   // and the wide coefficients against the 128-bit coefficients, on the
   // profiles small enough for the brute-force computation.
   const bzmc::Payouts bz = bzmc::payoutsFor(p);
   if (doBrute) {
      const bzmc::Payouts bzBrute = bzmc::payoutsBruteForce(p);
      if (p.M1 > 0 && !close(bz.correct, bzBrute.correct, 1e-12L))
         fail(r, r.banzhafBrute, p, "the Banzhaf value of a correct classifier differs from the brute force one");
      for (std::map<int, Real>::const_iterator it = bz.incorrect.begin(); it != bz.incorrect.end(); ++it)
         if (!close(it->second, bzBrute.incorrect.at(it->first), 1e-12L))
            fail(r, r.banzhafBrute, p, "the Banzhaf value of a classifier of "
                                        + std::to_string(it->first) + " votes differs from brute force");
   }
   const bzmc::Payouts bzWide  = bzmc::payoutsDP<shapmc::Wide >(p);
   const bzmc::Payouts bzWider = bzmc::payoutsDP<shapmc::Wider>(p);
   if (p.M1 > 0 && (!close(bz.correct, bzWide.correct, 1e-15L)
                 || !close(bz.correct, bzWider.correct, 1e-15L)))
      fail(r, r.banzhafWide, p, "the wide Banzhaf value of a correct classifier differs from the 128 bit one");
   for (std::map<int, Real>::const_iterator it = bz.incorrect.begin(); it != bz.incorrect.end(); ++it)
      if (!close(it->second, bzWide.incorrect.at(it->first), 1e-15L)
       || !close(it->second, bzWider.incorrect.at(it->first), 1e-15L))
         fail(r, r.banzhafWide, p, "the wide Banzhaf value of a classifier of "
                                    + std::to_string(it->first) + " votes differs");

   // the bounds of Lemma H.6 ----------------------------------------------
   long double telescope = 0;
   for (int h = 1; h <= f.Hstar; ++h)
      telescope += static_cast<long double>(f.K[h]) * static_cast<long double>(f.Phi[h]) * static_cast<long double>(f.Phi[h]);
   const long double ns = static_cast<long double>(f.Nstar);
   const long double lo = ns * ns * ns / 3.0L;
   const long double hi = lo + 2.0L * p.L * ns * ns;
   if (f.Nstar > 0 && (telescope < lo - 1e-6L || telescope > hi + 1e-6L))
      fail(r, r.telescoping, p, "sum_h K(h) Phi(h)^2 is outside the bounds of Lemma H.6");
}

// ---------------------------------------------------------------------------
// (8) the closed form of the Shapley values of ENPOWER_B, on every vote profile
// of an ensemble of up to nmax classifiers.
//
// The binary game is the multiclass game with two labels, so the closed form
// must agree with ENPOWER_MC-phi on the two-label profile (n_1, [n - n_1]),
// which auditProfile tests against the definition of the Shapley value. The
// closed form returns the absolute value of the (negative) value of an
// incorrect classifier.
void binarySweep(int nmax, Report& r)
{
   for (int n = 1; n <= nmax; ++n) {
      for (int nPlus = 0; nPlus <= n; ++nPlus) {
         if (n - nPlus > 126) continue;   // the 128-bit reference is exact for N <= 126
         ++r.binaryProfiles;
         const shapbin::Payouts got = shapbin::payoutsFor(n, nPlus);

         Profile p;
         p.n = n; p.L = 2; p.M1 = nPlus;
         if (n - nPlus > 0) p.inc.push_back(n - nPlus);
         const shapmc::Payouts want = shapmc::payoutsRefined<Coef>(p, nullptr);

         if (nPlus > 0 && !close(got.correct, want.correct, 1e-12L))
            fail(r, r.binaryShapley, p, "the binary value of a correct classifier differs from the multiclass one");
         if (n - nPlus > 0 && !close(-got.incorrect, want.incorrect.at(n - nPlus), 1e-12L))
            fail(r, r.binaryShapley, p, "the binary value of an incorrect classifier differs from the multiclass one");
      }
   }
}

// (9) the baseline WMG on every profile (n, c) with n <= nmax, under the quota
// floor(n/2) + 1 and, for small n, under every quota.
//
// The game of a data point is v(S) = 1[|S cap R| >= q], with R the set of the c
// correct classifiers. Only the classifiers of R affect the value, and they are
// symmetric, so a classifier of R is pivotal for the coalitions with exactly
// q - 1 other classifiers of R:
//
//     Banzhaf = C(c-1, q-1) / 2^(c-1),     Shapley = 1/c if c >= q, else 0,
//
// the latter by efficiency, since v(N) = 1[c >= q]. For n <= bruteMax both
// are also computed from the definitions, over every coalition.
void wmgSweep(int nmax, int bruteMax, Report& r)
{
   for (int n = 1; n <= nmax; ++n) {
      const int qDefault = n / 2 + 1;
      for (int q = 1; q <= n; ++q) {
         if (n > bruteMax && q != qDefault) continue;
         for (int c = 1; c <= n; ++c) {
            ++r.wmgProfiles;
            const wmg::Payouts got = wmg::payoutsFor(n, c, q);
            const Real bz = (c >= q) ? std::exp(utilities::log_binom(c - 1, q - 1)
                                                - (c - 1) * std::log(2.0L)) : 0.0L;
            const Real sh = (c >= q) ? 1.0L / c : 0.0L;
            Profile p; p.n = n; p.L = 2; p.M1 = c;   // for the failure message only
            if (!close(got.banzhaf, bz, 1e-12L))
               fail(r, r.wmgClosed, p, "WMG Banzhaf differs from C(c-1,q-1)/2^(c-1), q = " + std::to_string(q));
            if (!close(got.shapley, sh, 1e-12L))
               fail(r, r.wmgClosed, p, "WMG Shapley differs from 1/c, q = " + std::to_string(q));

            if (n > bruteMax) continue;
            // classifiers 0..c-1 are correct on the data point
            Real bzB = 0, shB = 0;
            for (unsigned S = 0; S < (1u << (n - 1)); ++S) {     // coalitions of the classifiers 1..n-1
               int right = 0, size = 0;
               for (int j = 1; j < n; ++j)
                  if (S >> (j - 1) & 1u) { ++size; right += (j < c); }
               const Real marginal = (right + 1 >= q) - (right >= q);
               bzB += marginal;
               shB += marginal * utilities::eta(size + 1, n);
            }
            bzB = std::ldexp(bzB, -(n - 1));
            if (!close(got.banzhaf, bzB, 1e-12L))
               fail(r, r.wmgBrute, p, "WMG Banzhaf differs from its definition, q = " + std::to_string(q));
            if (!close(got.shapley, shB, 1e-12L))
               fail(r, r.wmgBrute, p, "WMG Shapley differs from its definition, q = " + std::to_string(q));
         }
      }
   }
}

// (10) the ISCG values on random binary datasets against the definition: the
// weights recomputed from the data points with Eq. (5)-(7) of Ykhlef and
// Bouchaffra, the game v(S) = sum of w_ij over the edges and self-loops inside
// S, and the Shapley value as the weighted sum of the marginal contributions
// over every coalition.
void iscgRandom(int trials, unsigned seed, int nmax, Report& r)
{
   std::mt19937 gen(seed);
   for (int t = 0; t < trials; ++t) {
      const int n = std::uniform_int_distribution<int>(1, nmax)(gen);
      const int m = std::uniform_int_distribution<int>(1, 60)(gen);
      // some classifiers are always correct, to test the convention for 0/0
      const double pRight = std::uniform_real_distribution<double>(0.3, 1.0)(gen);
      std::vector<std::vector<uint32_t>> preds(m, std::vector<uint32_t>(n));
      std::vector<uint32_t> labs(m);
      for (int k = 0; k < m; ++k) {
         labs[k] = std::uniform_int_distribution<uint32_t>(0, 1)(gen);
         for (int i = 0; i < n; ++i) {
            const bool right = (i == 0) || std::bernoulli_distribution(pRight)(gen);
            preds[k][i] = right ? labs[k] : 1u - labs[k];
         }
      }
      ++r.iscgDatasets;
      const std::vector<double> got = iscg::shapley(preds, labs);

      std::vector<std::vector<Real>> w(n, std::vector<Real>(n, 0));
      for (int i = 0; i < n; ++i) {
         int right = 0;
         for (int k = 0; k < m; ++k) right += (preds[k][i] == labs[k]);
         w[i][i] = static_cast<Real>(right) / m;                              // Eq. (5)
      }
      for (int i = 0; i < n; ++i)
         for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int n10 = 0, n01 = 0, n00 = 0;
            for (int k = 0; k < m; ++k) {
               const bool zi = preds[k][i] == labs[k], zj = preds[k][j] == labs[k];
               n10 += zi && !zj; n01 += !zi && zj; n00 += !zi && !zj;
            }
            const Real a = (n00 + n10) ? static_cast<Real>(n10) / (n00 + n10) : 0;
            const Real b = (n00 + n01) ? static_cast<Real>(n01) / (n00 + n01) : 0;
            w[i][j] = 0.5L * (a + b);                                         // Eq. (6)
         }
      auto value = [&](unsigned S) {
         Real v = 0;
         for (int i = 0; i < n; ++i) {
            if (!(S >> i & 1u)) continue;
            v += w[i][i];
            for (int j = i + 1; j < n; ++j) if (S >> j & 1u) v += w[i][j];
         }
         return v;
      };
      for (int i = 0; i < n; ++i) {
         Real phi = 0;
         for (unsigned S = 0; S < (1u << n); ++S) {
            if (S >> i & 1u) continue;
            phi += (value(S | (1u << i)) - value(S)) * utilities::eta(__builtin_popcount(S) + 1, n);
         }
         Profile p; p.n = n; p.L = 2; p.M1 = i;   // for the failure message only
         if (!close(got[i], phi, 1e-12L))
            fail(r, r.iscgBrute, p, "ISCG value differs from the Shapley value of the induced subgraph game");
      }
   }
}

// ---------------------------------------------------------------------------
// every vote profile with the given n and L, up to permutation of the counts
void partitions(int rest, int parts, int cap, std::vector<int>& acc, std::vector<std::vector<int>>& out)
{
   if (rest == 0) { out.push_back(acc); return; }
   if (parts == 0) return;
   for (int v = std::min(cap, rest); v >= 1; --v) {
      acc.push_back(v);
      partitions(rest - v, parts - 1, v, acc, out);
      acc.pop_back();
   }
}

void sweep(int nmax, int Lmax, Report& r)
{
   for (int L = 2; L <= Lmax; ++L) {
      for (int n = 1; n <= nmax; ++n) {
         for (int M1 = 0; M1 <= n; ++M1) {
            std::vector<std::vector<int>> parts;
            std::vector<int> acc;
            if (n - M1 == 0) parts.push_back(std::vector<int>());
            else             partitions(n - M1, L - 1, n - M1, acc, parts);
            for (std::vector<int>& q : parts) {
               Profile p;
               p.n = n; p.L = L; p.M1 = M1;
               p.inc = q;
               std::sort(p.inc.begin(), p.inc.end());
               auditProfile(p, r);
            }
         }
      }
   }
}

void randomProfiles(int trials, unsigned seed, int nmax, int Lmax, Report& r)
{
   std::mt19937 gen(seed);
   for (int t = 0; t < trials; ++t) {
      std::uniform_int_distribution<int> dn(2, nmax);
      std::uniform_int_distribution<int> dl(2, Lmax);
      const int n = dn(gen);
      const int L = dl(gen);
      std::uniform_int_distribution<int> dlab(0, L - 1);
      std::vector<uint32_t> preds(n);
      for (int i = 0; i < n; ++i) preds[i] = static_cast<uint32_t>(dlab(gen));
      const uint32_t y = static_cast<uint32_t>(dlab(gen));
      auditProfile(shapmc::makeProfile(preds, y, L), r);
   }
}

// ---------------------------------------------------------------------------
// The number of operations of ENPOWER_MC-phi and ENPOWER_MC-phi++ as the
// number n_1 of correct classifiers grows
void bench(int n, int L, unsigned seed)
{
   std::cout << "--- Operations of ENPOWER_MC-phi and ENPOWER_MC-phi++, n = " << n
             << ", L = " << L << " ---\n";
   std::cout << std::setw(6) << "n_1" << std::setw(7) << "N^*" << std::setw(6) << "H^*"
             << std::setw(13) << "ops phi" << std::setw(13) << "ops phi*"
             << std::setw(13) << "ops phi++"
             << std::setw(10) << "speedup" << std::setw(12) << "L(n/N^*)^3"
             << std::setw(11) << "/(L n^3)" << std::setw(11) << "/(N^*)^3" << "\n";

   std::mt19937 gen(seed);
   for (int num = 1; num <= 9; ++num) {
      const int M1 = std::max(1, n * num / 10);
      // the incorrect votes are assigned uniformly at random to the other labels
      std::vector<int> counts(L, 0);
      std::uniform_int_distribution<int> dlab(1, L - 1);
      for (int i = 0; i < n - M1; ++i) ++counts[dlab(gen)];

      Profile p;
      p.n = n; p.L = L; p.M1 = M1;
      for (int j = 1; j < L; ++j) if (counts[j] > 0) p.inc.push_back(counts[j]);
      std::sort(p.inc.begin(), p.inc.end());
      if (p.inc.empty()) continue;

      const ProfileFns f = shapmc::profileFunctions(p);
      shapmc::Stats spaper, sstop, sr;
      shapmc::payoutsBaseline<Real>(p, &spaper, false);  // the loop over h runs up to n_1, as in Alg. 4
      shapmc::payoutsBaseline<Real>(p, &sstop,  true);   // the loop over h stops at n_(1)
      shapmc::payoutsRefined <Real>(p, &sr);

      const double measured  = sr.ops ? static_cast<double>(spaper.ops) / sr.ops : 0.0;
      const double predicted = f.Nstar ? L * std::pow(static_cast<double>(n) / static_cast<double>(f.Nstar), 3.0) : 0.0;
      std::cout << std::setw(6) << M1 << std::setw(7) << f.Nstar << std::setw(6) << f.Hstar
                << std::setw(13) << spaper.ops << std::setw(13) << sstop.ops
                << std::setw(13) << sr.ops
                << std::setw(9) << std::fixed << std::setprecision(1) << measured << "x"
                << std::setw(12) << predicted
                << std::setw(11) << std::setprecision(3)
                << static_cast<double>(spaper.ops) / (L * std::pow(static_cast<double>(n), 3.0))
                << std::setw(11)
                << (f.Nstar ? static_cast<double>(sr.ops) / std::pow(static_cast<double>(f.Nstar), 3.0) : 0.0)
                << std::setprecision(1) << "\n";
   }

   // the same comparison on a dataset, through Solver<T>
   const int points = 200;
   std::uniform_int_distribution<int> dlab(0, L - 1);
   std::vector<std::vector<uint32_t>> preds(points, std::vector<uint32_t>(n));
   std::vector<uint32_t> labs(points);
   for (int i = 0; i < points; ++i) {
      labs[i] = static_cast<uint32_t>(dlab(gen));
      for (int j = 0; j < n; ++j)
         preds[i][j] = (dlab(gen) < 1) ? labs[i] : static_cast<uint32_t>(dlab(gen)); // ~60% correct
   }
   Solver<uint32_t> solver(preds, labs);
   shapmc::Stats sb, sr;
   auto t0 = std::chrono::high_resolution_clock::now();
   std::vector<double> vb = solver.getShapleyMC(false, &sb);
   auto t1 = std::chrono::high_resolution_clock::now();
   std::vector<double> vr = solver.getShapleyMC(true, &sr);
   auto t2 = std::chrono::high_resolution_clock::now();

   double worst = 0;
   for (size_t j = 0; j < vb.size(); ++j) worst = std::max(worst, std::fabs(vb[j] - vr[j]));
   std::cout << "\ndataset of " << points << " points: ENPOWER_MC-phi "
             << std::chrono::duration<double>(t1 - t0).count() << " s, ENPOWER_MC-phi++ "
             << std::chrono::duration<double>(t2 - t1).count() << " s, largest difference between the values "
             << std::scientific << worst << std::fixed << "\n";
}

} // namespace

int main(int argc, char** argv)
{
   int nmax = 8, Lmax = 5, trials = 400, rnmax = 14, rLmax = 6, bnmax = 120;
   int benchN = 64, benchL = 8;
   bool doBench = false;
   unsigned seed = 42;
   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if      (a == "--bench")                  doBench = true;
      else if (a == "--benchN" && i + 1 < argc) benchN = std::stoi(argv[++i]);
      else if (a == "--benchL" && i + 1 < argc) benchL = std::stoi(argv[++i]);
      else if (a == "--nmax"   && i + 1 < argc) nmax   = std::stoi(argv[++i]);
      else if (a == "--Lmax"   && i + 1 < argc) Lmax   = std::stoi(argv[++i]);
      else if (a == "--trials" && i + 1 < argc) trials = std::stoi(argv[++i]);
      else if (a == "--rnmax"  && i + 1 < argc) rnmax  = std::stoi(argv[++i]);
      else if (a == "--rLmax"  && i + 1 < argc) rLmax  = std::stoi(argv[++i]);
      else if (a == "--bnmax"  && i + 1 < argc) bnmax  = std::stoi(argv[++i]);
      else if (a == "--seed"   && i + 1 < argc) seed   = static_cast<unsigned>(std::stoul(argv[++i]));
   }

   if (doBench) { bench(benchN, benchL, seed); return 0; }

   Report r;
   std::cout << "--- Exhaustive sweep: every vote profile with n <= " << nmax
             << " and L <= " << Lmax << " ---\n";
   sweep(nmax, Lmax, r);
   std::cout << "--- Random profiles: " << trials << " draws with n <= " << rnmax
             << ", L <= " << rLmax << ", seed " << seed << " ---\n";
   randomProfiles(trials, seed, rnmax, rLmax, r);
   std::cout << "--- Binary Shapley: every vote profile with n <= " << bnmax << " ---\n";
   binarySweep(bnmax, r);
   std::cout << "--- WMG: every profile with n <= 200, every quota and the definition for n <= 12 ---\n";
   wmgSweep(200, 12, r);
   std::cout << "--- ISCG: 300 random datasets with n <= 10 ---\n";
   iscgRandom(300, seed, 10, r);

   std::cout << "\nprofiles checked            : " << r.profiles << "\n";
   std::cout << "profile function identities : " << r.identities << " violations\n";
   std::cout << "coefficients outside window : " << r.window << " violations\n";
   std::cout << "corner (K(h), Phi(h)) > 0   : " << r.corner << " violations\n";
   std::cout << "dropped terms, 1st identity : " << r.droppedCorr << " violations\n";
   std::cout << "dropped terms, 2nd identity : " << r.droppedIncorr << " violations\n";
   std::cout << "coefficients above H^*      : " << r.aboveHstar << " violations\n";
   std::cout << "refined == baseline values  : " << r.valuesRefined << " violations\n";
   std::cout << "values == brute force       : " << r.valuesBrute << " violations ("
             << r.bruteChecked << " profiles small enough)\n";
   std::cout << "wide == 128 bit values      : " << r.valuesWide << " violations\n";
   std::cout << "telescoping bounds          : " << r.telescoping << " violations\n";
   std::cout << "Banzhaf == brute force      : " << r.banzhafBrute << " violations\n";
   std::cout << "Banzhaf wide == 128 bit     : " << r.banzhafWide << " violations\n";
   std::cout << "binary == multiclass values : " << r.binaryShapley << " violations ("
             << r.binaryProfiles << " binary profiles)\n";
   std::cout << "WMG == closed forms        : " << r.wmgClosed << " violations ("
             << r.wmgProfiles << " profiles)\n";
   std::cout << "WMG == definitions         : " << r.wmgBrute << " violations\n";
   std::cout << "ISCG == definition         : " << r.iscgBrute << " violations ("
             << r.iscgDatasets << " datasets)\n";
   std::cout << std::fixed << std::setprecision(2);
   std::cout << "entries of P_h visited     : baseline " << r.entriesBase
             << ", refined " << r.entriesRef
             << "  (x" << (r.entriesRef ? static_cast<double>(r.entriesBase) / r.entriesRef : 0.0) << ")\n";
   std::cout << "entries of P_h allocated    : baseline " << r.allocBase
             << ", refined " << r.allocRef
             << "  (x" << (r.allocRef ? static_cast<double>(r.allocBase) / r.allocRef : 0.0) << ")\n";

   if (r.total() == 0) { std::cout << "\nALL CHECKS PASSED\n"; return 0; }
   std::cout << "\n" << r.total() << " CHECKS FAILED\n";
   return 1;
}
