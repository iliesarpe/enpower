#ifndef BASELINES_H
#define BASELINES_H

#include <cstdint>
#include <stdexcept>
#include <vector>
#include "utils.h"

/**
* The exact baselines WMG and ISCG of the binary classification experiments.
* They are implemented in C++, as ENPOWER, so that the runtimes of all exact
* methods are measured in the same language.
*
*   wmg  -- Banzhaf and Shapley values under the weighted majority game of a
*           data point (Uno 2012);
*   iscg -- Shapley values under the induced subgraph game of Ykhlef and
*           Bouchaffra (IJAIT 2017), with the weights of their Eq. (5)-(7).
*/

namespace wmg {

using Real = long double;

// The value of a correct classifier. An incorrect classifier has weight 0, is
// never pivotal, and has value 0.
struct Payouts {
   Real banzhaf = 0;
   Real shapley = 0;
};

namespace detail {

   /**
   * The dynamic program of Uno (2012) on the game of one data point: n players,
   * c of weight 1 and the others of weight 0, and quota q.
   *
   * The forward pass counts the coalitions by weight (and by size, for the
   * Shapley value), with weights above q counted as q. The backward pass
   * removes one player of weight 1, which is pivotal for the coalitions of the
   * other players with weight q - 1. The counts are bounded by 2^n and the
   * backward pass subtracts them, so they are stored as integers of 64 W bits,
   * which are exact for n < 64 W.
   */
   template <int W>
   Payouts payouts(int n, int c, int q)
   {
      using C = utilities::WideUInt<W>;
      Payouts out;
      if (c == 0 || q <= 0) return out;         // no player is pivotal

      // Banzhaf: dp[k] = coalitions of weight k
      {
         std::vector<C> dp(q + 1, C(0));
         dp[0] = C(1);
         for (int p = 0; p < n; ++p) {
            const int w = (p < c) ? 1 : 0;
            for (int k = q; k >= w; --k) dp[k] += dp[k - w];
         }
         std::vector<C> ex(q + 1, C(0));
         ex[0] = dp[0];
         for (int k = 1; k <= q; ++k) ex[k] = dp[k] - ex[k - 1];
         out.banzhaf = std::ldexp(static_cast<Real>(static_cast<long double>(ex[q - 1])), -(n - 1));
      }

      // Shapley: dp[s][k] = coalitions of size s and weight k
      {
         std::vector<std::vector<C>> dp(n + 1, std::vector<C>(q + 1, C(0)));
         dp[0][0] = C(1);
         for (int p = 0; p < n; ++p) {
            const int w = (p < c) ? 1 : 0;
            for (int s = n; s >= 1; --s)
               for (int k = q; k >= w; --k) dp[s][k] += dp[s - 1][k - w];
         }
         std::vector<std::vector<C>> ex(n, std::vector<C>(q + 1, C(0)));
         for (int s = 0; s < n; ++s) {
            ex[s][0] = dp[s][0];
            for (int k = 1; k <= q; ++k)
               ex[s][k] = (s == 0) ? dp[0][k] : dp[s][k] - ex[s - 1][k - 1];
         }
         // the player follows a coalition of s other players with probability
         // s! (n - 1 - s)! / n! = eta(s + 1, n)
         Real acc = 0;
         for (int s = 0; s < n; ++s)
            acc += static_cast<Real>(static_cast<long double>(ex[s][q - 1])) * utilities::eta(s + 1, n);
         out.shapley = acc;
      }
      return out;
   }

} // namespace detail

/**
* The values of a correct classifier on a data point, for an ensemble of n
* classifiers of which c are correct on the data point, under quota q.
*/
inline Payouts payoutsFor(int n, int c, int q)
{
   if (n < 64)  return detail::payouts<1>(n, c, q);
   if (n < 128) return detail::payouts<2>(n, c, q);
   if (n < 256) return detail::payouts<4>(n, c, q);
   if (n < 512) return detail::payouts<8>(n, c, q);
   throw std::invalid_argument("wmg: ensembles of 512 classifiers or more are not supported");
}

} // namespace wmg


namespace iscg {

/**
* The Shapley values of the induced subgraph game of Ykhlef and Bouchaffra
* (IJAIT 2017) over a dataset.
*
* With Z_i = 1 when classifier i is correct on a data point drawn uniformly
* from the dataset, the weights of their Eq. (5)-(7) are
*
*     w_ii = Pr[Z_i = 1],
*     w_ij = (Pr[Z_i = 1 | Z_j = 0] + Pr[Z_j = 1 | Z_i = 0]) / 2,  i != j,
*
* and a probability conditioned on an event of probability 0, i.e., on a
* classifier that is always correct, is set to 0. The value of a coalition is the
* weight of the edges and self-loops inside it, so the Shapley value is the
* closed form of Eq. (8), phi_i = w_ii + (1/2) sum_{j != i} w_ij.
*
* Only the numbers N^11(i, j) of data points on which both classifiers are
* correct are required: with c_i the number of data points on which classifier i
* is correct and m the number of data points,
*
*     Pr[Z_i = 1 | Z_j = 0] = N^10(i, j) / (m - c_j) = (c_i - N^11(i, j)) / (m - c_j).
*
* The correctness of each classifier is stored as a bit vector, 64 data points
* per word, so N^11(i, j) is the popcount of the AND of two bit vectors.
*/
template <typename T>
std::vector<double> shapley(const std::vector<std::vector<T>>& preds, const std::vector<T>& labs)
{
   const size_t m = preds.size();
   const size_t n = m ? preds[0].size() : 0;
   const size_t words = (m + 63) / 64;

   std::vector<std::vector<std::uint64_t>> col(n, std::vector<std::uint64_t>(words, 0));
   for (size_t k = 0; k < m; ++k)
      for (size_t i = 0; i < n; ++i)
         if (preds[k][i] == labs[k]) col[i][k / 64] |= std::uint64_t(1) << (k % 64);

   std::vector<long long> c(n, 0);
   for (size_t i = 0; i < n; ++i)
      for (std::uint64_t w : col[i]) c[i] += __builtin_popcountll(w);

   // Pr[Z_i = 1 | Z_j = 0], 0 when classifier j is always correct
   auto given = [&](size_t i, size_t j, long long both) -> double {
      const long long wrongJ = static_cast<long long>(m) - c[j];
      return wrongJ ? static_cast<double>(c[i] - both) / static_cast<double>(wrongJ) : 0.0;
   };

   std::vector<double> phi(n, 0.0);
   for (size_t i = 0; i < n; ++i)
      phi[i] = m ? static_cast<double>(c[i]) / static_cast<double>(m) : 0.0;
   for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
         long long both = 0;
         for (size_t w = 0; w < words; ++w) both += __builtin_popcountll(col[i][w] & col[j][w]);
         const double div = 0.5 * (given(i, j, both) + given(j, i, both));
         phi[i] += 0.5 * div;
         phi[j] += 0.5 * div;
      }
   }
   return phi;
}

} // namespace iscg

#endif //BASELINES_H
