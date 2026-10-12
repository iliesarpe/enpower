#ifndef SAMPLING_H
#define SAMPLING_H

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

/**
* The Monte-Carlo baselines MC-beta and MC-phi of the paper (Alg. 8), with
* v(empty) = 0 as in Eq. (2) and (3).
*
* For each classifier i, s coalitions S of the other classifiers are drawn,
*   Banzhaf -- every other classifier joins S with probability 1/2;
*   Shapley -- S is the set of classifiers preceding i in a uniformly random
*              permutation of [n];
* and the estimate is the average of v(S + i) - v(S) over the s samples and
* the data points. v is the game of plurality voting with random tie-breaking
* of Eq. (3): on a data point with label y, v(S) = 1/|T(S)| if y is in the set
* T(S) of the labels with the most votes in S, and 0 otherwise.
*
* The number of votes of S for a label on a data point is the popcount of the
* AND of the bitmask of S and the bitmask of the classifiers that predict the
* label. Data points with the same label and the same predictions have the same
* game, so each is stored once with its multiplicity.
*/

namespace mcsample {

using Word = uint64_t;

class Game {
public:
   template <typename T>
   Game(const std::vector<std::vector<T>>& preds, const std::vector<T>& labs)
      : n_(preds.empty() ? 0 : static_cast<int>(preds[0].size())),
        words_((n_ + 63) / 64), points_(static_cast<int>(labs.size()))
   {
      std::map<std::pair<T, std::vector<T>>, int> seen;
      for (size_t p = 0; p < labs.size(); ++p) ++seen[{labs[p], preds[p]}];

      predIdx_.assign(n_, {});
      for (const auto& e : seen) {
         const std::vector<T>& row = e.first.second;
         Point pt;
         pt.weight = e.second;
         pt.first = slots_;
         std::vector<T> labels(row.begin(), row.end());
         std::sort(labels.begin(), labels.end());
         labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
         pt.count = static_cast<int>(labels.size());
         pt.truth = -1;
         for (int l = 0; l < pt.count; ++l) {
            if (labels[l] == e.first.first) pt.truth = l;
         }
         slots_ += pt.count;
         masks_.resize(static_cast<size_t>(slots_) * words_, 0);
         for (int j = 0; j < n_; ++j) {
            const int l = static_cast<int>(
               std::lower_bound(labels.begin(), labels.end(), row[j]) - labels.begin());
            masks_[(pt.first + l) * words_ + j / 64] |= Word(1) << (j % 64);
            predIdx_[j].push_back(l);
         }
         pts_.push_back(pt);
      }
   }

   int models() const { return n_; }
   int points() const { return points_; }
   int distinct() const { return static_cast<int>(pts_.size()); }

   // The estimates for every classifier, s samples each, drawn from the generator gen
   std::vector<double> estimate(bool shapley, long long s, std::mt19937_64& gen) const
   {
      std::vector<double> out(n_, 0.0);
      std::vector<Word> mask(words_);
      std::vector<int> perm(n_);
      std::vector<int> votes;
      for (int i = 0; i < n_; ++i) {
         long double sum = 0;
         for (long long k = 0; k < s; ++k) {
            std::fill(mask.begin(), mask.end(), 0);
            if (shapley) {
               std::iota(perm.begin(), perm.end(), 0);
               std::shuffle(perm.begin(), perm.end(), gen);
               for (int j : perm) {
                  if (j == i) break;
                  mask[j / 64] |= Word(1) << (j % 64);
               }
            } else {
               for (int w = 0; w < words_; ++w) mask[w] = gen();
               mask[i / 64] &= ~(Word(1) << (i % 64));
               if (n_ % 64) mask[words_ - 1] &= (Word(1) << (n_ % 64)) - 1;
            }
            sum += marginal(i, mask, votes);
         }
         out[i] = static_cast<double>(sum / (static_cast<long double>(s) * points_));
      }
      return out;
   }

private:
   struct Point {
      int weight;   // data points with this label and these predictions
      int first;    // index of its first (data point, label) entry
      int count;    // labels predicted by at least one classifier
      int truth;    // index of the true label among them, -1 if no classifier predicts it
   };

   int n_, words_, points_, slots_ = 0;
   std::vector<Point> pts_;
   std::vector<Word> masks_;               // classifiers predicting each (data point, label)
   std::vector<std::vector<int>> predIdx_; // [classifier][data point]: index of its prediction

   // v on a data point, from the votes of the coalition for each label
   static double value(const std::vector<int>& votes, int count, int truth)
   {
      if (truth < 0) return 0.0;
      int top = 0, ties = 0;
      for (int l = 0; l < count; ++l) {
         if (votes[l] > top) { top = votes[l]; ties = 1; }
         else if (votes[l] == top) ++ties;
      }
      if (top == 0) return 0.0;              // the empty coalition
      return votes[truth] == top ? 1.0 / ties : 0.0;
   }

   // sum of v(S + i) - v(S) over the data points, with multiplicity
   long double marginal(int i, const std::vector<Word>& mask, std::vector<int>& votes) const
   {
      long double sum = 0;
      for (size_t p = 0; p < pts_.size(); ++p) {
         const Point& pt = pts_[p];
         votes.assign(pt.count, 0);
         for (int l = 0; l < pt.count; ++l) {
            const Word* m = &masks_[(pt.first + l) * words_];
            for (int w = 0; w < words_; ++w) votes[l] += __builtin_popcountll(m[w] & mask[w]);
         }
         const double before = value(votes, pt.count, pt.truth);
         ++votes[predIdx_[i][p]];
         sum += pt.weight * (value(votes, pt.count, pt.truth) - before);
      }
      return sum;
   }
};

} // namespace mcsample

#endif // SAMPLING_H
