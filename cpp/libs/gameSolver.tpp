#include "utils.h"
#include <numeric>
#include <unordered_map>
#include <map>
#include <utility>
#include <iostream>

template <typename T>
const std::vector<double> Solver<T>::BanzhafBinary_() 
{

   const auto& predictions = this->preds_;
   const auto& labels = this-> labs_;

   size_t tot_points = predictions.size();
   size_t tot_models = predictions[0].size();
   std::vector<double> result(tot_models, 0);

   std::unordered_map<int, double> cache;
   
   size_t curr = 0; for(auto& point_pred : predictions)
   {
      auto lab = labels[curr];
      size_t n_corr = 0;
      for(size_t i = 0; i < tot_models; ++i)
      {
         n_corr += (point_pred[i] == lab);
      }

      if((cache.find(n_corr) == cache.end()) && (cache.find(tot_models - n_corr) == cache.end()) ){
         double banz_raw = utilities::binomial_pmf(tot_models, n_corr);
         cache[n_corr] = banz_raw;
      }
      double banz_raw = (cache.find(n_corr) != cache.end()) ? cache[n_corr] : cache[tot_models-n_corr];

      // add the values of the data point to the values of the classifiers
      for(size_t j = 0; j < tot_models; ++j)
      {
         double sign = (point_pred[j] == lab) ? 1.0 : -1.0;
         result[j] = std::fma(sign, banz_raw, result[j]);
      }
      curr += 1;
   }
   double term = std::ldexp(1.0, -tot_models);
   for(size_t j = 0; j < tot_models; ++j)
   {
      double average_pmf = result[j]/static_cast<double>(tot_points);
      result[j] = average_pmf + term;
   }

   return result;
}

template <typename T>
const std::vector<double> Solver<T>::ShapleyBinary_()
{

   const auto& predictions = this->preds_;
   const auto& labels = this->labs_;

   size_t tot_points = predictions.size();
   size_t tot_models = predictions[0].size();
   std::vector<double> result(tot_models, 0);

   // one entry per number of correct classifiers, which determines the vote
   // profile in the binary game
   std::unordered_map<int, shapbin::Payouts> cache;

   size_t curr = 0;
   for(auto& point_pred : predictions)
   {
      auto lab = labels[curr];
      int n_corr = 0;
      for(size_t i = 0; i < tot_models; ++i)
      {
         n_corr += (point_pred[i] == lab);
      }

      auto entry = cache.find(n_corr);
      if(entry == cache.end())
      {
         shapbin::Payouts payouts = shapbin::payoutsFor(static_cast<int>(tot_models), n_corr);
         entry = cache.insert(std::make_pair(n_corr, payouts)).first;
      }

      // add the values of the data point: positive for a correct classifier,
      // negative for an incorrect one
      for(size_t j = 0; j < tot_models; ++j)
      {
         result[j] += (point_pred[j] == lab)
            ? static_cast<double>(entry->second.correct)
            : -static_cast<double>(entry->second.incorrect);
      }
      curr += 1;
   }

   for(size_t j = 0; j < tot_models; ++j)
   {
      result[j] /= static_cast<double>(tot_points);
   }

   return result;
}

template <typename T>
const std::vector<double> Solver<T>::WMG_(bool shapley, int q)
{

   const auto& predictions = this->preds_;
   const auto& labels = this->labs_;

   size_t tot_points = predictions.size();
   size_t tot_models = predictions[0].size();
   std::vector<double> result(tot_models, 0);
   if (q <= 0) q = static_cast<int>(tot_models) / 2 + 1;

   // one entry per number of correct classifiers
   std::unordered_map<int, double> cache;

   size_t curr = 0;
   for(auto& point_pred : predictions)
   {
      auto lab = labels[curr];
      int n_corr = 0;
      for(size_t i = 0; i < tot_models; ++i)
      {
         n_corr += (point_pred[i] == lab);
      }

      auto entry = cache.find(n_corr);
      if(entry == cache.end())
      {
         wmg::Payouts payouts = wmg::payoutsFor(static_cast<int>(tot_models), n_corr, q);
         double payout = static_cast<double>(shapley ? payouts.shapley : payouts.banzhaf);
         entry = cache.insert(std::make_pair(n_corr, payout)).first;
      }

      // add the values of the data point: only a correct classifier has weight
      for(size_t j = 0; j < tot_models; ++j)
      {
         if (point_pred[j] == lab) result[j] += entry->second;
      }
      curr += 1;
   }

   for(size_t j = 0; j < tot_models; ++j)
   {
      result[j] /= static_cast<double>(tot_points);
   }

   return result;
}

template <typename T>
const std::vector<double> Solver<T>::BanzhafMC_(shapmc::Coefficients coef)
{

   const auto& predictions = this->preds_;
   const auto& labels = this->labs_;

   size_t tot_points = predictions.size();
   size_t tot_models = predictions[0].size();
   std::vector<double> result(tot_models, 0);

   // one entry per unique vote profile, since the values depend only on the profile
   std::map<std::pair<int, std::vector<int>>, bzmc::Payouts> cache;
   std::vector<int> counts(this->labels_, 0);

   for(size_t curr = 0; curr < tot_points; ++curr)
   {
      const auto& point_pred = predictions[curr];
      auto lab = labels[curr];

      shapmc::Profile profile = shapmc::makeProfile(point_pred, lab, this->labels_);
      auto key = std::make_pair(profile.M1, profile.inc);

      auto entry = cache.find(key);
      if(entry == cache.end())
      {
         bzmc::Payouts payouts = bzmc::payoutsFor(profile, coef);
         entry = cache.insert(std::make_pair(key, payouts)).first;
      }

      // add the values of the data point: the value of a classifier depends on
      // the number of votes of the label it predicts
      std::fill(counts.begin(), counts.end(), 0);
      for(size_t j = 0; j < tot_models; ++j)
      {
         counts[static_cast<size_t>(point_pred[j])] += 1;
      }
      for(size_t j = 0; j < tot_models; ++j)
      {
         double payout = (point_pred[j] == lab)
            ? static_cast<double>(entry->second.correct)
            : static_cast<double>(entry->second.incorrect.at(counts[static_cast<size_t>(point_pred[j])]));
         result[j] += payout;
      }
   }

   for(size_t j = 0; j < tot_models; ++j)
   {
      result[j] /= static_cast<double>(tot_points);
   }

   return result;
}

template <typename T>
const std::vector<double> Solver<T>::ShapleyMC_(bool refined, shapmc::Stats* stats,
                                                shapmc::Coefficients coef)
{

   const auto& predictions = this->preds_;
   const auto& labels = this->labs_;

   size_t tot_points = predictions.size();
   size_t tot_models = predictions[0].size();
   std::vector<double> result(tot_models, 0);

   // one entry per unique vote profile, since the values depend only on the profile
   std::map<std::pair<int, std::vector<int>>, shapmc::Payouts> cache;
   std::vector<int> counts(this->labels_, 0);

   for(size_t curr = 0; curr < tot_points; ++curr)
   {
      const auto& point_pred = predictions[curr];
      auto lab = labels[curr];

      shapmc::Profile profile = shapmc::makeProfile(point_pred, lab, this->labels_);
      auto key = std::make_pair(profile.M1, profile.inc);

      auto entry = cache.find(key);
      if(entry == cache.end())
      {
         shapmc::Payouts payouts = shapmc::payoutsFor(profile, refined, stats, coef);
         entry = cache.insert(std::make_pair(key, payouts)).first;
      }

      // add the values of the data point: the value of a classifier depends on
      // the number of votes of the label it predicts
      std::fill(counts.begin(), counts.end(), 0);
      for(size_t j = 0; j < tot_models; ++j)
      {
         counts[static_cast<size_t>(point_pred[j])] += 1;
      }
      for(size_t j = 0; j < tot_models; ++j)
      {
         double payout = (point_pred[j] == lab)
            ? static_cast<double>(entry->second.correct)
            : static_cast<double>(entry->second.incorrect.at(counts[static_cast<size_t>(point_pred[j])]));
         result[j] += payout;
      }
   }

   for(size_t j = 0; j < tot_models; ++j)
   {
      result[j] /= static_cast<double>(tot_points);
   }

   return result;
}
