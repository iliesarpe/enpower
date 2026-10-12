#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include "shapleyMC.h"
#include "banzhafMC.h"
#include "shapleyBinary.h"
#include "baselines.h"

template <typename T = uint32_t> // the type of the labels

class Solver
{
   public:
      Solver(const std::vector<std::vector<T>>& predictions, const std::vector<T>& labs):
      preds_(predictions), labs_(labs)
      {
         assert(predictions.size() == labs.size());
         classes_ = *std::max_element(labs.begin(), labs.end());
         binary_ = false;
         if ((classes_ == 0) || (classes_ == 1))
         {
            binary_ = true;
         }
         // the number of labels, as required by the multiclass game
         labels_ = classes_ + 1;
         for (const auto& row : predictions)
         {
            for (const auto& v : row)
            {
               labels_ = std::max(labels_, static_cast<int>(v) + 1);
            }
         }
      }

   // The Banzhaf values of the base classifiers in the game over the dataset.
   // With binary=true it computes the closed form of ENPOWER_B, otherwise the
   // dynamic program of ENPOWER_MC-beta; coef selects the arithmetic of the
   // coefficients of the dynamic program, as in getShapleyMC.
   const std::vector<double> getBanzhaf(bool binary=false,
                                        shapmc::Coefficients coef=shapmc::Coefficients::Exact)
   {
      if(binary)
      {
         return BanzhafBinary_();
      }
      else
      {
         return BanzhafMC_(coef);
      }

   }

   // The Shapley values of the base classifiers in the binary game over the
   // dataset, from the closed form of ENPOWER_B, in O(n) time per distinct vote
   // profile. The binary game is the multiclass game with two labels, so on
   // data with two classes the values are equal to those of getShapleyMC.
   const std::vector<double> getShapleyBinary()
   {
      return ShapleyBinary_();
   }

   // The Shapley values of the base classifiers in the multiclass game over
   // the dataset: ENPOWER_MC-phi++ with refined=true, ENPOWER_MC-phi with
   // refined=false. Both return the same values. stats, when given, receives
   // the operation counts of the computation.
   //
   // coef selects the arithmetic of the coefficients of P_h. The coefficients
   // are numbers of coalitions and RemoveClass divides one polynomial by
   // another; in floating point the error of each division is amplified over
   // the loop over h, so the coefficients are exact. See shapmc::Coefficients.
   const std::vector<double> getShapleyMC(bool refined=true, shapmc::Stats* stats=nullptr,
                                          shapmc::Coefficients coef=shapmc::Coefficients::Exact)
   {
      return ShapleyMC_(refined, stats, coef);
   }

   // The Banzhaf (shapley=false) or Shapley (shapley=true) values of the
   // baseline WMG: the weighted majority game of a data point gives weight 1 to
   // the correct classifiers and 0 to the others, with quota q (floor(n/2) + 1
   // when q <= 0), and the values are averaged over the data points. The game
   // of a data point depends only on its number of correct classifiers, so the
   // dynamic program of Uno (2012) runs once per distinct number.
   const std::vector<double> getWMG(bool shapley, int q = 0)
   {
      return WMG_(shapley, q);
   }

   // The Shapley values of the baseline ISCG, see iscg::shapley
   const std::vector<double> getISCG()
   {
      return iscg::shapley(preds_, labs_);
   }

   int labels() const { return labels_; }


   private:
      std::vector<std::vector<T>> preds_; // the prediction of each base classifier on each data point
      std::vector<T> labs_; // the true label of each data point
      int classes_; // the largest true label
      int labels_; // the number of labels, classes_ + 1
      bool binary_;

      const std::vector<double> BanzhafBinary_(); // ENPOWER_B, Banzhaf values
      const std::vector<double> BanzhafMC_(shapmc::Coefficients coef); // ENPOWER_MC-beta
      const std::vector<double> ShapleyBinary_(); // ENPOWER_B, Shapley values
      const std::vector<double> ShapleyMC_(bool refined, shapmc::Stats* stats, shapmc::Coefficients coef); // ENPOWER_MC-phi and ENPOWER_MC-phi++
      const std::vector<double> WMG_(bool shapley, int q); // the baseline WMG

};

#include "gameSolver.tpp"

#endif //SOLVER_H
