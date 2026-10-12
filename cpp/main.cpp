#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <cmath>
#include <string>
#include <vector>
#include "gameSolver.h"
#include "sampling.h"

// cpp/solver: the command-line interface of the library. Every mode writes one
// JSON object to stdout; all other output goes to stderr.
//
// Input format, for every mode that reads a dataset (from FILE, or from stdin
// when --input is absent):
//
//     points models labels
//     <true label> <label predicted by each base classifier>   (one line per data point)
//
//   ./solver shapley [--input FILE] [--baseline] [--refined] [--binary] [--float] [--sample K] [--seed S]
//       Shapley values. --baseline runs ENPOWER_MC-phi, --refined runs
//       ENPOWER_MC-phi++, and without either flag both run. --binary runs the
//       closed form of ENPOWER_B instead, for data with two classes. --sample K
//       measures the runtime on K of the unique vote profiles and extrapolates
//       the runtime on the whole dataset.
//
//   ./solver banzhaf [--input FILE] [--binary] [--float]
//       Banzhaf values: ENPOWER_MC-beta, or with --binary the closed form of
//       ENPOWER_B.
//
//   --float (shapley, banzhaf, prune) computes the coefficients of the dynamic
//   programs in long double instead of exact integers, for comparison.
//
//   ./solver prune [--input FILE] [--float]
//       greedy backward elimination with the multiclass Shapley values. Starting
//       from the whole ensemble, it computes the values of the remaining
//       classifiers in the game restricted to them, removes the classifier with
//       the smallest value, and repeats. Reports the remaining classifiers and
//       their values at each step.
//
//   ./solver gen --out FILE [--models n] [--labels L] [--profiles U] [--p P]
//       writes to FILE the synthetic dataset of Appendix G.6, with U unique vote
//       profiles and one data point each: the first U pairs (n_1, r) closest to
//       n_1 = p n, each with the incorrect votes split evenly over r labels (see
//       synthesize). The dataset is deterministic. If fewer than U pairs exist,
//       the instance is reported as "exhausted" and no file is written. The
//       metadata of the instance is written to stdout.
//
//   ./solver verify --input FILE
//       compares the multiclass values of FILE, from both Shapley algorithms,
//       with brute-force computations of the definitions when the vote profiles
//       are small enough, and, on data with two classes, with the closed forms
//       of ENPOWER_B.
//
//   ./solver wmg [--input FILE] [--shapley] [--quota q]
//       the baseline WMG (Uno 2012): Banzhaf, or Shapley, values in the weighted
//       majority game of each data point, averaged over the data points.
//
//   ./solver iscg [--input FILE]
//       the baseline ISCG (Ykhlef and Bouchaffra, IJAIT 2017): Shapley values of
//       the induced subgraph game with the weights of their Eq. (5)-(7).
//
//   ./solver mc [--input FILE] [--shapley] [--samples K] [--seed S]
//       the baselines MC-beta and MC-phi: Banzhaf, or Shapley, values of the
//       game of Eq. (3) estimated from K coalitions per classifier (n^2 when K
//       is absent or <= 0), see mcsample::Game.
//
//   ./solver demo [--points P] [--models M] [--seed S]
//       Banzhaf values of ENPOWER_B on a random dataset.

namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0)
{
   return std::chrono::duration<double>(Clock::now() - t0).count();
}

// --- JSON output ------------------------------------------------------------

void jsonKey(const char* key, int indent = 2)
{
   std::cout << "\n" << std::string(indent, ' ') << "\"" << key << "\": ";
}

// 17 significant digits, independently of the state of the stream: a value
// printed after "seconds" would otherwise inherit std::fixed, and every value
// below 1e-17 would be printed as 0
void jsonArray(const std::vector<double>& v, int indent = 2)
{
   const std::ios_base::fmtflags flags = std::cout.flags();
   std::cout << std::defaultfloat << "[";
   for (size_t i = 0; i < v.size(); ++i)
      std::cout << (i ? ", " : "") << std::setprecision(17) << v[i];
   std::cout << "]";
   std::cout.flags(flags);
   (void)indent;
}

// --- the predictions of one ensemble on a dataset ---------------------------

struct Dataset {
   int points = 0, models = 0, labels = 0;
   std::vector<std::vector<uint32_t>> preds;
   std::vector<uint32_t> labs;
};

bool readDataset(std::istream& in, Dataset& d)
{
   if (!(in >> d.points >> d.models >> d.labels)) return false;
   d.preds.assign(d.points, std::vector<uint32_t>(d.models));
   d.labs.assign(d.points, 0);
   for (int i = 0; i < d.points; ++i) {
      if (!(in >> d.labs[i])) return false;
      for (int j = 0; j < d.models; ++j)
         if (!(in >> d.preds[i][j])) return false;
   }
   return true;
}

// The unique vote profiles, with the same keys as the cache of Solver<T>::ShapleyMC_
std::vector<shapmc::Profile> distinctProfiles(const Dataset& d)
{
   std::map<std::pair<int, std::vector<int>>, shapmc::Profile> seen;
   for (int i = 0; i < d.points; ++i) {
      shapmc::Profile p = shapmc::makeProfile(d.preds[i], d.labs[i], d.labels);
      seen.insert({std::make_pair(p.M1, p.inc), p});
   }
   std::vector<shapmc::Profile> out;
   out.reserve(seen.size());
   for (const auto& e : seen) out.push_back(e.second);
   return out;
}

// The unique vote profiles of the binary game, determined by n_+, as in the
// caches of ShapleyBinary_ and BanzhafBinary_
size_t distinctCounts(const Dataset& d)
{
   std::vector<bool> seen(d.models + 1, false);
   size_t distinct = 0;
   for (int i = 0; i < d.points; ++i) {
      int nPlus = 0;
      for (int j = 0; j < d.models; ++j) nPlus += (d.preds[i][j] == d.labs[i]);
      if (!seen[nPlus]) { seen[nPlus] = true; ++distinct; }
   }
   return distinct;
}

// --- the modes --------------------------------------------------------------

int runDemo(int argc, char** argv)
{
   size_t points = 10000, models = 10000;
   unsigned seed = 100;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--points") && i + 1 < argc) points = std::stoul(argv[++i]);
      else if (!std::strcmp(argv[i], "--models") && i + 1 < argc) models = std::stoul(argv[++i]);
      else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::stoul(argv[++i]);
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }

   std::mt19937 gen(seed);
   std::uniform_int_distribution<uint32_t> dis(0, 1);
   std::vector<std::vector<uint32_t>> preds(points, std::vector<uint32_t>(models));
   std::vector<uint32_t> labs(points);
   for (size_t i = 0; i < points; ++i) {
      labs[i] = dis(gen);
      for (size_t j = 0; j < models; ++j) preds[i][j] = dis(gen);
   }

   Solver<uint32_t> solver(preds, labs);
   auto t0 = Clock::now();
   std::vector<double> values = solver.getBanzhaf(true);
   const double elapsed = since(t0);

   std::sort(values.begin(), values.end(), std::greater<double>());
   values.resize(std::min<size_t>(values.size(), 10));

   std::cout << "{";
   jsonKey("mode");    std::cout << "\"demo\",";
   jsonKey("points");  std::cout << points << ",";
   jsonKey("models");  std::cout << models << ",";
   jsonKey("seconds"); std::cout << std::fixed << std::setprecision(9) << elapsed << ",";
   jsonKey("top_values"); jsonArray(values);
   std::cout << "\n}\n";
   return 0;
}

int runShapley(int argc, char** argv)
{
   std::string input;
   bool baseline = false, refined = false, binary = false;
   int sample = 0;
   unsigned seed = 0;
   shapmc::Coefficients coef = shapmc::Coefficients::Exact;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
      else if (!std::strcmp(argv[i], "--float")) coef = shapmc::Coefficients::Float;
      else if (!std::strcmp(argv[i], "--baseline")) baseline = true;
      else if (!std::strcmp(argv[i], "--refined")) refined = true;
      else if (!std::strcmp(argv[i], "--binary")) binary = true;
      else if (!std::strcmp(argv[i], "--sample") && i + 1 < argc) sample = std::atoi(argv[++i]);
      else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::stoul(argv[++i]);
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }
   if (!baseline && !refined && !binary) baseline = refined = true;

   Dataset d;
   bool ok = false;
   if (input.empty()) {
      ok = readDataset(std::cin, d);
   } else {
      std::ifstream in(input);
      ok = in && readDataset(in, d);
   }
   if (!ok) { std::cerr << "cannot read the dataset\n"; return 1; }

   const std::vector<shapmc::Profile> profiles = distinctProfiles(d);

   std::cout << "{";
   jsonKey("mode");     std::cout << "\"shapley\",";
   jsonKey("points");   std::cout << d.points << ",";
   jsonKey("models");   std::cout << d.models << ",";
   jsonKey("labels");   std::cout << d.labels << ",";
   jsonKey("profiles"); std::cout << profiles.size() << ",";
   if (binary) {        // in the binary game the cache key is n_+
      jsonKey("binary_profiles"); std::cout << distinctCounts(d) << ",";
   }
   jsonKey("coefficients");
   std::cout << (coef == shapmc::Coefficients::Float ? "\"float\"," : "\"exact\",");

   if (sample > 0) {
      // measure the runtime on a subset of the vote profiles and extrapolate
      const int total = static_cast<int>(profiles.size());
      const int k = std::min(sample, total);
      std::vector<int> idx(total);
      for (int i = 0; i < total; ++i) idx[i] = i;
      std::shuffle(idx.begin(), idx.end(), std::mt19937(seed));
      idx.resize(k);

      jsonKey("sampled"); std::cout << k << ",";
      const char* names[3] = {"baseline", "refined", "binary"};
      const bool want[3] = {baseline, refined, binary};
      bool printed = false;
      for (int a = 0; a < 3; ++a) {
         if (!want[a]) continue;
         auto t0 = Clock::now();
         for (int i : idx) {
            if (a == 2) shapbin::payoutsFor(profiles[i].n, profiles[i].M1);
            else        shapmc::payoutsFor(profiles[i], a != 0, nullptr, coef);
         }
         const double elapsed = since(t0);
         if (printed) std::cout << ",";
         jsonKey(names[a]);
         std::cout << std::fixed << std::setprecision(9)
                   << "{\"sample_seconds\": " << elapsed
                   << ", \"seconds\": " << elapsed * total / k << "}";
         printed = true;
      }
      std::cout << "\n}\n";
      return 0;
   }

   Solver<uint32_t> solver(d.preds, d.labs);
   const char* names[3] = {"baseline", "refined", "binary"};
   const bool want[3] = {baseline, refined, binary};
   bool printed = false;
   for (int a = 0; a < 3; ++a) {
      if (!want[a]) continue;
      auto t0 = Clock::now();
      const std::vector<double> values = (a == 2) ? solver.getShapleyBinary()
                                                  : solver.getShapleyMC(a != 0, nullptr, coef);
      const double elapsed = since(t0);
      if (printed) std::cout << ",";
      jsonKey(names[a]);
      std::cout << std::fixed << std::setprecision(9)
                << "{\"seconds\": " << elapsed << ", \"values\": ";
      jsonArray(values);
      std::cout << "}";
      printed = true;
   }
   std::cout << "\n}\n";
   return 0;
}

// The Banzhaf values on a dataset: ENPOWER_MC-beta, or the closed form of
// ENPOWER_B. The output has the same keys as shapley.
int runBanzhaf(int argc, char** argv)
{
   std::string input;
   bool binary = false;
   shapmc::Coefficients coef = shapmc::Coefficients::Exact;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
      else if (!std::strcmp(argv[i], "--float")) coef = shapmc::Coefficients::Float;
      else if (!std::strcmp(argv[i], "--binary")) binary = true;
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }

   Dataset d;
   bool ok = false;
   if (input.empty()) {
      ok = readDataset(std::cin, d);
   } else {
      std::ifstream in(input);
      ok = in && readDataset(in, d);
   }
   if (!ok) { std::cerr << "cannot read the dataset\n"; return 1; }

   Solver<uint32_t> solver(d.preds, d.labs);
   auto t0 = Clock::now();
   const std::vector<double> values = solver.getBanzhaf(binary, coef);
   const double elapsed = since(t0);

   std::cout << "{";
   jsonKey("mode");     std::cout << "\"banzhaf\",";
   jsonKey("points");   std::cout << d.points << ",";
   jsonKey("models");   std::cout << d.models << ",";
   jsonKey("labels");   std::cout << d.labels << ",";
   jsonKey("profiles"); std::cout << distinctProfiles(d).size() << ",";
   if (binary) {        // in the binary game the cache key is n_+
      jsonKey("binary_profiles"); std::cout << distinctCounts(d) << ",";
   }
   jsonKey("game");     std::cout << (binary ? "\"binary\"," : "\"multiclass\",");
   jsonKey("coefficients");
   std::cout << (coef == shapmc::Coefficients::Float ? "\"float\"," : "\"exact\",");
   jsonKey("seconds"); std::cout << std::fixed << std::setprecision(9) << elapsed << ",";
   jsonKey("values");  jsonArray(values);
   std::cout << "\n}\n";
   return 0;
}

// A dataset from FILE, or from stdin when FILE is empty
bool loadDataset(const std::string& input, Dataset& d)
{
   if (input.empty()) return readDataset(std::cin, d);
   std::ifstream in(input);
   return in && readDataset(in, d);
}

// The baseline WMG (Uno 2012) on a dataset: the Banzhaf values, or with
// --shapley the Shapley values. --quota q replaces the default quota
// floor(n/2) + 1. The output has the same keys as banzhaf.
int runWMG(int argc, char** argv)
{
   std::string input;
   bool shapley = false;
   int q = 0;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
      else if (!std::strcmp(argv[i], "--shapley")) shapley = true;
      else if (!std::strcmp(argv[i], "--quota") && i + 1 < argc) q = std::atoi(argv[++i]);
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }
   Dataset d;
   if (!loadDataset(input, d)) { std::cerr << "cannot read the dataset\n"; return 1; }
   if (q <= 0) q = d.models / 2 + 1;

   Solver<uint32_t> solver(d.preds, d.labs);
   auto t0 = Clock::now();
   const std::vector<double> values = solver.getWMG(shapley, q);
   const double elapsed = since(t0);

   std::cout << "{";
   jsonKey("mode");     std::cout << "\"wmg\",";
   jsonKey("index");    std::cout << (shapley ? "\"shapley\"," : "\"banzhaf\",");
   jsonKey("points");   std::cout << d.points << ",";
   jsonKey("models");   std::cout << d.models << ",";
   jsonKey("labels");   std::cout << d.labels << ",";
   jsonKey("quota");    std::cout << q << ",";
   jsonKey("binary_profiles"); std::cout << distinctCounts(d) << ",";
   jsonKey("seconds"); std::cout << std::fixed << std::setprecision(9) << elapsed << ",";
   jsonKey("values");  jsonArray(values);
   std::cout << "\n}\n";
   return 0;
}

// The baseline ISCG (Ykhlef and Bouchaffra, IJAIT 2017) on a dataset: the
// Shapley values of the induced subgraph game.
int runISCG(int argc, char** argv)
{
   std::string input;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }
   Dataset d;
   if (!loadDataset(input, d)) { std::cerr << "cannot read the dataset\n"; return 1; }

   Solver<uint32_t> solver(d.preds, d.labs);
   auto t0 = Clock::now();
   const std::vector<double> values = solver.getISCG();
   const double elapsed = since(t0);

   std::cout << "{";
   jsonKey("mode");     std::cout << "\"iscg\",";
   jsonKey("points");   std::cout << d.points << ",";
   jsonKey("models");   std::cout << d.models << ",";
   jsonKey("labels");   std::cout << d.labels << ",";
   jsonKey("seconds"); std::cout << std::fixed << std::setprecision(9) << elapsed << ",";
   jsonKey("values");  jsonArray(values);
   std::cout << "\n}\n";
   return 0;
}

// The Monte-Carlo baselines on a dataset. The output has the same keys as
// banzhaf, together with the sample size and the seed.
int runMC(int argc, char** argv)
{
   std::string input;
   bool shapley = false;
   long long samples = 0;
   unsigned long long seed = 0;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
      else if (!std::strcmp(argv[i], "--shapley")) shapley = true;
      else if (!std::strcmp(argv[i], "--samples") && i + 1 < argc) samples = std::atoll(argv[++i]);
      else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::stoull(argv[++i]);
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }
   Dataset d;
   if (!loadDataset(input, d)) { std::cerr << "cannot read the dataset\n"; return 1; }
   if (samples <= 0) samples = static_cast<long long>(d.models) * d.models;

   auto t0 = Clock::now();
   const mcsample::Game game(d.preds, d.labs);
   std::mt19937_64 gen(seed);
   const std::vector<double> values = game.estimate(shapley, samples, gen);
   const double elapsed = since(t0);

   std::cout << "{";
   jsonKey("mode");     std::cout << "\"mc\",";
   jsonKey("index");    std::cout << (shapley ? "\"shapley\"," : "\"banzhaf\",");
   jsonKey("points");   std::cout << d.points << ",";
   jsonKey("models");   std::cout << d.models << ",";
   jsonKey("labels");   std::cout << d.labels << ",";
   jsonKey("distinct_points"); std::cout << game.distinct() << ",";
   jsonKey("samples");  std::cout << samples << ",";
   jsonKey("seed");     std::cout << seed << ",";
   jsonKey("seconds"); std::cout << std::fixed << std::setprecision(9) << elapsed << ",";
   jsonKey("values");  jsonArray(values);
   std::cout << "\n}\n";
   return 0;
}

// Writes a JSON array of integers
void jsonInts(const std::vector<int>& v)
{
   std::cout << "[";
   for (size_t i = 0; i < v.size(); ++i) std::cout << (i ? ", " : "") << v[i];
   std::cout << "]";
}

int runPrune(int argc, char** argv)
{
   std::string input;
   shapmc::Coefficients coef = shapmc::Coefficients::Exact;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
      else if (!std::strcmp(argv[i], "--float")) coef = shapmc::Coefficients::Float;
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }

   Dataset d;
   bool ok = false;
   if (input.empty()) {
      ok = readDataset(std::cin, d);
   } else {
      std::ifstream in(input);
      ok = in && readDataset(in, d);
   }
   if (!ok) { std::cerr << "cannot read the dataset\n"; return 1; }

   // the remaining classifiers, as indices in the whole ensemble
   std::vector<int> alive(d.models);
   for (int j = 0; j < d.models; ++j) alive[j] = j;

   std::cout << "{";
   jsonKey("mode");   std::cout << "\"prune\",";
   jsonKey("points"); std::cout << d.points << ",";
   jsonKey("models"); std::cout << d.models << ",";
   jsonKey("labels"); std::cout << d.labels << ",";
   jsonKey("coefficients");
   std::cout << (coef == shapmc::Coefficients::Float ? "\"float\"," : "\"exact\",");
   jsonKey("steps"); std::cout << "[";

   auto t0 = Clock::now();
   std::vector<int> dropped; // elimination order, smallest value first
   bool first = true;
   while (alive.size() > 1) {
      // the game restricted to the remaining classifiers
      std::vector<std::vector<uint32_t>> preds(d.points,
                                               std::vector<uint32_t>(alive.size()));
      for (int i = 0; i < d.points; ++i)
         for (size_t j = 0; j < alive.size(); ++j)
            preds[i][j] = d.preds[i][alive[j]];

      Solver<uint32_t> solver(preds, d.labs);
      const std::vector<double> values = solver.getShapleyMC(true, nullptr, coef);

      const size_t worst =
         std::min_element(values.begin(), values.end()) - values.begin();

      std::cout << (first ? "" : ",") << "\n    {\"k\": " << alive.size()
                << ", \"survivors\": ";
      jsonInts(alive);
      std::cout << ", \"values\": ";
      jsonArray(values);
      std::cout << ", \"dropped\": " << alive[worst] << "}";
      first = false;

      dropped.push_back(alive[worst]);
      alive.erase(alive.begin() + worst);
   }
   // the last remaining classifier, whose game has a single player
   std::cout << (first ? "" : ",") << "\n    {\"k\": 1, \"survivors\": ";
   jsonInts(alive);
   std::cout << ", \"values\": [], \"dropped\": " << alive[0] << "}";
   const double elapsed = since(t0);

   std::cout << "\n  ],";
   jsonKey("order"); // the remaining classifiers for each k, in reverse elimination order
   dropped.push_back(alive[0]);
   std::reverse(dropped.begin(), dropped.end());
   jsonInts(dropped);
   std::cout << ",";
   jsonKey("seconds"); std::cout << std::fixed << std::setprecision(9) << elapsed;
   std::cout << "\n}\n";
   return 0;
}

// --- synthetic instances with a prescribed number of vote profiles ----------

struct SynthInstance {
   Dataset d;
   int available = 0;        // number of pairs (n_1, r) that exist, i.e., the largest possible |U|
   double grand = 0;         // v(N), averaged over the data points: the sum of the Shapley values
   bool exhausted = false;   // fewer than |U| pairs exist; no instance is built
};

// r_max(n_1) = min{L - 1, n - n_1}, since there are L - 1 incorrect labels and
// n - n_1 incorrect votes
int rMax(int n, int L, int n1) { return std::min(L - 1, n - n1); }

// The instance of the synthetic experiments (Appendix G.6, "Data generation").
// A vote profile is a pair (n_1, r): n_1 votes for the correct label 0, and the
// n - n_1 incorrect votes split as evenly as possible over the labels 1..r, the
// first (n - n_1) mod r of which receive one additional vote. The profiles are,
// in order:
//
//   1. n_1^* = the integer closest to p n;
//   2. n_1 runs over n_1^*, n_1^* - 1, n_1^* + 1, n_1^* - 2, n_1^* + 2, ...,
//      skipping the values outside [0, n];
//   3. each n_1 contributes (n_1, r_max(n_1)), (n_1, r_max(n_1) - 1), ..., (n_1, 1),
//      or the single (n, 0) when n_1 = n;
//
// and the instance consists of the first |U| of them, one data point each, so
// |D| = |U|. Distinct pairs have distinct sorted counts, and are therefore
// distinct keys of the caches of Solver<T>. The instance is deterministic.
SynthInstance synthesize(int n, int L, int profiles, double p)
{
   SynthInstance s;
   s.d.models = n;
   s.d.labels = L;
   for (int n1 = 0; n1 <= n; ++n1) s.available += (n1 == n) ? 1 : rMax(n, L, n1);
   if (s.available < profiles) { s.exhausted = true; return s; }

   const int star = static_cast<int>(std::floor(p * n + 0.5));
   std::vector<int> order{star};
   for (int dist = 1; dist <= n; ++dist) {
      if (star - dist >= 0) order.push_back(star - dist);
      if (star + dist <= n) order.push_back(star + dist);
   }

   for (int n1 : order) {
      const int top = (n1 == n) ? 0 : rMax(n, L, n1);
      for (int r = top; r >= (n1 == n ? 0 : 1); --r) {
         if (s.d.points == profiles) break;
         const int wrong = n - n1;
         const int base = r ? wrong / r : 0, extra = r ? wrong % r : 0;

         // the data point: classifiers 0..n_1-1 predict label 0, the others labels 1..r
         std::vector<uint32_t> row;
         row.reserve(n);
         row.insert(row.end(), n1, 0u);
         for (int j = 1; j <= r; ++j)
            row.insert(row.end(), base + (j <= extra), static_cast<uint32_t>(j));
         s.d.preds.push_back(std::move(row));
         s.d.labs.push_back(0u);
         ++s.d.points;

         // v(N): the correct label has the most votes, possibly tied, if n_1 is
         // at least the largest incorrect count, base + 1 (or base when r divides
         // n - n_1)
         const int topWrong = r ? base + (extra > 0) : 0;
         const int atTop = !r ? 0 : (extra > 0 ? extra : r);
         if (n1 > topWrong)       s.grand += 1.0;
         else if (n1 == topWrong) s.grand += 1.0 / (1 + atTop);
      }
      if (s.d.points == profiles) break;
   }
   s.grand /= s.d.points;
   return s;
}

// The values on a dataset from brute-force computations of the definitions,
// assigned to the classifiers as in Solver<T>. Empty when a vote profile is
// too large for the brute-force computation.
std::vector<double> bruteValues(const Dataset& d, bool shapley)
{
   std::vector<double> out(d.models, 0.0);
   std::vector<int> counts(d.labels);
   for (int i = 0; i < d.points; ++i) {
      const shapmc::Profile p = shapmc::makeProfile(d.preds[i], d.labs[i], d.labels);
      double cost = p.M1 + 1.0;
      for (int c : p.inc) cost *= c + 1.0;
      if (cost * (p.inc.size() + 1.0) > 2e7) return {};

      shapmc::Real correct;
      std::map<int, shapmc::Real> incorrect;
      if (shapley) { auto b = shapmc::payoutsBruteForce(p); correct = b.correct; incorrect = b.incorrect; }
      else         { auto b = bzmc::payoutsBruteForce(p);   correct = b.correct; incorrect = b.incorrect; }

      std::fill(counts.begin(), counts.end(), 0);
      for (uint32_t v : d.preds[i]) ++counts[v];
      for (int j = 0; j < d.models; ++j) {
         const uint32_t v = d.preds[i][j];
         out[j] += static_cast<double>(v == d.labs[i] ? correct : incorrect.at(counts[v]));
      }
   }
   for (double& v : out) v /= d.points;
   return out;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
{
   double m = 0;
   for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
   return m;
}

// Writes a dataset in the input format of readDataset
void writeDataset(std::ostream& out, const Dataset& d)
{
   out << d.points << " " << d.models << " " << d.labels << "\n";
   for (int i = 0; i < d.points; ++i) {
      out << d.labs[i];
      for (uint32_t v : d.preds[i]) out << " " << v;
      out << "\n";
   }
}

// A synthetic instance of |U| unique vote profiles, written to --out; the
// metadata of the instance is written to stdout.
int runGen(int argc, char** argv)
{
   std::string out;
   int n = 50, L = 10, profiles = 100;
   double p = 0.5;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
      else if (!std::strcmp(argv[i], "--models") && i + 1 < argc) n = std::atoi(argv[++i]);
      else if (!std::strcmp(argv[i], "--labels") && i + 1 < argc) L = std::atoi(argv[++i]);
      else if (!std::strcmp(argv[i], "--profiles") && i + 1 < argc) profiles = std::atoi(argv[++i]);
      else if (!std::strcmp(argv[i], "--p") && i + 1 < argc) p = std::atof(argv[++i]);
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }
   if (L < 2 || n < 1 || profiles < 1) { std::cerr << "need n >= 1, L >= 2, |U| >= 1\n"; return 2; }

   auto t0 = Clock::now();
   const SynthInstance s = synthesize(n, L, profiles, p);
   const double genSeconds = since(t0);

   // an instance with fewer than |U| vote profiles is not written
   if (!s.exhausted && !out.empty()) {
      std::ofstream f(out);
      if (!f) { std::cerr << "cannot write " << out << "\n"; return 1; }
      writeDataset(f, s.d);
   }

   std::cout << "{";
   jsonKey("mode");     std::cout << "\"gen\",";
   jsonKey("models");   std::cout << n << ",";
   jsonKey("labels");   std::cout << L << ",";
   jsonKey("p");        std::cout << p << ",";
   jsonKey("target_profiles"); std::cout << profiles << ",";
   jsonKey("available"); std::cout << s.available << ",";
   jsonKey("profiles"); std::cout << s.d.points << ",";
   jsonKey("exhausted"); std::cout << (s.exhausted ? "true" : "false") << ",";
   jsonKey("grand_value"); std::cout << std::setprecision(17) << s.grand << ",";
   jsonKey("gen_seconds"); std::cout << std::fixed << std::setprecision(6) << genSeconds;
   std::cout << "\n}\n";
   return 0;
}

// Compares the exact values on a dataset with brute-force computations of the
// definitions, when the vote profiles are small enough, and, on data with two
// classes, with the closed forms of ENPOWER_B, which compute the same game.
int runVerify(int argc, char** argv)
{
   std::string input;
   for (int i = 0; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
      else { std::cerr << "unknown argument: " << argv[i] << "\n"; return 2; }
   }
   Dataset d;
   std::ifstream in(input);
   if (!in || !readDataset(in, d)) { std::cerr << "cannot read the dataset\n"; return 1; }

   Solver<uint32_t> solver(d.preds, d.labs);
   const std::vector<double> banz = solver.getBanzhaf(false), shap = solver.getShapleyMC(true),
                             shapBase = solver.getShapleyMC(false);

   std::cout << "{";
   jsonKey("mode");   std::cout << "\"verify\",";
   jsonKey("points"); std::cout << d.points << ",";
   jsonKey("models"); std::cout << d.models << ",";
   jsonKey("labels"); std::cout << d.labels;
   auto item = [](const char* k, double v) {
      std::cout << ",";
      jsonKey(k); std::cout << std::scientific << std::setprecision(3) << v << std::fixed;
   };
   const std::vector<double> bb = bruteValues(d, false), bs = bruteValues(d, true);
   if (!bb.empty()) item("banzhaf_vs_brute", maxAbsDiff(banz, bb));
   if (!bs.empty()) {
      item("shapley_vs_brute", maxAbsDiff(shap, bs));
      item("shapley_baseline_vs_brute", maxAbsDiff(shapBase, bs));
   }
   if (d.labels == 2) {
      const std::vector<double> shapBin = solver.getShapleyBinary();
      item("banzhaf_vs_binary", maxAbsDiff(banz, solver.getBanzhaf(true)));
      item("shapley_vs_binary", maxAbsDiff(shap, shapBin));
      item("shapley_baseline_vs_binary", maxAbsDiff(shapBase, shapBin));
   }
   std::cout << "\n}\n";
   return 0;
}


} // namespace

int main(int argc, char** argv)
{
   if (argc < 2) return runDemo(0, argv + 1);
   const std::string mode = argv[1];
   if (mode == "demo")    return runDemo(argc - 2, argv + 2);
   if (mode == "shapley") return runShapley(argc - 2, argv + 2);
   if (mode == "banzhaf") return runBanzhaf(argc - 2, argv + 2);
   if (mode == "prune")   return runPrune(argc - 2, argv + 2);
   if (mode == "gen")     return runGen(argc - 2, argv + 2);
   if (mode == "verify")  return runVerify(argc - 2, argv + 2);
   if (mode == "wmg")     return runWMG(argc - 2, argv + 2);
   if (mode == "iscg")    return runISCG(argc - 2, argv + 2);
   if (mode == "mc")      return runMC(argc - 2, argv + 2);
   std::cerr << "usage: " << argv[0] << " demo|shapley|banzhaf|prune|gen|verify|wmg|iscg|mc [options]\n";
   return 2;
}
