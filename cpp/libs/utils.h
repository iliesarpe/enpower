#ifndef UTILS_H
#define UTILS_H
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <stdexcept>

namespace utilities {

   /**
   * An unsigned integer of 64 W bits, with +, - and * modulo 2^(64 W).
   *
   * The coefficients of P_h(t,u) are numbers of coalitions of a vote profile,
   * bounded by 2^N with N = n - n_1, and RemoveClass divides one polynomial by
   * another. Since division amplifies the error of its input, the
   * coefficients are computed exactly, which long double does not allow.
   *
   * Addition, subtraction and multiplication modulo 2^(64 W) are exact
   * whenever the exact result is below 2^(64 W). This holds for every
   * intermediate value of AddClass and RemoveClass when N < 64 W, including a
   * partial sum of RemoveClass that is temporarily negative.
   */
   template <int W>
   struct WideUInt {
      static_assert(W >= 1, "a wide integer needs at least one word");
      std::uint64_t w[W];

      WideUInt() { for (int i = 0; i < W; ++i) w[i] = 0; }
      WideUInt(int v) {                                   // NOLINT: C(0) and C(1)
         w[0] = static_cast<std::uint64_t>(v);
         for (int i = 1; i < W; ++i) w[i] = 0;
      }

      friend bool operator==(const WideUInt& a, const WideUInt& b) {
         for (int i = 0; i < W; ++i) if (a.w[i] != b.w[i]) return false;
         return true;
      }
      friend bool operator!=(const WideUInt& a, const WideUInt& b) { return !(a == b); }

      WideUInt& operator+=(const WideUInt& o) {
         unsigned __int128 carry = 0;
         for (int i = 0; i < W; ++i) {
            carry += static_cast<unsigned __int128>(w[i]) + o.w[i];
            w[i] = static_cast<std::uint64_t>(carry);
            carry >>= 64;
         }
         return *this;
      }

      WideUInt& operator-=(const WideUInt& o) {
         unsigned __int128 borrow = 0;
         for (int i = 0; i < W; ++i) {
            unsigned __int128 cur = static_cast<unsigned __int128>(w[i])
                                  - o.w[i] - borrow;
            w[i] = static_cast<std::uint64_t>(cur);
            borrow = (cur >> 64) ? 1 : 0;
         }
         return *this;
      }

      friend WideUInt operator+(WideUInt a, const WideUInt& b) { a += b; return a; }
      friend WideUInt operator-(WideUInt a, const WideUInt& b) { a -= b; return a; }

      // a >>= bits, with the words shifted first and the remainder folded in
      WideUInt& operator>>=(int bits) {
         const int words = bits / 64, rem = bits % 64;
         for (int i = 0; i < W; ++i) {
            const std::uint64_t lo = (i + words     < W) ? w[i + words]     : 0;
            const std::uint64_t hi = (i + words + 1 < W) ? w[i + words + 1] : 0;
            w[i] = rem ? ((lo >> rem) | (hi << (64 - rem))) : lo;
         }
         return *this;
      }

      // the number of trailing zero bits, 64 W when the value is zero
      int ctz() const {
         for (int i = 0; i < W; ++i)
            if (w[i]) return i * 64 + __builtin_ctzll(w[i]);
         return 64 * W;
      }

      friend WideUInt operator*(const WideUInt& a, const WideUInt& b) {
         WideUInt r;
         for (int i = 0; i < W; ++i) {
            if (a.w[i] == 0) continue;
            std::uint64_t carry = 0;
            for (int j = 0; i + j < W; ++j) {
               unsigned __int128 cur = static_cast<unsigned __int128>(a.w[i]) * b.w[j]
                                     + r.w[i + j] + carry;
               r.w[i + j] = static_cast<std::uint64_t>(cur);
               carry = static_cast<std::uint64_t>(cur >> 64);
            }
         }
         return r;
      }

      explicit operator long double() const {
         long double v = 0.0L;
         for (int i = W - 1; i >= 0; --i)
            v = v * 18446744073709551616.0L + static_cast<long double>(w[i]);
         return v;
      }
   };
   /**
   * a / b, assuming that b divides a and that both are below 2^(64 W), as in
   * the division of polynomials with integer coefficients. b is factored into
   * 2^s and an odd factor, which is invertible modulo 2^(64 W), so the
   * quotient is a multiplication by that inverse. The inverse is computed with
   * Newton's iteration x <- x (2 - b x), which doubles the number of correct
   * bits at each step.
   */
   template <int W>
   inline WideUInt<W> divExact(WideUInt<W> a, WideUInt<W> b) {
      const int s = b.ctz();
      if (s >= 64 * W) return WideUInt<W>(0);        // not reached: the library never divides by zero
      a >>= s;                                       // b divides a, so these bits are zero
      b >>= s;
      std::uint64_t inv64 = 1;                       // correct modulo 2, b is odd
      for (int i = 0; i < 6; ++i) inv64 *= 2u - b.w[0] * inv64;   // 2, 4, ..., 64 bits
      WideUInt<W> inv;
      inv.w[0] = inv64;
      const WideUInt<W> two(2);
      for (int bits = 64; bits < 64 * W; bits *= 2) inv = inv * (two - b * inv);
      return a * inv;
   }

   /**
   * Computes P(X = k) for X ~ Binomial(n, 0.5).
   */
   inline double binomial_pmf(int n, int k) {
       if (n < 0) throw std::invalid_argument("n must be non-negative");
       if (k < 0 || k > n) return 0.0;

       double log_comb = std::lgamma(n + 1)
                       - std::lgamma(k + 1)
                       - std::lgamma(n - k + 1);

       return std::exp(log_comb - n * std::log(2.0)); // Since p = 0.5
   }

   /**
   * log of the binomial coefficient C(a, b), -infinity when the coefficient is 0.
   * The Shapley weights of a large ensemble overflow every fixed-width type,
   * so the values are computed in log scale.
   */
   inline long double log_binom(int a, int b) {
       if (a < 0 || b < 0 || b > a) return -std::numeric_limits<long double>::infinity();
       return std::lgamma(static_cast<long double>(a) + 1.0L)
            - std::lgamma(static_cast<long double>(b) + 1.0L)
            - std::lgamma(static_cast<long double>(a - b) + 1.0L);
   }

   /**
   * The binomial coefficient C(a, b) as a long double, rounded to the nearest
   * integer whenever it is small enough to be represented exactly.
   */
   inline long double binom(int a, int b) {
       long double lb = log_binom(a, b);
       if (lb == -std::numeric_limits<long double>::infinity()) return 0.0L;
       long double v = std::exp(lb);
       return (lb < 43.6L) ? std::round(v) : v; // 43.6 > log(2^63)
   }

   /**
   * eta(a) = (n-a)!(a-1)!/n! = 1/(n C(n-1, a-1)), the Shapley weight of a
   * coalition of size a, and 0 outside the domain [1, n].
   */
   inline long double eta(int a, int n) {
       if (a < 1 || a > n) return 0.0L;
       return std::exp(-std::log(static_cast<long double>(n)) - log_binom(n - 1, a - 1));
   }

   /**
   * The binomial coefficient C(a, b) as an exact 128-bit integer, for the
   * exact coefficients of P_h(t,u). Saturates at 2^127 and reports it.
   */
   inline __int128 binom_exact(int a, int b, bool* overflow = nullptr) {
       if (a < 0 || b < 0 || b > a) return 0;
       static std::vector<std::vector<__int128>> table;
       static bool saturated = false;
       if (static_cast<int>(table.size()) <= a) {
          const __int128 kMax = static_cast<__int128>((~static_cast<unsigned __int128>(0)) >> 1);
          int old = static_cast<int>(table.size());
          table.resize(a + 1);
          for (int i = old; i <= a; ++i) {
             table[i].assign(i + 1, 0);
             table[i][0] = 1;
             for (int j = 1; j <= i; ++j) {
                __int128 left  = table[i - 1][j - 1];
                __int128 right = (j <= i - 1) ? table[i - 1][j] : 0;
                if (left > kMax - right) { saturated = true; table[i][j] = kMax; }
                else                     { table[i][j] = left + right; }
             }
          }
       }
       if (overflow && saturated) *overflow = true;
       return table[a][b];
   }

   /**
   * The binomial coefficient C(a, b) as an exact integer of 64 W bits, from a
   * Pascal triangle computed once and cached. Since C(a, b) <= 2^a, the
   * triangle is exact for a < 64 W.
   */
   template <int W>
   inline const WideUInt<W>& binom_wide(int a, int b) {
       static std::vector<std::vector<WideUInt<W>>> table;
       static const WideUInt<W> zero;
       if (a < 0 || b < 0 || b > a) return zero;
       if (static_cast<int>(table.size()) <= a) {
          int old = static_cast<int>(table.size());
          table.resize(a + 1);
          for (int i = old; i <= a; ++i) {
             table[i].assign(i + 1, WideUInt<W>(0));
             table[i][0] = WideUInt<W>(1);
             for (int j = 1; j <= i; ++j) {
                table[i][j] = table[i - 1][j - 1];
                if (j <= i - 1) table[i][j] += table[i - 1][j];
             }
          }
       }
       return table[a][b];
   }
}

#endif //UTILS_H
