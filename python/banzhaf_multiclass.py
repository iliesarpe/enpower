"""Reference implementation in Python of ENPOWER_MC-beta (Thm 4.6 and Alg. 3), in exact arithmetic.

Used only to validate the C++ implementation (cpp/tools/enpower.py banzhaf).
"""
import numpy as np
from fractions import Fraction
from decimal import Decimal, getcontext

getcontext().prec = 50 

def compute_banzhaf_multiclass(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    num_samples, num_models = M.shape
    beta = np.zeros(num_models, dtype=np.float64)
    
    decimal_factor = Decimal(2 ** (num_models - 1))
    
    # cache: vote profile -> (value of a correct classifier, {votes v of an incorrect label -> value})
    dp_cache = {}
    
    for i in range(num_samples):
        y_true = int(y[i])
        preds = M[i]
        
        # number of votes of each label, in O(n) time
        counts = np.bincount(preds)
        n_1 = int(counts[y_true]) if y_true < len(counts) else 0
        
        incorrect_counts = [int(count) for c, count in enumerate(counts) 
                            if c != y_true and count > 0]
        
        # the vote profile, independent of the order of the labels
        signature = (n_1, tuple(sorted(incorrect_counts)))
        
        # --- vote profile already computed ---
        if signature in dp_cache:
            beta_correct_float, beta_inc_float = dp_cache[signature]
        else:
            # --- new vote profile: exact dynamic program ---
            L_active = len(incorrect_counts)
            
            unique_n_inc_counts = {}
            for count in incorrect_counts:
                unique_n_inc_counts[count] = unique_n_inc_counts.get(count, 0) + 1
                
            beta_correct = Fraction(1)
            beta_inc = {v: Fraction(0) for v in unique_n_inc_counts}
            
            if n_1 > 0:
                C1 = n_1 - 1
                C2 = 1
                C5 = n_1
                
                A_dict = {v: 1 for v in unique_n_inc_counts}
                B_dict = {v: v for v in unique_n_inc_counts}
                C3_dict = {v: max(0, v - 1) for v in unique_n_inc_counts}
                C4_dict = {v: 1 for v in unique_n_inc_counts}
                
                for h in range(1, n_1 + 1):
                    P = [0] * (L_active + 1)
                    P[0] = 1
                    for v, freq in unique_n_inc_counts.items():
                        Av = A_dict[v]
                        Bv = B_dict[v]
                        for _ in range(freq):
                            for k in range(L_active, 0, -1):
                                P[k] = P[k] * Av + P[k - 1] * Bv
                            P[0] *= Av
                    
                    term_correct = Fraction(0)
                    for k in range(1, L_active + 1):
                        term_correct += Fraction((k * C1 + C2), k + 1) * P[k]
                    beta_correct += term_correct
                    
                    for v in unique_n_inc_counts:
                        Av = A_dict[v]
                        Bv = B_dict[v]
                        C3 = C3_dict[v]
                        C4 = C4_dict[v]
                        
                        Q = [0] * L_active
                        Q[0] = P[0] // Av
                        for k in range(1, L_active):
                            Q[k] = (P[k] - Bv * Q[k - 1]) // Av
                            
                        sum_term = Fraction(0)
                        for k in range(L_active):
                            num = (C3 * (k + 1) + C4) * C5 * Q[k]
                            den = (k + 1) * (k + 2)
                            sum_term += Fraction(num, den)
                            
                        beta_inc[v] -= sum_term
                        
                    if h < n_1:
                        C2 = C1
                        C1 = C1 * (n_1 - 1 - h) // (h + 1) if h < n_1 - 1 else 0
                        C5 = C5 * (n_1 - h) // (h + 1)
                        
                        for v in unique_n_inc_counts:
                            A_dict[v] += B_dict[v]
                            B_dict[v] = B_dict[v] * (v - h) // (h + 1) if h < v else 0
                            tmp_C3 = C3_dict[v]
                            C4_dict[v] = tmp_C3
                            C3_dict[v] = tmp_C3 * (v - 1 - h) // (h + 1) if h < v - 1 else 0
                            
            # convert the exact values to floats
            beta_correct_float = float(Decimal(int(beta_correct.numerator)) / 
                                       Decimal(int(beta_correct.denominator)) / decimal_factor)
            
            beta_inc_float = {}
            for v, frac_val in beta_inc.items():
                beta_inc_float[v] = float(Decimal(int(frac_val.numerator)) / 
                                          Decimal(int(frac_val.denominator)) / decimal_factor)
                
            # store the values of the vote profile
            dp_cache[signature] = (beta_correct_float, beta_inc_float)
                
        # --- add the values of the data point to the classifiers, in O(n) time ---
        for m in range(num_models):
            pred = preds[m]
            if pred == y_true:
                beta[m] += beta_correct_float
            else:
                v = int(counts[pred]) 
                if v in beta_inc_float:
                    beta[m] += beta_inc_float[v]

    return beta / num_samples
