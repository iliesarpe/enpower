"""Reference implementation in Python of ENPOWER_MC-phi (Thm 4.8 and Alg. 4), in exact arithmetic.

Used to validate the C++ implementation (cpp/tools/enpower.py check) and to compute the values
of the prompts in Tables 9-11 (clip_inference.py).
"""
import math
from collections import defaultdict
from fractions import Fraction

# =====================================================================
# 1. The weights of Thm 4.8
# =====================================================================

def printDPMAT(mat):
    for row in mat:
        print(row)

def eta(S, n):
    """The Shapley weight eta(S) = (n-S)!(S-1)!/n! of a coalition of size S, for n players."""
    if S > n or S < 0:
        return Fraction(0)
    num = math.factorial(n - S) * math.factorial(S - 1)
    den = math.factorial(n)
    return Fraction(num, den)

def f_1(k, h, m, M_1, n):
    """
    The coefficient f_1(k, h, m) of Thm 4.8, for a classifier predicting the correct label.
    """
    if M_1 <= 0:  # no correct classifier
        return Fraction(0)
    weight = eta(m + h + 1, n)
    weight2 = eta(m + h, n)
    binom_1 = math.comb(M_1 - 1, h)
    binom_2 = math.comb(M_1 - 1, h - 1)
    
    # exact rational arithmetic
    return Fraction(1, k + 1) * ((k * weight * binom_1) + (weight2 * binom_2))


def f_c(k, h, m, c, M_1, n):
    """
    The coefficient f_j(k, h, m) of Thm 4.8, for a classifier predicting an incorrect label
    with c votes.
    """
    if c <= 0:  
        return Fraction(0)
    weight = eta(m + 2 * h + 1, n)
    weight2 = eta(m + 2 * h, n)
    binom_1 = math.comb(M_1, h)
    term_A = math.comb(c - 1, h)
    term_B = math.comb(c - 1, h - 1)
    
    t1 = Fraction(weight, k + 2) * binom_1 * term_A
    t2 = Fraction(weight2, (k + 1) * (k + 2)) * binom_1 * term_B
    return t1 + t2

# =====================================================================
# 2. AddClass and RemoveClass (Alg. 6 and 7), in place
# =====================================================================

def add_class_inplace(P_h, c, h, max_ties, max_votes):
    """AddClass: multiplies the polynomial P_h(t, u) by the factor of a label with c votes."""
    for k in range(max_ties, -1, -1):
        for m in range(max_votes, -1, -1):
            if P_h[k][m] > 0:
                # the label ties at h votes
                if c >= h:
                    P_h[k + 1][m + h] += P_h[k][m] * math.comb(c, h)
                
                # the label receives q < h votes
                q_max = min(c, h - 1)
                for q in range(q_max, 0, -1):
                    P_h[k][m + q] += P_h[k][m] * math.comb(c, q)
    return P_h

def remove_class_inplace(P_h, c, h, L, M_inc):
    """RemoveClass: divides the polynomial P_h(t, u) by the factor of a label with c votes."""
    q_max = min(c, h - 1)
    
    for k in range(L):
        for m in range(M_inc + 1):
            
            # inverse of the term of a tie at h votes
            if c >= h and k >= 1 and m >= h:
                P_h[k][m] -= P_h[k - 1][m - h] * math.comb(c, h)
                
            # inverse of the terms with q < h votes
            for q in range(1, q_max + 1):
                if m >= q:
                    P_h[k][m] -= P_h[k][m - q] * math.comb(c, q)
                    
    return P_h

def compute_coeffs_dp(M_counts, h, L, M_inc):
    """ComputeCoeffsDP (Alg. 5): the matrix of the coefficients of P_h(t, u)."""
    P_h = [[Fraction(0) for _ in range(M_inc + 1)] for _ in range(L)]
    P_h[0][0] = 1
    
    current_max_votes = 0
    # M_counts: the numbers of votes of the incorrect labels
    for j_idx, c in enumerate(M_counts):
        add_class_inplace(P_h, c, h, j_idx, current_max_votes)
        current_max_votes += c
        
    return P_h

# =====================================================================
# 3. ENPOWER_MC-phi (Alg. 4)
# =====================================================================
def compute_shapley_multiclass(dataset_predictions, dataset_labels, L):
    """
    Exact Shapley values of the base classifiers in the multiclass game, averaged over the
    data points. The O(L n^3) computation runs once per unique vote profile.
    
    :param dataset_predictions: 2D array of shape (num_samples, num_models)
    :param dataset_labels: 1D array of shape (num_samples,) with true labels
    :param L: Integer, total number of unique classes/labels in the task.
    :return: List of size `num_models` with the dataset-averaged Shapley values.
    """
    num_samples = len(dataset_labels)
    num_models = len(dataset_predictions[0])
    n = num_models
    
    # sum of the Shapley values of each classifier over the data points
    shap_global = [Fraction(0)] * num_models
    
    # cache: unique vote profile -> Shapley values
    # Key: (M_1, tuple(sorted(incorrect_sizes)))
    # Value: {'correct': float, 'incorrect': {size: float}}
    profile_cache = {}
    
    for sample_idx in range(num_samples):
        y_true = dataset_labels[sample_idx]
        preds = dataset_predictions[sample_idx]
        
        # 1. number of classifiers predicting each label
        counts = {label: 0 for label in range(L)}
        for p in preds:
            counts[p] += 1
            
        M_1 = counts[y_true]
        
        # the sorted non-zero counts of the incorrect labels, the key of the vote profile
        M_counts = sorted([counts[label] for label in range(L) if label != y_true and counts[label] > 0])
        profile_key = (M_1, tuple(M_counts))
        
        # 2. values of the vote profile, from the cache or computed
        if profile_key not in profile_cache:
            
            # initial values of the vote profile
            '''
            shap_correct_val = 1.0 / n
            # by symmetry, the value of an incorrect classifier depends only on the votes of its label
            shap_incorrect_dict = {c: 0.0 for c in set(M_counts)}
            '''
            shap_correct_val = Fraction(1, n)
            shap_incorrect_dict = {c: Fraction(0) for c in set(M_counts)}
            
            M_inc = sum(M_counts)
            max_wrong = max(M_counts) if M_counts else 0
            
            # 3. loop over the number h of votes of a tie
            for h in range(1, M_1 + 1):
                if h > max_wrong:
                    break
                
                # the matrix of the coefficients of P_h(t, u)
                P_h = compute_coeffs_dp(M_counts, h, L, M_inc)
                
                # --- value of a correct classifier ---
                for k in range(1, L):
                    for m in range(k * h, n - M_1 + 1):
                        shap_correct_val += f_1(k, h, m, M_1, n) * P_h[k][m]
                        
                # --- value of an incorrect classifier, per number of votes of its label ---
                for c in set(M_counts):
                    v_c = Fraction(0)
                    if c >= h:
                        # remove one label with this number of votes
                        P_h = remove_class_inplace(P_h, c, h, L, M_inc)
                        
                        # the term of the label
                        for k in range(0, L - 1):
                            for m in range(k * h, n - (M_1 + c) + 1):
                                v_c -= f_c(k, h, m, c, M_1, n) * P_h[k][m]
                                
                        # add the label back, to restore P_h(t, u)
                        # (with L - 1 rows, so that the last row is kept)
                        P_h = add_class_inplace(P_h, c, h, L - 2, M_inc - c)
                        
                    # store the value for this number of votes
                    shap_incorrect_dict[c] += v_c
                    
            # store the values of the vote profile
            profile_cache[profile_key] = {
                'correct': shap_correct_val,
                'incorrect': shap_incorrect_dict
            }
            
        # 4. add the values of the data point to the values of the classifiers
        cached_payouts = profile_cache[profile_key]
        
        for i in range(num_models):
            y_model = preds[i]
            
            if y_model == y_true:
                shap_global[i] += cached_payouts['correct']
            else:
                c_size = counts[y_model]
                shap_global[i] += cached_payouts['incorrect'][c_size]
                
    # average over the data points
    return [float(val / num_samples) for val in shap_global]


