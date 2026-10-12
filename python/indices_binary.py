"""Reference implementation in Python of the closed forms of ENPOWER_B (Thm 4.1 and 4.2).

Used only to validate the C++ implementation (cpp/tools/enpower.py binary).
"""
import numpy as np
from scipy.stats import binom
import scipy
import math

def compute_shap_summation(n, a, b, n1, n2, iscorrect=False):
    """The sum over the ties of the closed form of the Shapley value (Thm 4.2), with first term a
    and b terms; n1 and n2 are the numbers of votes of the label of the classifier and of the
    other label."""
    if n1 == 0:
        return 0
    shap_corr = 0
    if b > 0:  # the sum is empty for b <= 0
        weight = (3*n1 + n2 - 4)
        current_i = n2/(n*(n-1)*(n-2))
        shap_corr = current_i * weight
        for h in range(1,b):
            numerator = 2 * (n1-h) * (n2-h)*(2*h+1)
            denumerator = (h+1)*(n-2*h-1)*(n-2*h-2)
            if denumerator == 0:
                print("Should not happend")
                break
            ratio = numerator/denumerator
            current_i *= ratio 
            weight -= 4
            shap_corr += current_i*weight
    c = 1./n if iscorrect else 0
    shap_corr = c + 0.5*(a + shap_corr)
    return shap_corr

def get_banzaf(dataset):
    """Banzhaf values of ENPOWER_B, averaged over the data points.

    dataset[i][j] is 1 if classifier j is correct on data point i, and 0 otherwise.
    """
    cache = {}
    n = len(dataset) # number of data points
    m = len(dataset[0]) # number of base classifiers
    banzaf_vec = np.zeros(m)
    rv = binom(m, .5)
    for i in range(n):
        predictions = dataset[i]
        n_plus = sum(predictions)
        contribution = 0
        if n_plus in cache.keys():
            contribution = cache[n_plus]
        else:
            contribution = rv.pmf(n_plus)
            cache[n_plus] = contribution
        update = np.where(predictions == 1, contribution, -contribution)
        banzaf_vec += update

    order = np.log(n) - m * np.log(2)
    missingTerm = np.zeros(m)
    if math.pow(math.e, order) > 1e-50:
        missingTerm = math.pow(math.e, order) * np.ones(len(predictions)) # account for the terms 1/2^m and -1/2^m

    banzaf_vec = banzaf_vec + missingTerm
    return banzaf_vec/n


def get_shapley(dataset):
    """Shapley values of ENPOWER_B, averaged over the data points, with the same input as get_banzaf."""
    m = len(dataset) # number of data points
    n = len(dataset[0]) # number of base classifiers
    shap = np.zeros(n)
    v_N = 0 
    cache = {}
    corr = 0
    for predictions in dataset:
        n_plus = np.sum(predictions)
        n_minus = n - n_plus
        a_correct, a_wrong = 0, 0
        b_correct, b_wrong = 0, 0
        if n_minus < n_plus:
            b_correct = int(n_minus)
            a_wrong = scipy.special.beta(n - 2*n_minus + 1, 2*n_minus)*scipy.special.comb(n_plus, n_minus) # first term of the sum
            b_wrong = int(n_minus - 1)
            v_N = 1
        elif n_minus > n_plus:
            a_correct = scipy.special.beta(n - 2*n_plus + 1, 2*n_plus)*scipy.special.comb(n_minus, n_plus) # first term of the sum
            b_correct = int(n_plus - 1)
            b_wrong = n_plus
            v_N = 0
        else:
            a_correct = 1./n
            a_wrong = 1./n
            b_correct = int(n_plus - 1)
            b_wrong = int(n_minus - 1)
            v_N = 0.5
        shap_corr, shap_wrong = 0, 0
        if n_plus not in cache.keys():
            shap_corr = compute_shap_summation(n, a_correct, b_correct, n_plus, n_minus, True)
            shap_wrong = compute_shap_summation(n, a_wrong, b_wrong, n_minus, n_plus)
            cache[n_plus] = [shap_corr, shap_wrong]
        else:
            shap_corr = cache[n_plus][0]
            shap_wrong = cache[n_plus][1]

        shap = shap + np.where(predictions == 1, shap_corr, -shap_wrong) 

        tmp = np.where(predictions == 1, shap_corr, -shap_wrong) 
    print(corr, m)
    return shap/m

if __name__ == "__main__":
    totModels = 5
    totPoints = 10000
    np.random.seed(43)
    dataset = np.random.choice([0,1], size=(totPoints, totModels), p=[.2, .8])
    dataset = np.array([[1, 0, 0],[1,1,0]])
    print(dataset)
    bz = get_banzaf(dataset)
    sh = get_shapley(dataset)
    print(bz)
    print(sh)
    print("Sum banz",np.sum(bz))
    print("Sum shap",np.sum(sh))
