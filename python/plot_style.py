"""Labels, colors and matplotlib style of the figures of the binary experiments.

`enpower.py splits report` draws the runtime figures (RT), the Kendall-tau scatter (SC) and the
sub-ensemble selection figures (GE) with the names below. apply() sets the matplotlib state the
three figures are drawn in (LaTeX text with Libertine fonts, the seaborn "whitegrid" theme).
"""
from types import SimpleNamespace

import numpy as np

LATEX_PREAMBLE = (r'\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} '
                  r'\RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath} '
                  r'\RequirePackage{xfrac} \RequirePackage{amsmath}')

ENSNAMES = {
    'random_forest': 'RF',
    'adaboost': 'AB',
    'heterogeneous': 'HET'
}

DATANAMES = {
    'titanic': '\\texttt{TT}',
    'bank-marketing': '\\texttt{BM}',
    'blood-transfusion-service-center': '\\texttt{BTSC}',
    'cnae-9': '\\texttt{C9}',
    'credit-g': '\\texttt{CR}',
    'covertype': '\\texttt{CT}',
}

_ALGNAMES = {
    'Troupe_Permutation': '$\\mathtt{Tr\\text{-}PS}$',
    'Troupe_MLE': '$\\mathtt{Tr\\text{-}MLE}$',
    'WMG_Banzhaf_Exact': '$\\mathtt{WMG\\text{-}\\beta}$',
    'WMG_Shapley_Exact': '$\\mathtt{WMG\\text{-}\\phi}$',
    'Our_MC_Banzhaf': '$\\mathtt{MC\\text{-}\\beta}$',
    'Our_MC_Shapley': '$\\mathtt{MC\\text{-}\\phi}$',
    'ISCG_Ranking': '$\\mathtt{ISCG}$',
    'Our_Exact_Plurality_Banzhaf_Binary': '$\\mathtt{ENP}\\text{-}\\beta$',
    'Our_Exact_Plurality_Shapley_Binary': '$\\mathtt{ENP}\\text{-}\\phi$'
}


# --- runtime figures (Fig. 1 and 6) -----------------------------------------
# Viridis palette (n=10), mapped alphabetically
RT = SimpleNamespace(
    ALGNAMES=dict(_ALGNAMES, Troupe_Exact='$\\mathtt{Tr\\text{-}EX}$'),
    DATANAMES=DATANAMES,
    ENSNAMES=ENSNAMES,
    ALG_COLORS={
        '$\\mathtt{ENP}\\text{-}\\beta$': '#440154',
        '$\\mathtt{ENP}\\text{-}\\phi$': '#482878',
        '$\\mathtt{ISCG}$': '#3e4989',
        '$\\mathtt{MC\\text{-}\\beta}$': '#31688e',
        '$\\mathtt{MC\\text{-}\\phi}$': '#26828e',
        '$\\mathtt{Tr\\text{-}EX}$': '#1f9e89',
        '$\\mathtt{Tr\\text{-}MLE}$': '#35b779',
        '$\\mathtt{Tr\\text{-}PS}$': '#6ece58',
        '$\\mathtt{WMG\\text{-}\\beta}$': '#b5de2b',
        '$\\mathtt{WMG\\text{-}\\phi}$': '#fde725'
    },
)


# --- Kendall's tau scatter plot (Fig. 2) --------------------------------------
def safe_kendall(exact, approx):
    from scipy.stats import kendalltau

    kt, _ = kendalltau(exact, approx)
    return 0.0 if np.isnan(kt) else kt


def _viridis(k):
    import seaborn as sns

    return sns.color_palette("viridis", n_colors=k)


_BASELINES = {
    'ISCG_Ranking': 'ISCG',
    'Troupe_MLE': 'Tr-MLE',
    'WMG_Banzhaf_Exact': r'WMG-$\beta$',
    'WMG_Shapley_Exact': r'WMG-$\phi$'
}

SC = SimpleNamespace(
    ENSEMBLES=['random_forest', 'adaboost', 'heterogeneous'],
    ENSNAMES=ENSNAMES,
    TARGET_DATASETS=['titanic', 'bank-marketing', 'blood-transfusion-service-center',
                     'cnae-9', 'credit-g', 'Covertype'],
    DATASET_LABELS=DATANAMES,
    MARKERS={
        'titanic': 'o',
        'bank-marketing': 's',
        'blood-transfusion-service-center': '^',
        'cnae-9': 'D',
        'credit-g': 'v',
        'covertype': '*'
    },
    BASELINES=_BASELINES,
    COLORS=dict(zip(list(_BASELINES), _viridis(4))),
    safe_kendall=safe_kendall,
)


# --- sub-ensemble selection (Fig. 3 and 7) -------------------------------------
def get_linestyle(alg_name):
    """Line style by family of methods."""
    if "Our" in alg_name:
        return '-'
    elif "WMG" in alg_name:
        return '--'
    else:
        return '-.'


GE = SimpleNamespace(
    ALGNAMES=_ALGNAMES,
    TARGET_ALGS=[
        'Troupe_MLE',
        'WMG_Banzhaf_Exact',
        'WMG_Shapley_Exact',
        'ISCG_Ranking',
        'Our_Exact_Plurality_Banzhaf_Binary',
        'Our_Exact_Plurality_Shapley_Binary'
    ],
    DATANAMES=DATANAMES,
    ENSNAMES=ENSNAMES,
    get_linestyle=get_linestyle,
)


def apply():
    """The matplotlib state of the figures: LaTeX text, Libertine, seaborn whitegrid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=LATEX_PREAMBLE)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })
