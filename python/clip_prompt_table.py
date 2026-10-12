"""Per-prompt LaTeX table of the CLIP experiment (Tables 9-11 of the paper).

Reads the <dataset>_ViT-B-32_results.json written by clip_inference.py (without --votes-only) and
writes <out_dir>/<dataset>.tex: accuracy, Banzhaf and Shapley value of every prompt on D_val, with
the change of rank of the two indices with respect to accuracy.

    python clip_prompt_table.py ../outputs/pets_ViT-B-32_results.json [out_dir]
"""
import json
import os
import numpy as np
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_NAME = sys.argv[1]
FOLRES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "results", "tables")
os.makedirs(FOLRES, exist_ok=True)

with open(FILE_NAME, 'r') as f:
    data = json.load(f)

sentences = data['sentences']

# metrics on D_val, used only to rank the prompts
val_acc = data.get('val_accuracy', data['accuracy'])
val_banzhaf = data.get('val_banzhaf', data['banzhaf'])
val_shapley = data.get('val_shapley', data['shapley'])

# metrics on D_test, used only for the values shown in the table
test_acc = data.get('test_accuracy', data['accuracy'])
test_banzhaf = data.get('test_banzhaf', data['banzhaf'])
test_shapley = data.get('test_shapley', data['shapley'])

def get_ranks(arr):
    # ranks from 1 to N; the largest value has rank 1
    arr = np.array(arr)
    temp = arr.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr)) + 1
    return ranks

# the ranks are computed on D_val only
acc_ranks = get_ranks(val_acc)
ban_ranks = get_ranks(val_banzhaf)
shap_ranks = get_ranks(val_shapley)

def format_cell(val, rank, fmt, max_rank=len(acc_ranks)):
    # rank 1 -> 100% intensity, rank N -> 10% intensity (rank on D_val)
    intensity = int(100 - (rank - 1) * (90 / (max_rank - 1)))
    
    # white text on dark cells, black text on light cells
    text_color = r"\color{white}" if intensity > 60 else r"\color{black}"
    
    # the three best ranks in bold
    rank_str = f"\\textbf{{{rank}}}" if rank <= 3 else str(rank)
    
    # the value is measured on D_test; the rank and the color come from D_val
    return f"\\cellcolor{{CBB!{intensity}}} {text_color} {fmt.format(val)} & \\cellcolor{{CBB!{intensity}}} {text_color} {rank_str}"

def format_gap(gap):
    if gap > 0:
        return f"\\textcolor{{CBG}}{{\\textbf{{$\\uparrow$ {gap}}}}}"
    elif gap < 0:
        return f"\\textcolor{{CBR}}{{\\textbf{{$\\downarrow$ {abs(gap)}}}}}"
    else:
        return f"\\textcolor{{CBGy}}{{$-$}}"

table_descr = "Prompt ensemble evaluation for LLM zero-shot multiclass \
image classification (" + data['metadata']['dataset'].upper() + "). \
Each prompt corresponds to a different model of $\\mathcal{C}$. \
All values and rankings are computed on the validation split $\\dataSetVal$. \
Columns $\\Delta_{\\banz}$ and $\\Delta_{\\shap}$ display the difference in the validation rankings induced by the power indices \
$\\banz(G_{\\mathsf{MC}})$ and $\\shap(G_{\\mathsf{MC}})$ \
with respect to validation accuracy."

# LaTeX code of the table
latex = []
latex.append(r"\begin{table}[t]")
latex.append(r"\setlength{\aboverulesep}{0pt}")
latex.append(r"\setlength{\belowrulesep}{0pt}")
latex.append(r"\setlength{\extrarowheight}{0.8pt}")
latex.append(r"\centering")
latex.append(r"\caption{" + table_descr +"}")
latex.append(r"\label{tab:results_" + data['metadata']['dataset'] + "}")

# color-blind friendly palette (Okabe-Ito)
latex.append(r"\definecolor{CBB}{HTML}{0072B2} % Blue for heatmaps")
latex.append(r"\definecolor{CBG}{HTML}{009E73} % Green for positive gap")
latex.append(r"\definecolor{CBR}{HTML}{D55E00} % Vermilion/Red for negative gap")
latex.append(r"\definecolor{CBGy}{HTML}{999999} % Grey for neutral gap")

latex.append(r"\resizebox{\textwidth}{!}{")
latex.append(r"\begin{tabular}{l rr rr rr cc}")
latex.append(r"\toprule")
latex.append(r"\multirow{2}{*}{\textbf{Prompt template}} & \multicolumn{2}{c}{\textbf{Accuracy}} & \multicolumn{2}{c}{\textbf{Banzhaf}} & \multicolumn{2}{c}{\textbf{Shapley}} & \multirow{2}{*}{$\Delta_{\banz}$} & \multirow{2}{*}{$\Delta_{\shap}$} \\")
latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
latex.append(r"& Value & Rank & Value & Rank & Value & Rank & & \\")
latex.append(r"\midrule")

accs = np.array([acc for acc in test_acc])
sorted = np.argsort(accs)[::-1]
for i in sorted:
    # escape the special characters of LaTeX
    s = sentences[i].replace("_", "\\_").replace("{", "\\{").replace("}", "\\}")
    
    # cells: values on D_test, ranks on D_val
    c_acc = format_cell(test_acc[i], acc_ranks[i], "{:.4f}")
    c_ban = format_cell(test_banzhaf[i], ban_ranks[i], "{:.5f}")
    c_shap = format_cell(test_shapley[i], shap_ranks[i], "{:.5f}")
    
    # rank by accuracy minus rank by the power index, both on D_val
    gap_ban = int(acc_ranks[i] - ban_ranks[i])
    gap_shap = int(acc_ranks[i] - shap_ranks[i])
    
    gap_ban_str = format_gap(gap_ban)
    gap_shap_str = format_gap(gap_shap)
        
    latex.append(f"{s} & {c_acc} & {c_ban} & {c_shap} & {gap_ban_str} & {gap_shap_str} \\\\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"}")
latex.append(r"\end{table}")

print("\n".join(latex))
outname = os.path.join(FOLRES, data['metadata']['dataset'] + ".tex")
print(outname)
with open(outname, "w", encoding="utf-8") as f:
    for line in latex:
        f.write(f"{line}\n")
