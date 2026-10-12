#!/usr/bin/env python3
"""Driver of the C++ library: runs ./solver, parses its JSON output, and runs the experiments of the paper.

Validation of the C++ implementation against the Python reference implementations (python/):

  check   Multiclass Shapley values against shapley_multiclass.py, on a random
          dataset or on a given vote profile.

              python cpp/tools/enpower.py check --points 20 --models 12 --labels 4
              python cpp/tools/enpower.py check --profile 20:20,20,20

  banzhaf Multiclass Banzhaf values, and banzhaf_multiclass.py, against an exact
          computation of the definition of the Banzhaf value.

              python cpp/tools/enpower.py banzhaf --profile 6:6,6,6

  binary  The closed forms of ENPOWER_B against indices_binary.py, and against
          the multiclass algorithms on data with two labels.

              python cpp/tools/enpower.py binary --points 500 --models 30

  dataset Converts a *_predictions.npz file of outputs/ to the input format of ./solver.

Experiments of the paper. Each experiment has a `cells` command that lists its
cells, one per line, which scripts/cell.sh runs; `report` writes the figures
and tables to results/<experiment>/.

  splits  Binary classification (runtime, Kendall's tau, sub-ensemble
          selection, error of MC) over five random splits, in outputs/splits/:
          `run` measures ENPOWER_B, WMG and ISCG on one cell, and `mc` runs
          MC-beta and MC-phi. The ensembles, Tr-PS and Tr-MLE are computed by
          python/train_binary.py.

              python cpp/tools/enpower.py splits run --seeds 100 --datasets credit-g
              python cpp/tools/enpower.py splits report

  mc      Runtime for multiclass classification (Table 5) over five random
          splits, in outputs/mc_splits/, on the ensembles of
          python/train_multiclass.py: `run` measures ENPOWER_MC-beta,
          ENPOWER_MC-phi and ENPOWER_MC-phi++.

              python cpp/tools/enpower.py mc run --seeds 100 --datasets cnae-9
              python cpp/tools/enpower.py mc report

  clip    Prompt ensembling of CLIP (Sec. 5.4) over five random splits, in
          outputs/clip_splits/: ranks the prompts on D_val with Acc,
          ENPOWER_MC-beta, ENPOWER_MC-phi, Tr-PS and Tr-MLE, and reports the
          top-k accuracy and macro F1 on D_test with plurality voting and
          random tie-breaking. `votes` runs CLIP inference
          (python/clip_inference.py) for the datasets without stored votes.

              python cpp/tools/enpower.py clip votes
              python cpp/tools/enpower.py clip run --seeds 100 --datasets pets
              python cpp/tools/enpower.py clip report

  synth   Synthetic experiments (Appendix G.6), in three stages over the grid of
          a configuration in cpp/experiments/synth/: `gen` writes the datasets,
          `run` writes the runtime of each algorithm as JSON lines, and `report`
          plots the runtime per unique vote profile in log-log scale. Each
          stage skips the work whose output already exists.

              python cpp/tools/enpower.py synth gen --config small
              python cpp/tools/enpower.py synth run --config small
              python cpp/tools/enpower.py synth report --config small

The raw outputs are written to outputs/ and the figures and tables to results/;
the variables ENPOWER_OUTPUTS and ENPOWER_RESULTS set other locations.
"""
import argparse
import io
import json
import os
import random
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SOLVER = os.path.join(REPO, "cpp", "solver")
# the Python code: reference implementations, ensemble training, CLIP, Troupe
PYTHON_DIR = os.path.join(REPO, "python")
# the raw outputs of the runs, and the figures and tables made from them
OUTPUTS = os.environ.get("ENPOWER_OUTPUTS", os.path.join(REPO, "outputs"))
RESULTS = os.environ.get("ENPOWER_RESULTS", os.path.join(REPO, "results"))
# the outputs of the C++ methods, one subfolder per split
CPP_DIR = "cpp"

# The datasets of the multiclass experiments, by OpenML id
DATASETS = {"cnae-9": 1468, "Covertype": 150}
ENSEMBLES = ["random_forest", "adaboost", "heterogeneous"]


def reference():
    """compute_shapley_multiclass of the Python reference implementation."""
    sys.path.insert(0, PYTHON_DIR)
    from shapley_multiclass import compute_shapley_multiclass

    return compute_shapley_multiclass


def run_solver(args, payload=None):
    """One invocation of ./solver, with its JSON parsed."""
    out = subprocess.run([SOLVER] + args, input=payload,
                         capture_output=True, check=True)
    return json.loads(out.stdout)


def encode(preds, labs, labels):
    """The dataset in the format ./solver shapley reads."""
    header = "%d %d %d\n" % (len(labs), len(preds[0]), labels)
    try:
        import numpy as np

        buf = io.BytesIO()
        np.savetxt(buf, np.column_stack([labs, preds]).astype(np.int64), fmt="%d")
        return header.encode() + buf.getvalue()
    except ImportError:
        rows = [" ".join(str(v) for v in [y] + list(p)) for y, p in zip(labs, preds)]
        return (header + "\n".join(rows) + "\n").encode()


def gap(expected, got):
    """Largest relative gap between two vectors of values."""
    scale = max(abs(v) for v in expected) or 1.0
    return max(abs(a - b) for a, b in zip(expected, got)) / scale


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def one_point(spec):
    """A dataset of one data point with the vote profile "n_1:n_2,n_3,...".

    Label 0 is the correct label and receives n_1 votes; label j receives the
    j-th count.
    """
    head, _, tail = spec.partition(":")
    counts = [int(head)] + [int(c) for c in tail.split(",") if c]
    preds = [[j for j, c in enumerate(counts) for _ in range(c)]]
    return preds, [0], len(counts)


def banzhaf_bruteforce(counts, n):
    """The exact Banzhaf values of one vote profile, from the definition.

    counts[0] is n_1, and the other entries are the votes of the incorrect
    labels. With the characteristic function of Eq. (3), v(S) is 1 / (number of
    labels with the most votes) when the correct label is one of them, and 0
    otherwise, and beta_i = 2^-(n-1) sum_{S subset N - i} v(S + i) - v(S).
    Coalitions with the same number of votes for each label contribute the same
    term, so the sum runs over the vote profiles of S, weighted by the number of
    coalitions with each profile.
    """
    from fractions import Fraction
    from itertools import product
    from math import comb

    def value(s):
        total = sum(s)
        if total == 0:
            return Fraction(0)                     # v(empty) = 0
        top = max(s)
        if s[0] != top:
            return Fraction(0)                     # the correct label does not receive the most votes
        return Fraction(1, s.count(top))

    correct, incorrect = Fraction(0), {}
    for t, c in enumerate(counts):
        if c == 0:
            continue
        lim = list(counts)
        lim[t] -= 1                                # remove the player from the votes of its label
        acc = Fraction(0)
        for s in product(*[range(v + 1) for v in lim]):
            joined = list(s)
            joined[t] += 1
            dv = value(joined) - value(list(s))
            if dv:
                ways = 1
                for q, sq in enumerate(s):
                    ways *= comb(lim[q], sq)
                acc += dv * ways
        acc = Fraction(acc, 2 ** (n - 1))
        if t == 0:
            correct = acc
        else:
            incorrect[counts[t]] = acc
    return correct, incorrect


def cmd_banzhaf(args):
    """Diffs the C++ and banzhaf_multiclass.py against the definition."""
    import numpy as np
    from fractions import Fraction

    sys.path.insert(0, PYTHON_DIR)
    from banzhaf_multiclass import compute_banzhaf_multiclass

    if args.profile:
        preds, y, labels = one_point(args.profile)
    else:
        labels = args.labels
        rng = random.Random(args.seed)
        y = [rng.randrange(labels) for _ in range(args.points)]
        preds = [[y[i] if rng.random() < 0.6 else rng.randrange(labels)
                  for _ in range(args.models)] for i in range(args.points)]

    n = len(preds[0])
    python = compute_banzhaf_multiclass(np.array(preds), np.array(y))
    port = run_solver(["banzhaf"] + (["--float"] if args.float_coef else []),
                      encode(preds, y, labels))["values"]

    # the same values as the reference implementation, in exact arithmetic
    expected = [Fraction(0)] * n
    for row, label in zip(preds, y):
        tally = [0] * max(labels_of(preds, y), 1)
        for p in row:
            tally[p] += 1
        counts = [tally[label]] + [c for j, c in enumerate(tally)
                                   if j != label and c > 0]
        correct, incorrect = banzhaf_bruteforce(counts, n)
        for i, p in enumerate(row):
            expected[i] += correct if p == label else incorrect[tally[p]]
    expected = [float(v / len(y)) for v in expected]

    print("points %d, models %d, labels %d, profile %s"
          % (len(y), n, labels, args.profile or "random, seed %d" % args.seed))
    worst = 0.0
    for name, values in (("banzhaf_multiclass.py", python), ("the C++", port)):
        g = gap(expected, values)
        worst = max(worst, g)
        print("  %-21s vs %-21s relative gap %.3e" % (name, "the definition", g))
    g = gap(python, port)
    worst = max(worst, g)
    print("  %-21s vs %-21s relative gap %.3e"
          % ("the C++", "banzhaf_multiclass.py", g))
    print("OK" if worst < 1e-12 else "MISMATCH")
    return 0 if worst < 1e-12 else 1


# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------

def cmd_binary(args):
    """Diffs the two binary indices of the C++ against indices_binary.py.

    The binary game is the multiclass one of two labels, so the same dataset
    goes through four implementations: the closed forms of indices_binary.py, the closed
    forms of the C++ library, and the two multiclass ones, which are the reference
    implementations of the experiments. All four have to agree.
    """
    import numpy as np

    sys.path.insert(0, PYTHON_DIR)
    from indices_binary import get_banzaf, get_shapley
    from banzhaf_multiclass import compute_banzhaf_multiclass

    rng = random.Random(args.seed)
    y = [rng.randrange(2) for _ in range(args.points)]
    preds = [[y[i] if rng.random() < 0.6 else 1 - y[i]
              for _ in range(args.models)] for i in range(args.points)]
    # the input of indices_binary.py: 1 where the classifier is correct on the data point
    correct = np.array([[1 if p == y[i] else 0 for p in row]
                        for i, row in enumerate(preds)])
    payload = encode(preds, y, 2)

    print("points %d, models %d, seed %d" % (args.points, args.models, args.seed))
    worst = 0.0

    port = run_solver(["banzhaf", "--binary"], payload)["values"]
    rows = [("indices_binary.py", list(get_banzaf(correct))),
            ("the multiclass C++", run_solver(["banzhaf"], payload)["values"]),
            ("banzhaf_multiclass.py",
             list(compute_banzhaf_multiclass(np.array(preds), np.array(y))))]
    for name, values in rows:
        g = gap(values, port)
        worst = max(worst, g)
        print("  Banzhaf  C++ vs %-22s relative gap %.3e" % (name, g))

    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):   # get_shapley prints a counter
        alg_shapley = list(get_shapley(correct))

    port = run_solver(["shapley", "--binary"], payload)["binary"]["values"]
    rows = [("indices_binary.py", alg_shapley),
            ("the multiclass C++",
             run_solver(["shapley", "--refined"], payload)["refined"]["values"]),
            ("shapley_multiclass.py", reference()(preds, y, 2))]
    for name, values in rows:
        g = gap(values, port)
        worst = max(worst, g)
        print("  Shapley  C++ vs %-22s relative gap %.3e" % (name, g))

    print("OK" if worst < 1e-12 else "MISMATCH")
    return 0 if worst < 1e-12 else 1


def labels_of(preds, y):
    return max(max(max(r) for r in preds), max(y)) + 1


def cmd_check(args):
    if args.profile:
        preds, y, labels = one_point(args.profile)
        args.points, args.models, args.labels = 1, len(preds[0]), labels
    else:
        labels = args.labels
        rng = random.Random(args.seed)
        y = [rng.randrange(labels) for _ in range(args.points)]
        # a random ensemble in which each classifier is correct with probability 0.6
        preds = [[y[i] if rng.random() < 0.6 else rng.randrange(labels)
                  for _ in range(args.models)] for i in range(args.points)]

    expected = reference()(preds, y, labels)
    out = run_solver(["shapley"] + (["--float"] if args.float_coef else []),
                     encode(preds, y, labels))

    print("points %d, models %d, labels %d, seed %d, profiles %d"
          % (args.points, args.models, labels, args.seed, out["profiles"]))
    worst = 0.0
    for name in ("baseline", "refined"):
        g = gap(expected, out[name]["values"])
        worst = max(worst, g)
        print("  python vs %-8s  relative gap %.3e" % (name, g))
    print("OK" if worst < 1e-12 else "MISMATCH")
    return 0 if worst < 1e-12 else 1


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

# An .npz file is a zip archive of .npy files, and an .npy file is a short
# header followed by the raw data, so the integer arrays of a prediction file
# can be read with struct. This code does not import numpy, so that the C++
# library can be used without a scientific Python installation.
NPY_CODE = {
    "|i1": "b", "|u1": "B", "<i2": "h", "<u2": "H",
    "<i4": "i", "<u4": "I", "<i8": "q", "<u8": "Q",
}


def read_npy(raw):
    """One .npy member, as a list (1-D) or a list of rows (2-D)."""
    import ast
    import struct

    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy member")
    if raw[6] == 1:
        (hlen,) = struct.unpack("<H", raw[8:10])
        start = 10
    else:
        (hlen,) = struct.unpack("<I", raw[8:12])
        start = 12
    header = ast.literal_eval(raw[start:start + hlen].decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("fortran order is not handled")
    code = NPY_CODE.get(header["descr"])
    if code is None:
        raise ValueError("%s is not an integer dtype" % header["descr"])
    shape = header["shape"]
    n = 1
    for s in shape:
        n *= s
    body = raw[start + hlen:]
    vals = struct.unpack("<%d%s" % (n, code), body[:n * struct.calcsize(code)])
    if len(shape) == 1:
        return list(vals)
    cols = shape[1]
    return [list(vals[r * cols:(r + 1) * cols]) for r in range(cols and shape[0])]


def cmd_dataset(args):
    """An .npz of cached predictions -> the text dataset ./solver reads."""
    import zipfile

    split = args.split
    with zipfile.ZipFile(args.npz) as z:
        preds = read_npy(z.read("preds_%s.npy" % split))
        labs = read_npy(z.read("labels_%s.npy" % split))
    rows = preds if len(preds) == len(labs) else [list(c) for c in zip(*preds)]
    if len(rows) != len(labs):
        print("preds_%s and labels_%s disagree on the number of points" % (split, split))
        return 1
    labels = max(max(max(r) for r in rows), max(labs)) + 1
    with open(args.out, "w") as f:
        f.write("%d %d %d\n" % (len(rows), len(rows[0]), labels))
        for lab, row in zip(labs, rows):
            f.write("%d %s\n" % (lab, " ".join(str(v) for v in row)))
    print("points=%d models=%d labels=%d -> %s"
          % (len(rows), len(rows[0]), labels, args.out))
    return 0


# --- synthetic experiments (Appendix G.6) -----------------------------------
#
# Three stages over the grid of a configuration file in experiments/synth/:
#
#   gen     ./solver gen writes one dataset per (task, n, L, |U|, p) to
#           data/synth/<name>/, and its metadata to gen.jsonl. The instance is
#           deterministic (Appendix G.6, "Data generation"), so there is one
#           instance per configuration and no seed.
#   run     measures the runtime of the exact algorithms on every generated
#           dataset, `repeats` times each, with one line of runs.jsonl per
#           (dataset, algorithm, repetition), and runs ./solver verify on the
#           small datasets. --shard I --shards K selects the I-th of K disjoint
#           parts of the grid and writes to runs.<I>.jsonl, for a SLURM array.
#   report  plots the runtime per unique vote profile in log-log scale with the
#           slopes of the bounds, and prints the speedup of ENPOWER_MC-phi++
#           over ENPOWER_MC-phi (requires matplotlib).
#
# The logs are written to outputs/synth/<name>/ and the figures to
# results/synth/<name>/. Each stage skips the entries already present in its
# .jsonl file, so rerunning an interrupted stage resumes it.

SYNTH_CONFIGS = os.path.join(REPO, "cpp", "experiments", "synth")
SYNTH_DATA = os.path.join(REPO, "data", "synth")
SYNTH_RESULTS = os.path.join(OUTPUTS, "synth")

# The arguments of ./solver and the key of its JSON output that holds the
# result, per (task, index). The binary task runs the closed forms of ENPOWER_B;
# on the multiclass task, "banzhaf" is ENPOWER_MC-beta, "shapley" is
# ENPOWER_MC-phi, and "shapley_refined" is ENPOWER_MC-phi++ (Appendix H).
SYNTH_ALGOS = {
    ("binary", "banzhaf"): (["banzhaf", "--binary"], None),
    ("binary", "shapley"): (["shapley", "--binary"], "binary"),
    ("multiclass", "banzhaf"): (["banzhaf"], None),
    ("multiclass", "shapley"): (["shapley", "--baseline"], "baseline"),
    ("multiclass", "shapley_refined"): (["shapley", "--refined"], "refined"),
}
SYNTH_INDICES = {"binary": ["banzhaf", "shapley"],
                 "multiclass": ["banzhaf", "shapley", "shapley_refined"]}

# The exponents of (n, L, |U|) in the runtime bounds of the paper, plotted as
# guides: Thm 4.3 for ENPOWER_B (with |D| = |U|), and Thm 4.7 and 4.9 for
# ENPOWER_MC-beta and ENPOWER_MC-phi. Both Shapley algorithms are plotted
# against the bound n^3 L of Thm 4.9.
SYNTH_BOUNDS = {
    ("binary", "banzhaf"): (1, 0, 1),
    ("binary", "shapley"): (1, 0, 1),
    ("multiclass", "banzhaf"): (1, 2, 1),
    ("multiclass", "shapley"): (3, 1, 1),
}


def synth_config(path):
    if not os.path.exists(path):
        path = os.path.join(SYNTH_CONFIGS, path + ".json")
    with open(path) as f:
        return json.load(f)


def synth_grid(cfg):
    """Every (task, n, L, |U|, p) of the config, skipping the |U| no binary
    task can reach: a binary profile is n_+ alone, so |U| <= n + 1."""
    for task in ("binary", "multiclass"):
        g = cfg.get(task)
        if not g:
            continue
        for n in g["models"]:
            for L in ([2] if task == "binary" else g["labels"]):
                for U in g["profiles"]:
                    if task == "binary" and U > n + 1:
                        continue
                    for p in g["p"]:
                        yield dict(task=task, n=n, L=L, U=U, p=p)


def synth_id(c):
    return "%s_n%d_L%d_U%d_p%g" % (c["task"], c["n"], c["L"], c["U"], c["p"])


def synth_shard(task, L, U, p, shards):
    """The shard of a cell. Every n of a (task, L, |U|, p) lands in the same
    shard, so a timeout there still spares the larger n."""
    return zlib.crc32(("%s_L%d_U%d_p%g" % (task, L, U, p)).encode()) % shards


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def synth_runs(res):
    """runs.jsonl and the runs.<I>.jsonl of every shard."""
    return [r for f in sorted(os.listdir(res)) if f.startswith("runs") and f.endswith(".jsonl")
            for r in read_jsonl(os.path.join(res, f))]


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_synth_gen(cfg, args):
    name = cfg["name"]
    data, res = os.path.join(SYNTH_DATA, name), os.path.join(SYNTH_RESULTS, name)
    os.makedirs(data, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    log = os.path.join(res, "gen.jsonl")
    done = {r["id"] for r in read_jsonl(log)}

    made = skipped = 0
    for c in synth_grid(cfg):
        cid = synth_id(c)
        if cid in done:
            continue
        path = os.path.join(data, cid + ".txt")
        meta = run_solver(["gen", "--out", path, "--models", str(c["n"]),
                           "--labels", str(c["L"]), "--profiles", str(c["U"]),
                           "--p", str(c["p"])])
        rec = dict(c, id=cid, path=os.path.relpath(path, SYNTH_DATA),
                   exhausted=meta["exhausted"], profiles=meta["profiles"],
                   available=meta["available"], grand_value=meta["grand_value"],
                   gen_seconds=meta["gen_seconds"])
        append_jsonl(log, rec)
        made += 1
        skipped += meta["exhausted"]
        print("gen %-40s |U|=%5d/%-5d (%d pairs)%s" % (cid, meta["profiles"], c["U"],
              meta["available"], "  EXHAUSTED, skipped" if meta["exhausted"] else ""),
              flush=True)
    print("generated %d instances, %d skipped as exhausted -> %s" % (made, skipped, log))


def cmd_synth_run(cfg, args):
    res = os.path.join(SYNTH_RESULTS, cfg["name"])
    log = os.path.join(res, "runs.jsonl" if args.shards == 1 else "runs.%d.jsonl" % args.shard)
    done = {(r["id"], r["index"], r.get("rep", 0)) for r in read_jsonl(log)}
    timeout = args.timeout or cfg.get("timeout", 3600)
    repeats = cfg.get("repeats", 5)
    # a cell whose smaller instance timed out is not tried on a larger one
    timed_out = set()

    instances = [g for g in read_jsonl(os.path.join(res, "gen.jsonl")) if not g["exhausted"]
                 and synth_shard(g["task"], g["L"], g["U"], g["p"], args.shards) == args.shard]
    instances.sort(key=lambda g: (g["n"], g["L"], g["U"]))
    print("shard %d/%d: %d instances, %d repeats -> %s" % (args.shard, args.shards,
          len(instances), repeats, log), flush=True)
    for g in instances:
        path = os.path.join(SYNTH_DATA, g["path"])
        verify = g["n"] <= cfg.get("verify_max_models", 12) and g["task"] == "multiclass"
        jobs = [(index, rep) for index in SYNTH_INDICES[g["task"]] for rep in range(repeats)]
        jobs += [("verify", 0)] if verify else []
        for index, rep in jobs:
            if (g["id"], index, rep) in done:
                continue
            key = (g["task"], index, g["L"], g["U"], g["p"])
            rec = dict(id=g["id"], index=index, rep=rep, task=g["task"], n=g["n"], L=g["L"],
                       U=g["U"], p=g["p"])
            if key in timed_out:
                rec.update(status="skipped", reason="timed out on this or a smaller n")
                append_jsonl(log, rec)
                continue
            cmd, part_key = (["verify"], None) if index == "verify" else SYNTH_ALGOS[(g["task"], index)]
            try:
                out = subprocess.run([SOLVER] + cmd + ["--input", path], capture_output=True,
                                     text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                rec.update(status="timeout", timeout=timeout)
                timed_out.add(key)
                append_jsonl(log, rec)
                print("run %-40s %-15s TIMEOUT (%ds)" % (g["id"], index, timeout), flush=True)
                continue
            if out.returncode != 0:
                rec.update(status="error", stderr=out.stderr[-500:])
                append_jsonl(log, rec)
                print("run %-40s %-15s ERROR %s" % (g["id"], index, out.stderr.strip()[-200:]))
                continue
            d = json.loads(out.stdout)
            rec["status"] = "ok"
            if index == "verify":
                rec.update({k: v for k, v in d.items() if "_vs_" in k})
            else:
                part = d[part_key] if part_key else d
                rec["seconds"] = part["seconds"]
                if index.startswith("shapley"):   # efficiency: the values sum to v(N)
                    rec["efficiency_gap"] = abs(sum(part["values"]) - g["grand_value"])
            append_jsonl(log, rec)
            print("run %-40s %-15s %s" % (g["id"], "%s#%d" % (index, rep), "%.4fs" % rec["seconds"]
                  if "seconds" in rec else {k: v for k, v in rec.items() if "_vs_" in k}),
                  flush=True)
    print("-> %s" % log)


# Labels of the figures and tables, as the paper typesets them (macros.tex:
# \suite is \textsc{enpower}, \banz is beta, \shap is phi)
SYNTH_ALGO_LABELS = {
    ("binary", "banzhaf"): r"\textsc{enpower}$_{\mathsf{B}}$-$\beta$",
    ("binary", "shapley"): r"\textsc{enpower}$_{\mathsf{B}}$-$\phi$",
    ("multiclass", "banzhaf"): r"\textsc{enpower}$_{\mathsf{MC}}$-$\beta$",
    ("multiclass", "shapley"): r"\textsc{enpower}$_{\mathsf{MC}}$-$\phi$",
    ("multiclass", "shapley_refined"): r"\textsc{enpower}$_{\mathsf{MC}}$-$\phi$++",
}
SYNTH_AXIS_LABELS = {
    "n": r"$n$",
    "nL2": r"$n\ell^2$",
    "n3L": r"$n^3\ell$",
    "per_profile": r"Runtime for $\bm{p}\in U$ (sec)",
    "L": r"$\ell$",
}


def synth_cells(runs):
    """Median runtime over the repeats of each (task, index, n, L, U, p)."""
    from statistics import median
    cells = {}
    for r in runs:
        if r["index"] == "verify" or r["status"] != "ok":
            continue
        cells.setdefault((r["task"], r["index"], r["n"], r["L"], r["U"], r["p"]), []).append(r["seconds"])
    return {k: median(v) for k, v in cells.items()}


def synth_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Times, the font of the paper (neurips_2026.sty)
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}\usepackage{bm}")
    plt.rc("font", family="serif", size=9)
    return plt


def synth_axis(ax):
    ax.grid(True, linestyle="--", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def synth_palette(k):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if k <= 5 else "viridis")
    return [cmap(i) for i in range(k)] if k <= 5 else [cmap(i / max(1, k - 1)) for i in range(k)]


def synth_legend(plt, handles, labels, path, ncol):
    fig = plt.figure()
    leg = fig.legend(handles, labels, ncol=ncol, frameon=False, loc="center")
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=bbox)
    plt.close(fig)


def synth_guide(ax, xs, ys, slope, label):
    """A dotted line of the given slope in log-log scale, through the first point of the series."""
    x0, y0 = xs[0], ys[0]
    gx = [x0, xs[-1]]
    ax.plot(gx, [y0 * (x / x0) ** slope for x in gx], ls=":", c="0.4", lw=0.9)
    ax.annotate(label, (gx[-1], y0 * (gx[-1] / x0) ** slope), fontsize=7, color="0.3",
                xytext=(2, 0), textcoords="offset points", va="center")


# The marker of each multiclass Shapley algorithm in the panel against n^3 L;
# the colors encode L, as in the Banzhaf panel
SYNTH_SHAPLEY_MARKERS = {"shapley": "o", "shapley_refined": "^"}


def synth_note(ax, text):
    """The parameters of a panel, in gray under its title, outside the data."""
    ax.text(0.5, 1.02, text, transform=ax.transAxes, fontsize=6.5, va="bottom", ha="center",
            color="0.35")


def synth_set(values):
    """A set of values as the notes print it, e.g. \\{10, 10^2, 10^3\\}."""
    def fmt(v):
        e = len(str(v)) - 1
        return "10^%d" % e if v == 10 ** e and e > 1 else str(v)
    return r"\{%s\}" % ",".join(fmt(v) for v in sorted(values))


def synth_bound(task, index, n, L):
    """The bound n^a L^b per vote profile of Thm 4.3, 4.7 and 4.9 (for
    ENPOWER_MC-phi++, the bound of ENPOWER_MC-phi)."""
    a, b, _ = SYNTH_BOUNDS[(task, "shapley" if index == "shapley_refined" else index)]
    return n ** a * L ** b


def plot_synth_bound(ax, cells, p, task, series, style):
    """Runtime per unique profile against its bound, at accuracy p: one line
    per (index, L) through the values of n, median over the repeats, then over
    |U|; style(index, L) gives the line's keywords. A dotted guide y ~ x
    through the leftmost point: points below it at larger x mean the bound is
    pessimistic, a series parallel to it means the bound is tight in n."""
    from statistics import median
    first = None
    for ix, L in series:
        per = {}
        for (t, i, n, l, U, q), sec in cells.items():
            if t == task and i == ix and l == L and q == p:
                per.setdefault(synth_bound(t, i, n, l), []).append(sec / U)
        if not per:
            continue
        xs = sorted(per)
        ys = [median(per[x]) for x in xs]
        ax.plot(xs, ys, **style(ix, L))
        if first is None or (xs[0], ys[0]) < first:
            first = (xs[0], ys[0])
    xmax = max((x for (t, i, n, l, U, q) in cells if t == task and q == p and (i, l) in series
                for x in [synth_bound(t, i, n, l)]), default=None)
    if first and xmax > first[0]:
        synth_guide(ax, [first[0], xmax], [first[1]], 1, "")
    ax.set_xscale("log")
    ax.set_yscale("log")
    synth_axis(ax)


def plot_synth_per_profile(plt, cells, p, out):
    """Runtime per unique profile at accuracy p, each panel against the bound
    of its algorithm: binary (both indices) against n, multiclass Banzhaf
    against n L^2, the multiclass Shapley algorithms against n^3 L. One line
    per index (binary) or per L (multiclass), with a dotted guide of slope 1.
    Every panel notes what it fixes; p and the medians are in the caption."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    Ls = sorted({k[3] for k in cells if k[0] == "multiclass"})
    colors = dict(zip(Ls, synth_palette(len(Ls))))
    idx_colors = dict(zip(["banzhaf", "shapley"], synth_palette(2)))
    Us = {task: {k[4] for k in cells if k[0] == task and k[5] == p} for task in ("binary", "multiclass")}

    plot_synth_bound(axes[0], cells, p, "binary", [("banzhaf", 2), ("shapley", 2)],
                     lambda ix, L: dict(marker="o", ms=3, lw=1.2, c=idx_colors[ix]))
    plot_synth_bound(axes[1], cells, p, "multiclass", [("banzhaf", L) for L in Ls],
                     lambda ix, L: dict(marker="o", ms=3, lw=1.2, c=colors[L]))
    plot_synth_bound(axes[2], cells, p, "multiclass",
                     [(ix, L) for ix in ("shapley", "shapley_refined") for L in Ls],
                     lambda ix, L: dict(marker=SYNTH_SHAPLEY_MARKERS[ix], ms=3.5, c=colors[L],
                                        lw=0.9, ls="-" if ix == "shapley" else "--",
                                        mec="0.2" if ix == "shapley_refined" else colors[L], mew=0.4))
    for ax, title, xlabel in zip(axes, [r"\textsc{enpower}$_{\mathsf{B}}$",
                                        SYNTH_ALGO_LABELS[("multiclass", "banzhaf")],
                                        SYNTH_ALGO_LABELS[("multiclass", "shapley")] + " and ++"],
                                 ["n", "nL2", "n3L"]):
        ax.set_title(title, fontsize=9, pad=11)
        ax.set_xlabel(SYNTH_AXIS_LABELS[xlabel])
    synth_note(axes[0], r"$\ell=2$, $|U|\in%s$" % synth_set(Us["binary"]))
    synth_note(axes[1], r"color: $\ell$, $|U|\in%s$" % synth_set(Us["multiclass"]))
    synth_note(axes[2], r"color: $\ell$, $|U|\in%s$" % synth_set(Us["multiclass"]))
    # p and the aggregation (median of the repeats, then over |U|) are in the caption
    axes[0].set_ylabel(SYNTH_AXIS_LABELS["per_profile"])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_profile_p%g.pdf" % p), bbox_inches="tight")
    plt.close(fig)

    from matplotlib.lines import Line2D
    synth_legend(plt, [Line2D([], [], c=idx_colors[i], marker="o", ms=3) for i in ("banzhaf", "shapley")],
                 [r"$\beta$", r"$\phi$"], os.path.join(out, "legend-index.pdf"), 2)
    synth_legend(plt, [Line2D([], [], c=colors[L], marker="o", ms=3) for L in Ls],
                 [r"$\ell=%d$" % L for L in Ls], os.path.join(out, "legend-labels.pdf"), len(Ls))
    synth_legend(plt, [Line2D([], [], c="0.4", mec="0.2", marker=SYNTH_SHAPLEY_MARKERS[i], ms=4, lw=0.9,
                              ls="-" if i == "shapley" else "--")
                       for i in ("shapley", "shapley_refined")],
                 [SYNTH_ALGO_LABELS[("multiclass", i)] for i in ("shapley", "shapley_refined")],
                 os.path.join(out, "legend-shapley.pdf"), 2)


def synth_speedup(cells):
    """Speedup of ENPOWER_MC-phi++ over ENPOWER_MC-phi, per (n, L, |U|, p)
    where both completed, summarized per p."""
    from statistics import median
    rows = []
    for (t, i, n, L, U, p), sec in cells.items():
        if t == "multiclass" and i == "shapley":
            ref = cells.get((t, "shapley_refined", n, L, U, p))
            if ref:
                rows.append((p, n, L, U, sec / ref))
    if not rows:
        return
    print("\nspeedup of %s over %s (ENPOWER_MC-phi++ over ENPOWER_MC-phi)" % ("shapley_refined", "shapley"))
    print("%6s %6s %6s %6s %10s" % ("p", "n", "L", "|U|", "speedup"))
    for r in sorted(rows):
        print("%6g %6d %6d %6d %9.2fx" % r)
    for p in sorted({r[0] for r in rows}):
        s = [r[4] for r in rows if r[0] == p]
        print("p=%g: median %.2fx, min %.2fx, max %.2fx over %d cells" % (p, median(s), min(s), max(s), len(s)))


def synth_worst(cells):
    """The slowest cell of each algorithm, in total and per unique profile."""
    print("\n%-24s %14s %22s" % ("algorithm", "max time (s)", "max per profile (s)"))
    for key in SYNTH_ALGO_LABELS:
        mine = {k: t for k, t in cells.items() if k[:2] == key}
        if mine:
            print("%-24s %14.4f %22.2e" % ("-".join(key), max(mine.values()),
                  max(t / k[4] for k, t in mine.items())))


def cmd_synth_report(cfg, args):
    res = os.path.join(SYNTH_RESULTS, cfg["name"])
    gens, runs = read_jsonl(os.path.join(res, "gen.jsonl")), synth_runs(res)
    out = args.out or os.path.join(RESULTS, "synth", cfg["name"])
    os.makedirs(out, exist_ok=True)

    # the instances that did not complete, and the reason
    ex = sorted({(g["task"], g["n"], g["L"], g["U"], g["p"]) for g in gens if g["exhausted"]})
    bad = [r for r in runs if r["status"] in ("timeout", "error", "skipped")]
    print("instances: %d generated, %d discarded (exhausted)" % (len(gens), sum(g["exhausted"] for g in gens)))
    for e in ex:
        print("  exhausted: %s n=%d L=%d |U|=%d p=%g" % e)
    for r in bad:
        print("  %s: %s %s" % (r["status"], r["id"], r["index"]))

    # correctness: brute-force comparison on the small instances, and the
    # efficiency of the Shapley values on all instances
    vs = [r for r in runs if r["index"] == "verify" and r["status"] == "ok"]
    gaps = {k: max(r[k] for r in vs if k in r) for k in sorted({k for r in vs for k in r if "_vs_" in k})}
    eff = max((r["efficiency_gap"] for r in runs if "efficiency_gap" in r), default=None)
    print("correctness: %d instances against the brute force; max gaps %s; max efficiency gap %s"
          % (len(vs), {k: v for k, v in gaps.items() if v is not None}, eff))

    cells = synth_cells(runs)
    synth_worst(cells)
    synth_speedup(cells)
    plt = synth_style()
    for p in sorted({k[5] for k in cells}):
        plot_synth_per_profile(plt, cells, p, out)
    print("-> %s" % out)


def cmd_synth(args):
    cfg = synth_config(args.config)
    for stage in (["gen", "run", "report"] if args.stage == "all" else [args.stage]):
        {"gen": cmd_synth_gen, "run": cmd_synth_run, "report": cmd_synth_report}[stage](cfg, args)


# ---------------------------------------------------------------------------
# splits: the binary experiments of the paper over five random splits
# ---------------------------------------------------------------------------
#
# Every split lives under outputs/splits/seed<s>/:
#
#   ensembles/<dataset>_<ensemble>_predictions.npz   python/train_binary.py --seed s
#   baselines/binary_hpc_<id>_<ensemble>.json        the same run: Tr-PS, Tr-MLE
#   cpp/<dataset>_<ensemble>.jsonl                   splits run and splits mc: the C++ methods

SPLITS_ROOT = os.path.join(OUTPUTS, "splits")
SPLITS_DATASETS = {"credit-g": 31, "bank-marketing": 1461,
                   "blood-transfusion-service-center": 1464, "cnae-9": 1468,
                   "titanic": 40945, "Covertype": 150}
SPLITS_SEEDS = [100, 101, 102, 103, 104]
# key -> (arguments of ./solver, part of its JSON holding seconds and values)
SPLITS_CPP = {
    "Our_Exact_Plurality_Banzhaf_Binary": (["banzhaf", "--binary"], None),
    "Our_Exact_Plurality_Shapley_Binary": (["shapley", "--binary"], "binary"),
    "WMG_Banzhaf_Exact": (["wmg"], None),
    "WMG_Shapley_Exact": (["wmg", "--shapley"], None),
    "ISCG_Ranking": (["iscg"], None),
}
# The Monte-Carlo baselines MC-beta and MC-phi (./solver mc), with n^2 samples
# per classifier and the seed of the split; `splits mc` runs them.
SPLITS_MC = {"Our_MC_Banzhaf": ["mc"], "Our_MC_Shapley": ["mc", "--shapley"]}


def splits_dir(seed):
    return os.path.join(SPLITS_ROOT, "seed%d" % seed)


def splits_npz(root, dataset, ensemble):
    """preds_val, labels_val, preds_test, labels_test of one cell."""
    import numpy as np

    path = os.path.join(root, "ensembles", "%s_%s_predictions.npz" % (dataset, ensemble))
    with np.load(path) as d:
        return (d["preds_val"].astype(int), d["labels_val"].astype(int),
                d["preds_test"].astype(int), d["labels_test"].astype(int))


def splits_cpp(preds, labs, method):
    """One C++ method on a dataset: (seconds, values)."""
    cmd, part = SPLITS_CPP[method]
    out = run_solver(cmd, encode(preds, labs, 2))
    out = out[part] if part else out
    return out["seconds"], out["values"]


def splits_cells(args):
    return [(s, d, e) for s in args.seeds for d in args.datasets for e in args.ensembles]


def cmd_splits_run(args):
    """The C++ methods on the validation split of every cell asked for; resumes."""
    for seed, dataset, ensemble in splits_cells(args):
        root = splits_dir(seed)
        try:
            preds, labs, _, _ = splits_npz(root, dataset, ensemble)
        except FileNotFoundError:
            print("seed %d %s/%s: no cached predictions, skipped" % (seed, dataset, ensemble))
            continue
        log = os.path.join(root, CPP_DIR, "%s_%s.jsonl" % (dataset, ensemble))
        os.makedirs(os.path.dirname(log), exist_ok=True)
        done = {r["method"] for r in read_jsonl(log)}
        for method in SPLITS_CPP:
            if method in done:
                continue
            seconds, values = splits_cpp(preds, labs, method)
            append_jsonl(log, dict(seed=seed, dataset=dataset, ensemble=ensemble, method=method,
                                   points=len(labs), models=len(preds[0]),
                                   seconds=seconds, values=values))
            print("seed %d %-34s %-14s %-36s %10.4fs" % (seed, dataset, ensemble, method, seconds),
                  flush=True)
    return 0


def cmd_splits_mc(args):
    """The C++ Monte-Carlo baseline on the validation split of every cell asked
    for, into the same log as `splits run`; resumes."""
    for seed, dataset, ensemble in splits_cells(args):
        root = splits_dir(seed)
        try:
            preds, labs, _, _ = splits_npz(root, dataset, ensemble)
        except FileNotFoundError:
            print("seed %d %s/%s: no cached predictions, skipped" % (seed, dataset, ensemble))
            continue
        log = os.path.join(root, CPP_DIR, "%s_%s.jsonl" % (dataset, ensemble))
        os.makedirs(os.path.dirname(log), exist_ok=True)
        done = {r["method"] for r in read_jsonl(log)}
        for method, cmd in SPLITS_MC.items():
            if method in done:
                continue
            out = run_solver(cmd + ["--seed", str(seed)], encode(preds, labs, 2))
            append_jsonl(log, dict(seed=seed, dataset=dataset, ensemble=ensemble, method=method,
                                   points=len(labs), models=len(preds[0]), samples=out["samples"],
                                   seconds=out["seconds"], values=out["values"]))
            print("seed %d %-34s %-14s %-36s %10.4fs" % (seed, dataset, ensemble, method,
                                                        out["seconds"]), flush=True)
    return 0


def splits_load(seed, dataset, ensemble):
    """{method: {"vals": [...], "seconds": float}} of one cell, from every file
    of the split; methods that did not run are absent."""
    root = splits_dir(seed)
    did = SPLITS_DATASETS[dataset]
    out = {}
    for r in read_jsonl(os.path.join(root, CPP_DIR, "%s_%s.jsonl" % (dataset, ensemble))):
        out[r["method"]] = {"vals": r["values"], "seconds": r["seconds"]}
    # one file per ensemble when the ensembles ran as separate jobs, one per
    # dataset otherwise
    def files(folder, stem):
        return [p for p in (os.path.join(root, folder, "%s_%d_%s.json" % (stem, did, ensemble)),
                            os.path.join(root, folder, "%s_%d.json" % (stem, did)))
                if os.path.exists(p)][:1]

    for path in files("baselines", "binary_hpc"):
        with open(path) as f:
            for e in json.load(f).values():
                for k, v in e["ensembles"].get(ensemble, {}).get("solvers", {}).items():
                    if k in ("Troupe_Permutation", "Troupe_MLE"):
                        out[k] = {"vals": v["vals"], "seconds": v.get("time", v.get("time_sec"))}
    return out


def ranking(vals):
    """Classifiers by decreasing value; equal values by increasing index."""
    return sorted(range(len(vals)), key=lambda i: (-vals[i], i))


def top_k_curves(order, preds_test, labels_test, seed):
    """Accuracy and macro F1 of the plurality vote of the top k, every k.

    Plurality voting with random tie breaking: a point the two classes tie on
    goes to one of them uniformly at random. The coins are drawn from the seed
    of the split, one per (point, k), so every ranking sees the same coins."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    votes = np.zeros(len(labels_test))
    for k, j in enumerate(order, start=1):
        votes += preds_test[:, j]
        coins = rng.integers(0, 2, size=len(labels_test))
        pred = np.where(2 * votes > k, 1, np.where(2 * votes < k, 0, coins))
        acc.append(accuracy_score(labels_test, pred))
        f1.append(f1_score(labels_test, pred, average="macro"))
    return acc, f1


def splits_spines(ax):
    for side, vis in (("top", False), ("right", False), ("bottom", True), ("left", False)):
        ax.spines[side].set_visible(vis)


def cmd_splits_report(args):
    import numpy as np

    sys.path.insert(0, PYTHON_DIR)
    import plot_style

    plot_style.apply()
    rt = plot_style.RT    # labels and colors of the runtime figures
    sc = plot_style.SC    # ... of the Kendall-tau scatter
    ge = plot_style.GE    # ... of the sub-ensemble selection figures
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.lines import Line2D

    out = args.out or os.path.join(RESULTS, "splits")
    for sub_dir in ("runtime", "corr", "generalization"):
        os.makedirs(os.path.join(out, sub_dir), exist_ok=True)

    data = {(s, d, e): splits_load(s, d, e)
            for s in args.seeds for d in args.datasets for e in args.ensembles}
    present = {k: v for k, v in data.items() if v}
    print("cells with results: %d of %d" % (len(present), len(data)))

    # --- Figs. 1 and 6: runtime, mean over the splits, error bars = std ------
    table = {}
    for dataset in args.datasets:
        rows = []
        for (s, d, e), cell in present.items():
            if d != dataset:
                continue
            for method, r in cell.items():
                if method in rt.ALGNAMES and r["seconds"] is not None:
                    rows.append({"Ensemble": "\\texttt{%s}" % rt.ENSNAMES[e],
                                 "Algorithm": rt.ALGNAMES[method],
                                 "Execution Time (s)": r["seconds"]})
                    table.setdefault((dataset, e, method), []).append(r["seconds"])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        order = [a for a in sorted(set(rt.ALGNAMES.values())) if a in set(df["Algorithm"])]
        ens_order = [x for x in ("\\texttt{%s}" % rt.ENSNAMES[e] for e in args.ensembles)
                     if x in set(df["Ensemble"])]
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(data=df, x="Ensemble", y="Execution Time (s)", hue="Algorithm",
                    palette=rt.ALG_COLORS, hue_order=order, order=ens_order, ax=ax,
                    edgecolor="black", linewidth=1, errorbar="sd", capsize=0.05,
                    err_kws={"color": ".2", "linewidth": 1.5})
        ax.set_yscale("log")
        ax.set_ylabel(r"Runtime (sec)", fontsize=16, fontweight="bold")
        ax.set_xlabel(r"Ensemble type", fontsize=16, fontweight="bold")
        ax.set_title(rf"Dataset: {rt.DATANAMES.get(dataset.lower(), dataset)}", fontsize=18, pad=15)
        splits_spines(ax)
        ax.grid(linestyle="--", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        plt.tight_layout()
        fig.savefig(os.path.join(out, "runtime", "rt_%s.pdf" % dataset.lower()), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        lfig, lax = plt.subplots(figsize=(10, 1))
        lax.axis("off")
        leg = lax.legend(handles, labels, loc="center", ncol=9, frameon=False)
        lfig.savefig(os.path.join(out, "runtime", "rt_leg_%s.pdf" % dataset.lower()),
                     bbox_inches="tight", bbox_extra_artists=[leg])
        plt.close(lfig)

    print("\nruntime (s), mean +- std over the splits")
    for (dataset, e, method), ts in sorted(table.items()):
        print("  %-34s %-14s %-36s %12.4f +- %-10.4f (%d splits)"
              % (dataset, e, method, np.mean(ts), np.std(ts), len(ts)))

    # --- Fig. 2: Kendall's tau against ENPOWER_B, mean over the splits -------
    taus = {}
    for (s, d, e), cell in present.items():
        try:
            shap = cell["Our_Exact_Plurality_Shapley_Binary"]["vals"]
            banz = cell["Our_Exact_Plurality_Banzhaf_Binary"]["vals"]
        except KeyError:
            continue
        for b in sc.BASELINES:
            if b in cell:
                taus.setdefault((e, b, d), []).append(
                    (sc.safe_kendall(shap, cell[b]["vals"]), sc.safe_kendall(banz, cell[b]["vals"])))
    fig, axes = plt.subplots(1, 3, figsize=(10, 2), sharey=True, sharex=True)
    for i, e in enumerate(sc.ENSEMBLES):
        ax = axes[i]
        for b in sc.BASELINES:
            for d in sc.TARGET_DATASETS:
                if (e, b, d) in taus:
                    x, y = np.mean(taus[(e, b, d)], axis=0)
                    ax.scatter(x, y, color=sc.COLORS[b], marker=sc.MARKERS[d.lower()], s=110,
                               alpha=0.8, edgecolors="w", linewidth=0.8)
        ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, zorder=0)
        ax.set_title(sc.ENSNAMES[e], fontsize=14, fontweight="bold")
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        splits_spines(ax)
        ax.grid(linestyle="--", alpha=0.6)
        if i == 0:
            ax.set_ylabel(r"Kendall's $\tau$ (vs \textsc{enpower}-$\beta$)", fontsize=12)
        ax.set_xlabel(r"Kendall's $\tau$ (vs \textsc{enpower}-$\phi$)", fontsize=12)
    plt.savefig(os.path.join(out, "corr", "scatter_correlation.pdf"), format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    for name, elements, ncol, title in (
            ("baselines", [Line2D([0], [0], marker="o", color="w", label=l,
                                  markerfacecolor=sc.COLORS[k], markersize=12)
                           for k, l in sc.BASELINES.items()], 4, "Baselines"),
            ("datasets", [Line2D([0], [0], marker=sc.MARKERS[k], color="w", label=l,
                                 markerfacecolor="gray", markersize=12)
                          for k, l in sc.DATASET_LABELS.items()], 6, "Datasets")):
        lfig = plt.figure()
        leg = lfig.legend(handles=elements, loc="center", ncol=ncol, fontsize=12, title=title,
                          title_fontsize=12, frameon=False)
        lfig.canvas.draw()
        bbox = leg.get_window_extent().transformed(lfig.dpi_scale_trans.inverted())
        lfig.savefig(os.path.join(out, "corr", "scatter_legend_%s.pdf" % name), bbox_inches=bbox)
        plt.close(lfig)
    print("\nKendall's tau (vs ENPOWER-phi, vs ENPOWER-beta), mean over the splits")
    for (e, b, d), v in sorted(taus.items()):
        x, y = np.mean(v, axis=0)
        print("  %-14s %-20s %-34s %+.3f %+.3f (%d splits)" % (e, b, d, x, y, len(v)))

    # --- Figs. 3 and 7: sub-ensemble selection, mean over the splits ---------
    # Acc is drawn in gray; the other methods keep their colors
    methods = list(ge.TARGET_ALGS) + ["Acc"]
    colors = dict(zip(sorted(ge.TARGET_ALGS),
                      sns.color_palette("viridis", n_colors=len(ge.TARGET_ALGS))))
    colors["Acc"] = "#7f7f7f"
    names = dict(ge.ALGNAMES, Acc=r"$\textsc{Acc}$")
    legend = None
    for dataset in args.datasets:
        for e in args.ensembles:
            curves = {}
            for s in args.seeds:
                cell = present.get((s, dataset, e))
                if not cell:
                    continue
                pv, lv, pt, lt = splits_npz(splits_dir(s), dataset, e)
                cell = dict(cell, Acc={"vals": list((pv == lv[:, None]).mean(axis=0))})
                for m in methods:
                    if m in cell:
                        curves.setdefault(m, []).append(
                            top_k_curves(ranking(cell[m]["vals"]), pt, lt, s))
            if not curves:
                continue
            M = len(next(iter(curves.values()))[0][0])
            k = np.arange(1, M + 1)
            fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharex=True)
            for m in methods:
                if m not in curves:
                    continue
                acc = np.mean([c[0] for c in curves[m]], axis=0)
                f1 = np.mean([c[1] for c in curves[m]], axis=0)
                ls = ":" if m == "Acc" else ge.get_linestyle(m)
                axes[0].plot(k, acc, color=colors[m], linestyle=ls, linewidth=1.02, label=names[m])
                axes[1].plot(k, f1, color=colors[m], linestyle=ls, linewidth=1.02)
            axes[0].set_ylabel("Accuracy", fontsize=14)
            axes[1].set_ylabel("F1-Score", fontsize=14)
            for ax in axes:
                splits_spines(ax)
                ax.set_xlabel(r"Top $k$ ranked models", fontsize=14)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.set_xticks(np.arange(1, M + 1, max(1, M // 5)))
                ax.tick_params(axis="x", labelsize=8)
            plt.suptitle(f"Dataset: {ge.DATANAMES[dataset.lower()]}, Ensemble: {ge.ENSNAMES[e]}",
                         fontsize=13, y=0.9)
            plt.tight_layout()
            plt.savefig(os.path.join(out, "generalization", "acc_f1_%s_%s.pdf" % (dataset, e)),
                        dpi=300, bbox_inches="tight")
            legend = axes[0].get_legend_handles_labels()
            plt.close(fig)
    if legend:
        lfig, lax = plt.subplots(figsize=(10, 0.5))
        lax.axis("off")
        leg = lax.legend(*legend, loc="center", ncol=len(legend[0]), frameon=False)
        lfig.savefig(os.path.join(out, "generalization", "legend.pdf"), bbox_inches="tight",
                     bbox_extra_artists=[leg])
        plt.close(lfig)

    # --- table: accuracies of the base classifiers ---------------------------
    # mean +- std over the classifiers of their accuracy on D_val, each
    # averaged over the splits (Table 6). When the std is large, every method
    # that rewards correct votes ranks the classifiers by accuracy, and the
    # rankings agree (Fig. 2). One row per ensemble, one column per dataset.
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    order = [d for d in ["blood-transfusion-service-center", "credit-g", "cnae-9", "titanic",
                         "bank-marketing", "Covertype"] if d in args.datasets]
    cell = {}
    print("\naccuracy of the base classifiers on D_val, mean +- std over the classifiers, "
          "mean over the splits")
    for dataset in order:
        for e in args.ensembles:
            stats = []
            for s in args.seeds:
                if (s, dataset, e) in present:
                    pv, lv, _, _ = splits_npz(splits_dir(s), dataset, e)
                    acc = (pv == lv[:, None]).mean(axis=0)
                    stats.append((acc.mean(), acc.std()))
            cell[dataset, e] = ("$%.3f \\pm %.3f$" % tuple(np.mean(stats, axis=0))
                                if stats else "--")
            print("  %-34s %-14s %s" % (dataset, e, cell[dataset, e]))
    lines = ["% generated by cpp/tools/enpower.py splits report, do not edit by hand",
             r"\begin{tabular}{l" + "c" * len(order) + "}", r"\toprule",
             "Ensemble & " + " & ".join(rt.DATANAMES[d.lower()] for d in order) + r" \\",
             r"\midrule"]
    for e in args.ensembles:
        lines.append(r"\texttt{%s} & %s \\" % (rt.ENSNAMES[e],
                                                " & ".join(cell[d, e] for d in order)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "tables", "acc_spread.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- Table 4: approximation error of MC-beta and MC-phi --------------------
    # per split, the largest absolute and relative error over the classifiers
    # and the rank correlations with the exact values; each entry is the
    # mean +- std over the splits
    from scipy.stats import kendalltau, spearmanr

    def mc_metrics(exact, mc):
        exact, mc = np.asarray(exact), np.asarray(mc)
        err = np.abs(exact - mc)
        tau, rho = kendalltau(exact, mc)[0], spearmanr(exact, mc)[0]
        return (err.max(), (err / (np.abs(exact) + 1e-12)).max(),
                0.0 if np.isnan(tau) else tau, 0.0 if np.isnan(rho) else rho)

    indices = (("Shapley", r"$\shap$"), ("Banzhaf", r"$\banz$"))
    lines = ["% generated by cpp/tools/enpower.py splits report, do not edit by hand",
             r"\begin{tabular}{lll cccc}", r"\toprule",
             r"Dataset & Ensemble & Index & $\varepsilon_{\text{abs}}$ & "
             r"$\varepsilon_{\text{rel}}$ & Kendall's ($\tau$) & Spearman ($\rho$) \\"]
    print("\nMC approximation error (max abs, max rel, tau, rho), mean +- std over the splits")
    for dataset in order:
        block = []
        for e in args.ensembles:
            rows = []
            for index, symbol in indices:
                m = [mc_metrics(present[s, dataset, e]["Our_Exact_Plurality_%s_Binary" % index]["vals"],
                                present[s, dataset, e]["Our_MC_%s" % index]["vals"])
                     for s in args.seeds
                     if "Our_MC_%s" % index in present.get((s, dataset, e), {})
                     and "Our_Exact_Plurality_%s_Binary" % index in present[s, dataset, e]]
                if not m:
                    continue
                mean, std = np.mean(m, axis=0), np.std(m, axis=0)
                rows.append((symbol, "$%.4f \\pm %.4f$" % (mean[0], std[0]),
                             "$%.2f \\pm %.2f$" % (mean[1], std[1]),
                             "$%.3f \\pm %.3f$" % (mean[2], std[2]),
                             "$%.3f \\pm %.3f$" % (mean[3], std[3])))
                print("  %-34s %-14s %-8s %s (%d splits)" % (dataset, e, index,
                                                             " ".join(rows[-1][1:]), len(m)))
            if rows:
                block.append((e, rows))
        if not block:
            continue
        lines.append(r"\midrule")
        for b, (e, rows) in enumerate(block):
            if b:
                lines.append(r"\cmidrule{2-7}")
            for r, row in enumerate(rows):
                head = (r"\multirow{%d}{*}{%s}" % (sum(len(x) for _, x in block), rt.DATANAMES[dataset.lower()])
                        if b == 0 and r == 0 else "")
                ens = (r"\multirow{%d}{*}{\texttt{%s}}" % (len(rows), rt.ENSNAMES[e])
                       if r == 0 else "")
                lines.append("%s & %s & %s \\\\" % (head, ens, " & ".join(row)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "tables", "mc_err.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nfigures and tables -> %s" % os.path.abspath(out))
    return 0


def cmd_splits_cells(args):
    """The cells of the experiment, one per line: seed dataset id ensemble."""
    for seed, dataset, ensemble in splits_cells(args):
        print(seed, dataset, SPLITS_DATASETS[dataset], ensemble)
    return 0


def cmd_splits(args):
    return {"run": cmd_splits_run, "report": cmd_splits_report,
            "cells": cmd_splits_cells, "mc": cmd_splits_mc}[args.stage](args)


# ---------------------------------------------------------------------------
# mc: runtime for multiclass classification (Table 5) over five random splits
# ---------------------------------------------------------------------------
#
# python/train_multiclass.py --seed s --out-root outputs/mc_splits/seed<s>
# trains the three ensembles of split s on the labels of the dataset and stores
#
#   ensembles_multiclass/<dataset>_<ensemble>_predictions.npz
#
# `mc run` measures the runtime of the C++ algorithms on D_val of each cell,
# with one record per algorithm in
#
#   cpp/<dataset>_<ensemble>.jsonl

MC_ROOT = os.path.join(OUTPUTS, "mc_splits")
MC_SEEDS = SPLITS_SEEDS
# key -> (arguments of ./solver, part of its JSON holding seconds and values)
MC_CPP = {
    "Multiclass_Banzhaf": (["banzhaf"], None),
    "Multiclass_Shapley": (["shapley"], "baseline"),
    "Multiclass_Shapley_Refined": (["shapley"], "refined"),
}


def mc_dir(seed):
    return os.path.join(MC_ROOT, "seed%d" % seed)


def mc_npz(seed, dataset, ensemble):
    """preds_val, labels_val of one cell, and its number of labels."""
    import numpy as np

    path = os.path.join(mc_dir(seed), "ensembles_multiclass",
                        "%s_%s_predictions.npz" % (dataset, ensemble))
    with np.load(path) as d:
        preds, labs = d["preds_val"].astype(int), d["labels_val"].astype(int)
    # the L of train_multiclass.py
    return preds, labs, int(max(preds.max(), labs.max()) + 1)


def mc_cells(args):
    return [(s, d, e) for s in args.seeds for d in args.datasets for e in args.ensembles]


def cmd_mc_run(args):
    """The C++ algorithms on D_val of every selected cell; skips the records
    already present. Both Shapley records come from one invocation of
    ./solver shapley, which runs ENPOWER_MC-phi and ENPOWER_MC-phi++ one after
    the other."""
    for seed, dataset, ensemble in mc_cells(args):
        try:
            preds, labs, labels = mc_npz(seed, dataset, ensemble)
        except FileNotFoundError:
            print("seed %d %s/%s: no cached predictions, skipped" % (seed, dataset, ensemble))
            continue
        log = os.path.join(mc_dir(seed), CPP_DIR, "%s_%s.jsonl" % (dataset, ensemble))
        os.makedirs(os.path.dirname(log), exist_ok=True)
        done = {r["method"] for r in read_jsonl(log)}
        outs = {}
        for method, (cmd, part) in MC_CPP.items():
            if method in done:
                continue
            key = tuple(cmd)
            if key not in outs:
                outs[key] = run_solver(cmd, encode(preds, labs, labels))
            out = outs[key][part] if part else outs[key]
            append_jsonl(log, dict(seed=seed, dataset=dataset, ensemble=ensemble, method=method,
                                   points=len(labs), models=len(preds[0]), labels=labels,
                                   profiles=outs[key].get("profiles"),
                                   seconds=out["seconds"], values=out["values"]))
            print("seed %d %-10s %-14s %-28s %10.4fs" % (seed, dataset, ensemble, method,
                                                         out["seconds"]), flush=True)
    return 0


def mc_load(seed, dataset, ensemble):
    """{method: record} of one cell; methods that did not run are absent."""
    return {r["method"]: r for r in read_jsonl(os.path.join(
        mc_dir(seed), CPP_DIR, "%s_%s.jsonl" % (dataset, ensemble)))}


def cmd_mc_report(args):
    """Table 5: for each dataset and ensemble, the runtime of each algorithm,
    mean and std over the splits."""
    import numpy as np

    out = args.out or os.path.join(RESULTS, "mc")
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    names = {"cnae-9": r"\texttt{C9}", "Covertype": r"\texttt{CT}"}
    methods = [("Multiclass_Banzhaf", r"\suite-$\banz$"), ("Multiclass_Shapley", r"\suite-$\shap$"),
               ("Multiclass_Shapley_Refined", r"\suiteShapImp")]
    models = {"random_forest": 50, "adaboost": 150, "heterogeneous": 30}
    titles = {"random_forest": r"\texttt{RF}", "adaboost": r"\texttt{AB}",
              "heterogeneous": r"\texttt{HET}"}

    def cell(xs):
        # two significant digits on the mean and the std to the same decimal;
        # below 0.1 s both share a factor 10^{-k}, so the column stays narrow
        mean, std = np.mean(xs), np.std(xs)
        exp = int(np.floor(np.log10(mean)))
        if exp >= -1:
            dec = max(0, 1 - exp)
            return "$%.*f \\pm %.*f$" % (dec, mean, dec, std)
        return "$(%.1f \\pm %.1f) \\cdot 10^{%d}$" % (mean / 10.0 ** exp, std / 10.0 ** exp, exp)

    print("runtime (s), mean +- std over the splits")
    rows = []
    for e in args.ensembles:
        group = []
        for m, label in methods:
            row = [label]
            for d in args.datasets:
                ts = [r[m]["seconds"] for r in (mc_load(s, d, e) for s in args.seeds) if m in r]
                row.append(cell(ts) if ts else "--")
                if ts:
                    print("  %-10s %-14s %-28s %10.4f +- %.4f  (%d splits)"
                          % (d, e, m, np.mean(ts), np.std(ts), len(ts)))
            group.append(row)
        group[0].insert(0, r"\multirow{%d}{*}{%s ($\cardM=%d$)}"
                        % (len(methods), titles[e], models[e]))
        for row in group[1:]:
            row.insert(0, "")
        rows.append(group)
    lines = ["% generated by cpp/tools/enpower.py mc report, do not edit by hand",
             r"\begin{tabular}{ll" + "r" * len(args.datasets) + "}", r"\toprule",
             "Ensemble & Method & " + " & ".join(names.get(d, d) for d in args.datasets)
             + r" \\", r"\midrule"]
    for i, group in enumerate(rows):
        if i:
            lines.append(r"\midrule")
        lines += [" & ".join(r) + r" \\" for r in group]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(out, "tables", "mc_runtime.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\ntable -> %s" % os.path.abspath(path))
    if args.paper:
        import shutil

        os.makedirs(os.path.join(args.paper, "tables"), exist_ok=True)
        shutil.copy(path, os.path.join(args.paper, "tables"))
        print("copied -> %s" % os.path.abspath(os.path.join(args.paper, "tables")))
    return 0


def cmd_mc_cells(args):
    """The cells of the experiment, one per line: seed dataset id ensemble."""
    for seed, dataset, ensemble in mc_cells(args):
        print(seed, dataset, DATASETS[dataset], ensemble)
    return 0


def cmd_mc(args):
    return {"run": cmd_mc_run, "report": cmd_mc_report,
            "cells": cmd_mc_cells}[args.stage](args)


# ---------------------------------------------------------------------------
# clip: prompt ensembling of CLIP (Sec. 5.4) over five random splits
# ---------------------------------------------------------------------------
#
# The only input is the vote of each of the 22 prompt templates on each image,
# in the order of the torchvision dataset:
#
#   outputs/clip/<dataset>_ViT-B-32_votes.npz      preds (N x 22), labels (N)
#
# `clip votes` writes it by CLIP inference (python/clip_inference.py
# --votes-only) when it is missing.
#
# Split s is the procedure of clip_inference.py with seed s: a permutation of
# the N images from np.random.seed(s), the first int(N / 2) to D_val, the rest
# to D_test. Everything of split s lives under outputs/clip_splits/seed<s>/:
#
#   <dataset>_ViT-B-32_predictions.npz   the split
#   cpp/<dataset>.jsonl                  clip run: one record per method

CLIP_MODEL = "ViT-B-32"
CLIP_DATASETS = ["cifar100", "imagenet-a", "pets"]
CLIP_CLASSES = {"cifar100": 100, "imagenet-a": 200, "pets": 37}
CLIP_SEEDS = SPLITS_SEEDS
CLIP_VOTES = os.path.join(OUTPUTS, "clip")
CLIP_ROOT = os.path.join(OUTPUTS, "clip_splits")
# Troupe as in train_binary.py: weight 1/n per correct classifier, a
# coalition wins when its weight exceeds the quota, i.e. with more than n/2
# correct classifiers; Tr-PS with the permutations of the binary splits
CLIP_QUOTA = 0.5 + 1e-5
CLIP_PS_SAMPLES = 1000
# method key -> label in the figures and tables, in plotting order
CLIP_METHODS = {
    "Acc": r"\textsc{Acc}",
    "Our_Exact_Plurality_Banzhaf": r"\textsc{enpower}-$\beta$",
    "Our_Exact_Plurality_Shapley": r"\textsc{enpower}-$\phi$",
    "Troupe_Permutation": r"$\mathtt{Tr\text{-}PS}$",
    "Troupe_MLE": r"$\mathtt{Tr\text{-}MLE}$",
}
# markers of the CLIP figures; colors are tab10 in this order
CLIP_MARKERS = {"Acc": ".", "Our_Exact_Plurality_Banzhaf": "v",
                "Our_Exact_Plurality_Shapley": "^", "Troupe_Permutation": "s",
                "Troupe_MLE": "p"}


def clip_votes_path(dataset):
    return os.path.join(CLIP_VOTES, "%s_%s_votes.npz" % (dataset, CLIP_MODEL))


def clip_dir(seed):
    return os.path.join(CLIP_ROOT, "seed%d" % seed)


def clip_split_path(seed, dataset):
    return os.path.join(clip_dir(seed), "%s_%s_predictions.npz" % (dataset, CLIP_MODEL))


def clip_permutation(N, seed):
    """The permutation of clip_inference.py: np.random.seed(seed), then
    np.random.permutation(N); RandomState(seed) draws the same stream."""
    import numpy as np

    return np.random.RandomState(seed).permutation(N)


def clip_split(preds, labels, seed):
    """preds_val, labels_val, preds_test, labels_test of split seed."""
    perm = clip_permutation(len(labels), seed)
    h = int(len(labels) * 0.5)
    return preds[perm[:h]], labels[perm[:h]], preds[perm[h:]], labels[perm[h:]]


def clip_load_votes(dataset):
    import numpy as np

    path = clip_votes_path(dataset)
    if not os.path.exists(path):
        raise SystemExit("%s missing: run `enpower.py clip votes` (CLIP inference)" % path)
    with np.load(path) as d:
        return d["preds"].astype(np.int64), d["labels"].astype(np.int64)


def clip_load_split(seed, dataset):
    """The split of seed, written from the votes the first time it is asked for."""
    import numpy as np

    path = clip_split_path(seed, dataset)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pv, lv, pt, lt = clip_split(*clip_load_votes(dataset), seed)
        np.savez_compressed(path, preds_val=pv, labels_val=lv, preds_test=pt, labels_test=lt)
    with np.load(path) as d:
        return tuple(d[k].astype(np.int64)
                     for k in ("preds_val", "labels_val", "preds_test", "labels_test"))


def clip_cells(args):
    return [(s, d) for s in args.seeds for d in args.datasets]


def cmd_clip_votes(args):
    """CLIP inference for every dataset whose vote cache is missing."""
    for dataset in args.datasets:
        path = clip_votes_path(dataset)
        if os.path.exists(path):
            print("%s: %s exists, skipped" % (dataset, path))
            continue
        print("%s: CLIP inference -> %s" % (dataset, path), flush=True)
        subprocess.run([sys.executable, "clip_inference.py", "--dataset", dataset,
                        "--votes-only", "--votes_dir", CLIP_VOTES],
                       cwd=PYTHON_DIR, check=True)
    return 0


def clip_troupe():
    """The two solvers of the Troupe library, as train_binary.py imports them."""
    sys.path.insert(0, os.path.join(PYTHON_DIR, "shapley"))
    from shapley import MultilinearExtension, PermutationSampler

    return PermutationSampler, MultilinearExtension


def clip_method(method, preds, labs, labels, seed):
    """One ranking method on D_val: (seconds, values, extra fields)."""
    import numpy as np

    if method == "Acc":
        start = time.time()
        values = (preds == labs[:, None]).mean(axis=0)
        return time.time() - start, [float(v) for v in values], {}
    if method == "Our_Exact_Plurality_Banzhaf":
        out = run_solver(["banzhaf"], encode(preds, labs, labels))
        return out["seconds"], out["values"], {}
    if method == "Our_Exact_Plurality_Shapley":
        # the values and runtime of ENPOWER_MC-phi are reported; ENPOWER_MC-phi++
        # computes the same values, and its runtime is stored as well
        out = run_solver(["shapley"], encode(preds, labs, labels))
        base, refined = out["baseline"], out["refined"]
        return base["seconds"], base["values"], {
            "refined_seconds": refined["seconds"],
            "refined_gap": gap(base["values"], refined["values"])}
    PermutationSampler, MultilinearExtension = clip_troupe()
    W = np.where(preds == labs[:, None], 1.0 / preds.shape[1], 0.0)
    if method == "Troupe_Permutation":
        solver, extra = PermutationSampler(CLIP_PS_SAMPLES, seed), {"permutations": CLIP_PS_SAMPLES}
    else:
        solver, extra = MultilinearExtension(), {}
    start = time.time()
    solver.solve_game(W, q=CLIP_QUOTA)
    seconds = time.time() - start
    return seconds, [float(v) for v in solver.get_solution().mean(axis=0)], extra


def cmd_clip_run(args):
    """Every ranking method on D_val of every selected cell; skips the
    records already present."""
    for seed, dataset in clip_cells(args):
        pv, lv, _, _ = clip_load_split(seed, dataset)
        n, labels = pv.shape[1], CLIP_CLASSES[dataset]
        # fraction of data points with at most n/2 correct votes, on which the
        # whole ensemble does not reach the quota of the game of Troupe
        no_majority = float(((pv == lv[:, None]).sum(axis=1) <= n / 2).mean())
        log = os.path.join(clip_dir(seed), CPP_DIR, "%s.jsonl" % dataset)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        done = {r["method"] for r in read_jsonl(log)}
        for method in CLIP_METHODS:
            if method in done:
                continue
            seconds, values, extra = clip_method(method, pv, lv, labels, seed)
            append_jsonl(log, dict(seed=seed, dataset=dataset, model=CLIP_MODEL, method=method,
                                   points=len(lv), models=n, labels=labels,
                                   no_majority=no_majority, seconds=seconds, values=values,
                                   **extra))
            print("seed %d %-10s %-28s %10.4fs" % (seed, dataset, method, seconds), flush=True)
    return 0


def cmd_clip_cells(args):
    """The cells of the experiment, one per line: seed dataset."""
    for seed, dataset in clip_cells(args):
        print(seed, dataset)
    return 0


def top_k_curves_mc(order, preds_test, labels_test, labels, seed):
    """Accuracy, macro F1 and tie rate of the plurality vote of the top k,
    every k. Plurality voting with random tie breaking: a point on which
    several labels tie for the most votes goes to one of them uniformly at
    random. The draws come from the seed of the split, one per (point, k), so
    every ranking sees the same draws."""
    import numpy as np
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    rows = np.arange(len(labels_test))
    counts = np.zeros((len(labels_test), labels), dtype=np.int64)
    acc, f1, ties = [], [], []
    for j in order:
        counts[rows, preds_test[:, j]] += 1
        tied = counts == counts.max(axis=1, keepdims=True)
        noise = rng.random(counts.shape)
        pred = np.where(tied, noise, -1.0).argmax(axis=1)
        acc.append(float((pred == labels_test).mean()))
        f1.append(float(f1_score(labels_test, pred, average="macro")))
        ties.append(float((tied.sum(axis=1) > 1).mean()))
    return acc, f1, ties


def clip_load(seed, dataset):
    """{method: record} of one cell; methods that did not run are absent."""
    return {r["method"]: r for r in read_jsonl(os.path.join(clip_dir(seed), CPP_DIR,
                                                            "%s.jsonl" % dataset))}


def clip_style(notex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not notex:
        # LaTeX text with Libertine fonts, as in the figures of the paper
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=(
            r"\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} "
            r"\RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath} "
            r"\RequirePackage{xfrac} \RequirePackage{amsmath}"))
    return plt


def clip_table(path, header, rows, left=2, groups=True):
    """A booktabs tabular; with groups, a rule before every row whose first
    cell is not empty (the first row of a dataset)."""
    lines = ["% generated by cpp/tools/enpower.py clip report, do not edit by hand",
             r"\begin{tabular}{" + "l" * left + "c" * (len(header) - left) + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        if groups and i and row[0]:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_clip_report(args):
    import numpy as np

    plt = clip_style(args.notex)
    import seaborn as sns

    out = args.out or os.path.join(RESULTS, "clip")
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    colors = dict(zip(CLIP_METHODS, sns.color_palette("tab10")))
    names = {"cifar100": "CIFAR100", "imagenet-a": "IMAGENET-A", "pets": "PETS"}

    summary, runtime, legend = [], {}, None
    for dataset in args.datasets:
        curves, seconds, no_majority = {}, {}, []
        for s in args.seeds:
            cell = clip_load(s, dataset)
            if not cell:
                continue
            _, _, pt, lt = clip_load_split(s, dataset)
            no_majority.append(next(iter(cell.values()))["no_majority"])
            for m, r in cell.items():
                curves.setdefault(m, []).append(
                    top_k_curves_mc(ranking(r["values"]), pt, lt, CLIP_CLASSES[dataset], s))
                seconds.setdefault(m, []).append(r["seconds"])
                if m == "Our_Exact_Plurality_Shapley":
                    seconds.setdefault("refined", []).append(r["refined_seconds"])
        if not curves:
            print("%s: no results" % dataset)
            continue
        missing = [m for m in CLIP_METHODS if len(curves.get(m, [])) != len(no_majority)]
        print("\n%s: %d splits%s" % (dataset, len(no_majority),
                                     ", incomplete: " + ", ".join(missing) if missing else ""))
        print("  points of D_val with at most n/2 correct prompts: %.3f +- %.3f"
              % (np.mean(no_majority), np.std(no_majority)))

        n = len(next(iter(curves.values()))[0][0])
        k = np.arange(1, n + 1)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for m in CLIP_METHODS:
            if m not in curves:
                continue
            for ax, idx in zip(axes, (0, 1)):
                runs = np.array([c[idx] for c in curves[m]])
                mean, std = runs.mean(axis=0), runs.std(axis=0)
                ax.plot(k, mean, label=CLIP_METHODS[m], color=colors[m], linewidth=2,
                        marker=CLIP_MARKERS[m], markerfacecolor="none",
                        markeredgecolor=colors[m])
                ax.fill_between(k, mean - std, mean + std, color=colors[m], alpha=0.15,
                                linewidth=0)
            acc = np.array([c[0] for c in curves[m]])
            f1 = np.array([c[1] for c in curves[m]])
            ties = np.array([c[2] for c in curves[m]])
            ka, kf = int(acc.mean(axis=0).argmax()), int(f1.mean(axis=0).argmax())
            summary.append((dataset, m, acc[:, ka], ka + 1, acc.mean(axis=1),
                            f1[:, kf], kf + 1, f1.mean(axis=1)))
            print("  %-28s best acc %.4f +- %.4f (k=%2d)  best F1 %.4f +- %.4f (k=%2d)  "
                  "ties max %.3f" % (m, acc[:, ka].mean(), acc[:, ka].std(), ka + 1,
                                     f1[:, kf].mean(), f1[:, kf].std(), kf + 1,
                                     ties.mean(axis=0).max()))
        for ax, title in zip(axes, ("Accuracy", "$F1$-score")):
            ax.set_xlabel("Top $k$ prompts", fontsize=14)
            ax.set_ylabel(title, fontsize=14)
            ax.set_xticks(range(1, n + 1, max(1, n // 10)))
            ax.grid(True, linestyle="--", alpha=0.6)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
        plt.suptitle("Prompt selection (%s)" % dataset.upper(), fontsize=18, y=0.95)
        plt.tight_layout()
        fig.savefig(os.path.join(out, "%s_%s_top_k.pdf" % (dataset, CLIP_MODEL)),
                    format="pdf", bbox_inches="tight")
        legend = axes[0].get_legend_handles_labels()
        plt.close(fig)
        for m, ts in seconds.items():
            runtime[dataset, m] = ts

    if legend:
        lfig, lax = plt.subplots(figsize=(6, 0.5))
        lax.axis("off")
        lax.legend(*legend, loc="center", ncol=len(legend[0]), frameon=False)
        lfig.savefig(os.path.join(out, "legend.pdf"), bbox_inches="tight")
        plt.close(lfig)

    def pm(xs, fmt="%.3f", bold=False):
        cell = ("$" + fmt + " \\pm " + fmt + "$") % (np.mean(xs), np.std(xs))
        # \boldmath, so that \pm is bold too
        return "{\\boldmath" + cell + "}" if bold else cell

    # bold the largest mean of each metric within a dataset, and every mean
    # tied with it at the printed precision
    best = {}
    for dataset, _, *metrics in summary:
        for i in (0, 2, 3, 5):
            r = round(float(np.mean(metrics[i])), 3)
            best[dataset, i] = max(best.get((dataset, i), r), r)
    rows, last = [], None
    for dataset, m, *metrics in summary:
        acc, ka, acc_avg, f1, kf, f1_avg = [
            str(x) if i in (1, 4) else
            pm(x, bold=round(float(np.mean(x)), 3) == best[dataset, i])
            for i, x in enumerate(metrics)]
        rows.append([names[dataset] if dataset != last else "", CLIP_METHODS[m],
                     acc, ka, acc_avg, f1, kf, f1_avg])
        last = dataset
    clip_table(os.path.join(out, "tables", "clip_summary.tex"),
               ["Dataset", "Ranking", "best acc.", "$k$", "avg.\\ acc.", "best $F_1$", "$k$",
                "avg.\\ $F_1$"], rows)

    timed = [("Our_Exact_Plurality_Banzhaf", CLIP_METHODS["Our_Exact_Plurality_Banzhaf"]),
             ("Our_Exact_Plurality_Shapley", CLIP_METHODS["Our_Exact_Plurality_Shapley"]),
             ("Troupe_Permutation", CLIP_METHODS["Troupe_Permutation"]),
             ("Troupe_MLE", CLIP_METHODS["Troupe_MLE"])]
    present = [d for d in args.datasets if any((d, m) in runtime for m, _ in timed)]

    def sci(xs):
        # every cell as (mean +- std) 10^k, three significant digits on the mean,
        # so that no std rounds to zero
        mean, std = np.mean(xs), np.std(xs)
        exp = int(np.floor(np.log10(mean)))
        return "$(%.2f \\pm %.2f) \\cdot 10^{%d}$" % (mean / 10.0 ** exp, std / 10.0 ** exp, exp)

    print("\nruntime (s), mean +- std over the splits")
    rows = []
    for m, label in timed:
        rows.append([label] + [sci(runtime[d, m]) if (d, m) in runtime else "--"
                                   for d in present])
        for d in present:
            if (d, m) in runtime:
                print("  %-10s %-28s %10.4f +- %.4f" % (d, m, np.mean(runtime[d, m]),
                                                        np.std(runtime[d, m])))
    clip_table(os.path.join(out, "tables", "clip_runtime.tex"),
               ["Method"] + [names[d] for d in present], rows, left=1, groups=False)
    clip_dataset_tables(args, out)
    print("\nfigures and tables -> %s" % os.path.abspath(out))
    if args.paper:
        import glob
        import shutil

        for sub, pattern in (("plots/CLIP", "*.pdf"), ("tables", "tables/*.tex")):
            os.makedirs(os.path.join(args.paper, sub), exist_ok=True)
            for f in glob.glob(os.path.join(out, pattern)):
                shutil.copy(f, os.path.join(args.paper, sub))
        print("copied -> %s" % os.path.abspath(args.paper))
    return 0


def clip_profiles(preds, labs, labels):
    """The unique vote profiles of a split, with the key of the cache of
    shapley_multiclass.py: the correct votes, and the sorted positive counts of
    the incorrect labels."""
    import numpy as np

    keys = set()
    for p, y in zip(preds, labs):
        c = np.bincount(p, minlength=labels)
        keys.add((int(c[y]), tuple(sorted(int(x) for i, x in enumerate(c) if i != y and x))))
    return len(keys)


def clip_imbalance(labels):
    """min, max, IR, CV (%) and normalized entropy of the class counts."""
    import numpy as np

    counts = np.unique(labels, return_counts=True)[1].astype(float)
    p = counts / counts.sum()
    return (len(counts), int(counts.min()), int(counts.max()), counts.max() / counts.min(),
            100.0 * counts.std() / counts.mean(), -(p * np.log(p)).sum() / np.log(len(counts)))


def clip_dataset_tables(args, out):
    """The statistics of the datasets (Table 2) and of their classes
    (Table 3). Sizes and classes do not depend on the split, which is
    verified; |U| depends on the split and is reported as mean and std over
    the splits."""
    import numpy as np

    names = {"cifar100": "CIFAR (Cifar-100)", "pets": "PETS (Oxford-IIIT Pets)",
             "imagenet-a": "IMAGENET-A (ImageNet-A)"}

    def num(n):
        return ("{:,}".format(n)).replace(",", "\\,")

    sizes, classes = [], []
    print("\ndatasets, over %d splits" % len(args.seeds))
    for d in [d for d in names if d in args.datasets]:
        fixed, profiles = set(), []
        for s in args.seeds:
            pv, lv, pt, lt = clip_load_split(s, d)
            fixed.add((len(lv), len(lt), clip_imbalance(np.concatenate([lv, lt]))))
            profiles.append(clip_profiles(pv, lv, CLIP_CLASSES[d]))
        if len(fixed) != 1:
            raise SystemExit("%s: sizes or classes differ across splits" % d)
        val, test, (L, lo, hi, ir, cv, hn) = fixed.pop()
        print("  %-10s |D_val| %d |D_test| %d  |U| %s  mean %.1f +- %.1f"
              % (d, val, test, profiles, np.mean(profiles), np.std(profiles)))
        sizes.append([names[d], num(val + test), str(CLIP_CLASSES[d]), num(val), num(test),
                      "$%.1f \\pm %.1f$" % (np.mean(profiles), np.std(profiles))])
        classes.append([names[d], str(L), str(lo), str(hi), "%.2f" % ir, "%.1f" % cv,
                        "%.3f" % hn])
    clip_table(os.path.join(out, "tables", "clip_datasets.tex"),
               ["Dataset", "Total images", r"Classes (\totLabs)", r"$|\dataSet_{\text{val}}|$",
                r"$|\dataSet_{\text{test}}|$", "$|U|$"], sizes, left=1, groups=False)
    clip_table(os.path.join(out, "tables", "clip_imbalance.tex"),
               ["Dataset", r"Classes ($\totLabs$)", r"$\min$", r"$\max$", "$IR$", r"CV (\%)",
                r"$H_{\text{norm}}$"], classes, left=1, groups=False)


def cmd_clip(args):
    return {"votes": cmd_clip_votes, "cells": cmd_clip_cells,
            "run": cmd_clip_run, "report": cmd_clip_report}[args.stage](args)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="diff the multiclass Shapley values against shapley_multiclass.py")
    c.add_argument("--points", type=int, default=20)
    c.add_argument("--models", type=int, default=12)
    c.add_argument("--labels", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--profile", metavar="N1:N2,N3,...",
                   help="one data point with this vote profile, instead of a random dataset")
    c.add_argument("--float", dest="float_coef", action="store_true",
                   help="long double coefficients, to reproduce what they lose")
    c.set_defaults(func=cmd_check)

    b = sub.add_parser("banzhaf",
                       help="diff the multiclass Banzhaf values and banzhaf_multiclass.py "
                            "against the definition")
    b.add_argument("--points", type=int, default=20)
    b.add_argument("--models", type=int, default=12)
    b.add_argument("--labels", type=int, default=4)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--profile", metavar="N1:N2,N3,...",
                   help="one data point with this vote profile, instead of a random dataset")
    b.add_argument("--float", dest="float_coef", action="store_true",
                   help="long double coefficients, to reproduce what they lose")
    b.set_defaults(func=cmd_banzhaf)

    n = sub.add_parser("binary", help="compare the closed forms of ENPOWER_B with indices_binary.py")
    n.add_argument("--points", type=int, default=200)
    n.add_argument("--models", type=int, default=12)
    n.add_argument("--seed", type=int, default=0)
    n.set_defaults(func=cmd_binary)

    d = sub.add_parser("dataset",
                       help="cached .npz predictions -> the text dataset of ./solver")
    d.add_argument("npz", help="a *_predictions.npz of outputs/")
    d.add_argument("out", help="where to write the text dataset")
    d.add_argument("--split", default="val", choices=["val", "test"])
    d.set_defaults(func=cmd_dataset)

    y = sub.add_parser("synth", help="synthetic experiments (Appendix G.6): gen, run, report")
    y.add_argument("stage", choices=["gen", "run", "report", "all"])
    y.add_argument("--config", default="small",
                   help="a file of experiments/synth/, by name or path (default: small)")
    y.add_argument("--timeout", type=int, default=None,
                   help="seconds per run, overriding the config")
    y.add_argument("--out", default=None,
                   help="directory of the table and figures (default: the results directory)")
    y.add_argument("--shard", type=int, default=0, help="run: the part of the grid to run")
    y.add_argument("--shards", type=int, default=1, help="run: the parts the grid is split into")
    y.set_defaults(func=cmd_synth)

    s = sub.add_parser("splits", help="binary classification experiments over random splits: run, mc, report, cells")
    s.add_argument("stage", choices=["run", "mc", "report", "cells"])
    s.add_argument("--seeds", nargs="*", type=int, default=SPLITS_SEEDS)
    s.add_argument("--datasets", nargs="*", default=list(SPLITS_DATASETS),
                   choices=list(SPLITS_DATASETS))
    s.add_argument("--ensembles", nargs="*", default=ENSEMBLES, choices=ENSEMBLES)
    s.add_argument("--out", default=None,
                   help="report: directory of the figures (default: results/splits)")
    s.set_defaults(func=cmd_splits)

    r = sub.add_parser("mc", help="runtime for multiclass classification over random splits: "
                                  "run, report, cells")
    r.add_argument("stage", choices=["run", "report", "cells"])
    r.add_argument("--seeds", nargs="*", type=int, default=MC_SEEDS)
    r.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    r.add_argument("--ensembles", nargs="*", default=ENSEMBLES, choices=ENSEMBLES)
    r.add_argument("--out", default=None,
                   help="report: directory of the table (default: results/mc)")
    r.add_argument("--paper", default=None, metavar="DIR",
                   help="report: also copy the table to DIR/tables")
    r.set_defaults(func=cmd_mc)

    a = sub.add_parser("clip", help="CLIP prompt ensembles over random splits: "
                                    "votes, cells, run, report")
    a.add_argument("stage", choices=["votes", "cells", "run", "report"])
    a.add_argument("--seeds", nargs="*", type=int, default=CLIP_SEEDS)
    a.add_argument("--datasets", nargs="*", default=CLIP_DATASETS, choices=CLIP_DATASETS)
    a.add_argument("--out", default=None,
                   help="report: directory of the figures and tables (default: results/clip)")
    a.add_argument("--notex", action="store_true", help="report: no LaTeX in the figures")
    a.add_argument("--paper", default=None, metavar="DIR",
                   help="report: also copy the figures to DIR/plots/CLIP and the tables "
                        "to DIR/tables")
    a.set_defaults(func=cmd_clip)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
