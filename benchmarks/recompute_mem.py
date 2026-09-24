"""Recompute Util / Mem / Avg / Gibb for existing TOFU runs with mem_metric.py.

Walks one or more directories for TOFU_EVAL.json files (no model is loaded),
computes the metrics per run, then groups runs that differ only by seed and
reports mean and SE (= sample std / sqrt(n)).

    python benchmarks/recompute_mem.py RUNS_DIR [RUNS_DIR ...] --out results/mem

writes  results/mem_runs.csv  (one row per TOFU_EVAL.json)
        results/mem_groups.csv (one row per seed group)

By default the seed is taken from a run-directory name ending in ``_seed-N``
(open-unlearning / baseline layout ``METHOD/split/model/<hparams>_seed-N/``).
Use ``--seed-regex`` for other layouts; the matched part is removed to form
the group key.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_metric import load_eval, summarize  # noqa: E402

FIELDS = ["Util", "Mem", "Avg", "Gibb", "ES", "EM", "ParaProb", "TR"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="Directories (or TOFU_EVAL.json files) to scan.")
    ap.add_argument("--out", required=True, help="Output prefix, e.g. results/mem")
    ap.add_argument("--seed-regex", default=r"_seed-(\d+)$",
                    help="Regex on the run directory name; group 1 is the seed (default: %(default)s)")
    ap.add_argument("--path-regex", default=None,
                    help="Regex on the full TOFU_EVAL.json path with a named group 'seed'; the other "
                         "groups, joined by '/', form the group key. Overrides --seed-regex. Example for "
                         r"run_all.sh: '(?P<run>[^/]+)_s(?P<seed>\d+)/logs/[^/]+/(?P<coeff>[^/]+)/TOFU_EVAL'")
    ap.add_argument("--min-gibb", type=float, default=0.8,
                    help="Flag groups whose mean Gibb is below this (default: %(default)s)")
    args = ap.parse_args()

    seed_re = re.compile(args.seed_regex)
    path_re = re.compile(args.path_regex) if args.path_regex else None
    files: list[Path] = []
    for r in args.roots:
        p = Path(r)
        files += [p] if p.is_file() else sorted(p.rglob("TOFU_EVAL.json"))
    if not files:
        sys.exit("no TOFU_EVAL.json found")

    runs = []
    for f in files:
        run_dir = f.parent
        if path_re is not None:
            pm = path_re.search(str(f))
            if not pm:
                print(f"WARNING: --path-regex does not match, skipped: {f}", file=sys.stderr)
                continue
            seed = pm.group("seed")
            group = "/".join(v for k, v in pm.groupdict().items() if k != "seed" and v is not None)
        else:
            m = seed_re.search(run_dir.name)
            seed = m.group(1) if m else ""
            group = str(run_dir.parent / seed_re.sub("", run_dir.name)) if m else str(run_dir)
        s = summarize(load_eval(f))
        if s["Mem"] is None:
            print(f"WARNING: a Mem component is missing, Mem not computable: {f}", file=sys.stderr)
        runs.append({"path": str(f), "group": group, "seed": seed, **s})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "group", "seed", *FIELDS])
        w.writeheader()
        w.writerows(runs)

    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    cols = ["group", "n_seeds", "seeds"]
    for k in FIELDS:
        cols += [f"{k}_mean", f"{k}_se"]
    cols.append(f"gibb_ok_{args.min_gibb}")
    rows = []
    for g, rs_all in sorted(groups.items()):
        # only complete runs (Util and Mem computable) enter the seed statistics
        rs = [r for r in rs_all if r["Util"] is not None and r["Mem"] is not None]
        skipped = sorted(r["seed"] for r in rs_all if r not in rs)
        if skipped:
            print(f"WARNING: {g}: incomplete run(s) excluded, seed(s) {' '.join(skipped)}", file=sys.stderr)
        if not rs:
            continue
        row = {"group": g, "n_seeds": len(rs), "seeds": " ".join(sorted(r["seed"] for r in rs))}
        for k in FIELDS:
            vals = [r[k] for r in rs if r[k] is not None]
            row[f"{k}_mean"] = mean(vals) if vals else None
            row[f"{k}_se"] = stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        gm = row["Gibb_mean"]
        row[f"gibb_ok_{args.min_gibb}"] = None if gm is None else gm >= args.min_gibb
        rows.append(row)
    with open(f"{out}_groups.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(runs)} runs, {len(rows)} groups -> {out}_runs.csv, {out}_groups.csv")


if __name__ == "__main__":
    main()
