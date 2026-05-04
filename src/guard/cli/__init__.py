"""CLI dispatcher for the `guard` command."""


import argparse
import sys

from dotenv import load_dotenv

load_dotenv(override=True)  # .env values override shell environment


def main() -> None:
    """Main entry point for the `guard` CLI.

    Dispatches to sub-commands:

    * ``guard generate <config.yaml>``  — compute and save steering vectors
    * ``guard eval <sv.pt>``            — quick coefficient-sweep eval
    """
    parser = argparse.ArgumentParser(
        prog="guard",
        description="GUARD — Gated Unlearning via Activation Redirection",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # guard generate
    gen_parser = subparsers.add_parser("generate", help="Compute steering vectors from a YAML config.")
    gen_parser.add_argument("config", help="Path to generate.yaml")
    gen_parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    gen_parser.add_argument("--dry-run", action="store_true")
    gen_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # guard eval
    eval_parser = subparsers.add_parser("eval", help="Quick coefficient-sweep perplexity eval.")
    eval_parser.add_argument("sv_path", help="Path to sv.pt")
    eval_parser.add_argument("--model", default="")
    eval_parser.add_argument("--coeffs", nargs="+", type=float, default=[-0.8, -0.3, -0.15, -0.05, 0.0])
    eval_parser.add_argument("--forget-docs", default=None)
    eval_parser.add_argument("--retain-docs", default=None)
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--max-length", type=int, default=512)
    eval_parser.add_argument("--cuda", default=None)
    eval_parser.add_argument("--output", default=None)
    eval_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # guard cluster
    clust_parser = subparsers.add_parser("cluster", help="Cluster forget docs and compute per-cluster SVs.")
    clust_parser.add_argument("config", help="Path to generate.yaml")
    clust_parser.add_argument("--n-clusters", default=None, metavar="K",
                              help="Override n_clusters from config ('auto' or int).")
    clust_parser.add_argument("--k-min", type=int, default=None, metavar="K_MIN")
    clust_parser.add_argument("--k-max", type=int, default=None, metavar="K_MAX")
    clust_parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    clust_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # guard benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run a benchmark (tofu | muse).")
    bench_parser.add_argument("suite", choices=["tofu", "muse"], help="Benchmark suite.")
    bench_parser.add_argument("config", help="Path to benchmark YAML config.")
    bench_parser.add_argument(
        "--sv-dir", default=None, metavar="PATH",
        help="Path to clustered SV layer dir (overrides config variants).",
    )
    bench_parser.add_argument("--retain-logs", default=None)
    bench_parser.add_argument("--output-path", default=None, metavar="PATH",
                              help="Override output_path from the benchmark config.")
    bench_parser.add_argument("--batch-size", type=int, default=None)
    bench_parser.add_argument("--cuda", default=None)
    bench_parser.add_argument(
        "--coeffs", default=None, metavar="C1,C2,...",
        help="Comma-separated coefficients to evaluate, overriding config.",
    )
    bench_parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override a flat config key, e.g. --override routing_threshold=0.3",
    )

    # guard plot
    plot_parser = subparsers.add_parser("plot", help="Plot clusters, results, or similarity distributions.")
    plot_sub = plot_parser.add_subparsers(dest="plot_cmd", metavar="WHAT")
    plot_sub.required = True
    pc = plot_sub.add_parser("clusters", help="Plot text-space cluster projection.")
    pc.add_argument("cluster_dir", help="Behavior-level cluster directory.")
    pc.add_argument("--out", default=None)
    pr = plot_sub.add_parser("results", help="Plot eval metrics vs coefficient.")
    pr.add_argument("results_json", help="JSON output from guard benchmark.")
    pr.add_argument("--out", default=None)
    ps = plot_sub.add_parser(
        "similarity",
        help="Plot cosine-similarity distributions to choose a routing threshold.",
    )
    ps.add_argument("config", nargs="?", default=None,
                    help="Path to a guard generate/cluster YAML config.")
    ps.add_argument("--hf-dataset", default=None, metavar="HF_ID",
                    help="HuggingFace dataset ID (e.g. 'locuslab/TOFU').")
    ps.add_argument("--hf-forget-split", default=None, metavar="SPLIT")
    ps.add_argument("--hf-retain-split", default=None, metavar="SPLIT")
    ps.add_argument("--hf-text-key", default="text", metavar="KEY")
    ps.add_argument("--forget-split", default=None, metavar="SPLIT")
    ps.add_argument("--retain-split", default=None, metavar="SPLIT")
    ps.add_argument("--threshold", type=float, default=0.5)
    ps.add_argument("--n-clusters", type=int, default=10, metavar="K")
    ps.add_argument("--out", default="experiments/plots/similarity.png", metavar="PATH")
    ps.add_argument("--fineweb-n", type=int, default=5000, metavar="N")
    ps.add_argument("--st-model", default="all-MiniLM-L6-v2", metavar="MODEL")
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.command == "generate":
        from guard.cli.generate import main as gen_main
        gen_main(sys.argv[2:])
    elif args.command == "eval":
        from guard.cli.eval import main as eval_main
        eval_main(sys.argv[2:])
    elif args.command == "cluster":
        from guard.cli.cluster import main as clust_main
        clust_main(sys.argv[2:])
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "plot":
        _run_plot(args)


def _run_benchmark(args: argparse.Namespace) -> None:
    import importlib.util as ilu
    from pathlib import Path

    _HERE = Path(__file__).resolve().parent  # src/guard/cli/
    guard_root = _HERE.parents[2]            # guard/
    benchmarks_dir = guard_root / "benchmarks"

    script = benchmarks_dir / f"{args.suite}.py"
    spec = ilu.spec_from_file_location(f"benchmarks.{args.suite}", script)
    mod = ilu.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)      # type: ignore[union-attr]

    # Reconstruct argv for the benchmark's own argparse
    bench_argv = [args.config]
    if getattr(args, "sv_dir", None):
        bench_argv += ["--sv-dir", args.sv_dir]
    if getattr(args, "retain_logs", None):
        bench_argv += ["--retain-logs", args.retain_logs]
    if getattr(args, "batch_size", None):
        bench_argv += ["--batch-size", str(args.batch_size)]
    if getattr(args, "cuda", None):
        bench_argv += ["--cuda", args.cuda]
    if getattr(args, "output_path", None):
        bench_argv += ["--output-path", args.output_path]
    if getattr(args, "coeffs", None):
        bench_argv += [f"--coeffs={args.coeffs}"]
    for ov in getattr(args, "override", []):
        bench_argv += ["--override", ov]

    sys.argv = [str(script)] + bench_argv
    mod.main()


def _run_plot(args: argparse.Namespace) -> None:
    import importlib.util as ilu
    from pathlib import Path

    if args.plot_cmd == "similarity":
        from guard.cli.plot_similarity import main as sim_main
        # Reconstruct argv for the similarity subcommand's own argparse
        argv: list[str] = []
        if args.config:
            argv.append(args.config)
        if args.hf_dataset:
            argv += ["--hf-dataset", args.hf_dataset]
        if args.hf_forget_split:
            argv += ["--hf-forget-split", args.hf_forget_split]
        if args.hf_retain_split:
            argv += ["--hf-retain-split", args.hf_retain_split]
        if args.hf_text_key != "text":
            argv += ["--hf-text-key", args.hf_text_key]
        if args.forget_split:
            argv += ["--forget-split", args.forget_split]
        if args.retain_split:
            argv += ["--retain-split", args.retain_split]
        argv += ["--threshold", str(args.threshold)]
        argv += ["--n-clusters", str(args.n_clusters)]
        argv += ["--out", args.out]
        argv += ["--fineweb-n", str(args.fineweb_n)]
        argv += ["--st-model", args.st_model]
        argv += ["--seed", str(args.seed)]
        argv += ["--log-level", args.log_level]
        sim_main(argv)
        return

    _HERE = Path(__file__).resolve().parent
    guard_root = _HERE.parents[2]
    scripts_dir = guard_root / "scripts"

    if args.plot_cmd == "clusters":
        script = scripts_dir / "plot_clusters.py"
        script_argv = [args.cluster_dir]
        if args.out:
            script_argv += ["--out", args.out]
    else:
        script = scripts_dir / "plot_results.py"
        script_argv = [args.results_json]
        if args.out:
            script_argv += ["--out", args.out]

    spec = ilu.spec_from_file_location("scripts.plot", script)
    mod = ilu.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)      # type: ignore[union-attr]
    sys.argv = [str(script)] + script_argv
    mod.main()


def _version() -> str:
    from guard._version import __version__
    return __version__
