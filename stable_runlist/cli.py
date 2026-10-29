"""One command interface; load optional ROOT/COOL dependencies on demand."""

import argparse
import importlib
import sys

COMMANDS = {
    "fetch-runs": ("get_ahcal_run_times", "main", "Download AHCAL run metadata"),
    "runlist": ("stable_runlist_reader", "main", "Build stable/good intervals (COOL)"),
    "raw-counts": ("raw_trigger_rate", "counts_main", "Generate missing raw count CSVs"),
    "luminosity": ("add_stable_luminosity", "main", "Integrate stable luminosity and D1-D3 effective luminosity"),
    "prepost": ("prepost", "main", "Inspect one run's PreVeto/PostVeto counters"),
    "collect-prepost": ("prepost", "collect_main", "Collect counter ratios for selected runs"),
    "raw-rate": ("raw_trigger_rate", "main", "Inspect raw recorded trigger rates"),
    "deadtime": ("deadtime_and_trigger_rate", "main", "Fit deadtime edges and plot rates (ROOT)"),
    "summary": ("rate_summary", "main", "Collect fits and build a shared deadtime model"),
    "refresh-summary": ("rate_summary", "refresh_main", "Recalculate an existing rate summary"),
    "rebuild-raw-rate": ("legacy_rates", "rebuild_main", "Rebuild the legacy corrected rate CSV"),
    "refresh-rate-bins": ("legacy_rates", "refresh_main", "Refresh legacy prescale-corrected bins"),
    "plot-tlu": ("timestamp_diagnostics", "main", "Plot TLU timestamp differences"),
    "plot-cpu": ("timestamp_diagnostics", "cpu_main", "Plot CPU timestamp differences"),
    "fit-rate-lumi": ("fit_luminosity_true_rate_pyroot", "main", "Fit rate versus luminosity (ROOT)"),
    "toy-mc": ("toy_deadtime_live_mc", "main", "Simulate deadtime/live-window variations (ROOT)"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="stable-runlist",
        description="AHCAL stable run lists, trigger rates, and luminosity.",
        epilog="Commands:\n" + "\n".join(
            f"  {name:20s} {description}" for name, (_, _, description) in COMMANDS.items()
        ) + "\n\nUse stable-runlist COMMAND --help for command options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS, help="workflow to run")
    args = parser.parse_args(argv[:1])
    if args.command is None:
        parser.print_help()
        return 0
    module_name, entry_point, _ = COMMANDS[args.command]
    module = importlib.import_module("." + module_name, __package__)
    previous_argv = sys.argv
    try:
        sys.argv = [f"{parser.prog} {args.command}", *argv[1:]]
        return getattr(module, entry_point)()
    finally:
        sys.argv = previous_argv
