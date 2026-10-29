#!/usr/bin/env python3
"""Fit true trigger rate versus luminosity separately with PyROOT."""

from .config import OUTPUT_DIR

import argparse
import csv
import json
import math
from array import array
from pathlib import Path

import ROOT


HERE = OUTPUT_DIR
DEFAULT_INPUT = (
    HERE
    / "deadtime_trigger_rate"
    / "summary_anyveto_no_prescale_stat"
    / "deadtime_trigger_rate_summary.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent

CONFIGURATIONS = {
    "nonveto": "AHCALPhysics_NoVeto",
    "prescale": "AHCALPhysics_Veto_PreScale",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--nonveto-x-range", nargs=2, type=float, metavar=("MIN", "MAX"),
        default=(0, 20000.0),
    )
    parser.add_argument(
        "--prescale-x-range", nargs=2, type=float, metavar=("MIN", "MAX"),
        default=(0, 30000.0),
    )
    parser.add_argument(
        "--nonveto-y-range", nargs=2, type=float, metavar=("MIN", "MAX"),
        default=(0, 4000.0),

    )
    parser.add_argument(
        "--prescale-y-range", nargs=2, type=float, metavar=("MIN", "MAX"),
        default=(0.0, 1600.0),
    )
    args = parser.parse_args()
    for name in (
        "nonveto_x_range", "prescale_x_range",
        "nonveto_y_range", "prescale_y_range",
    ):
        values = getattr(args, name)
        if values is None:
            continue
        low, high = values
        if not all(math.isfinite(value) for value in values) or high <= low:
            parser.error(
                f"--{name.replace('_', '-')} requires finite MIN < MAX"
            )
    return args


def finite_value(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_points(path, configuration):
    points = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("configuration") != configuration:
                continue
            luminosity = finite_value(row, "stable_luminosity_mean_ub_inv_s")
            rate = finite_value(row, "stable_trigger_rate_true_hz")
            if luminosity is None or rate is None:
                continue
            points.append((luminosity, rate))
    points.sort()
    if len(points) < 3:
        raise ValueError(
            f"need at least three points for {configuration}, got {len(points)}"
        )
    return points


def padded_range(values, fraction=0.08):
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        span = max(abs(low), 1.0)
    return low - fraction * span, high + fraction * span


def make_plot(
    points, key, configuration, output_dir, x_display_range, y_display_range
):
    display_configuration = configuration.replace("_", " ")
    x = array("d", (point[0] for point in points))
    y = array("d", (point[1] for point in points))

    # No measurement errors are assigned.  ROOT therefore performs an
    # unweighted chi-square fit, effectively using sigma_i = 1 for every run.
    graph = ROOT.TGraph(len(points), x, y)
    graph.SetName(f"graph_{key}")
    graph.SetTitle("")
    graph.SetMarkerStyle(20 if key == "nonveto" else 21)
    graph.SetMarkerSize(1.15)
    graph.SetMarkerColor(ROOT.kBlue + 1 if key == "nonveto" else ROOT.kOrange + 7)
    graph.SetLineColor(graph.GetMarkerColor())

    x_fit_min, x_fit_max = min(x), max(x)
    fit = ROOT.TF1(f"linear_fit_{key}", "[0]+[1]*x", x_fit_min, x_fit_max)
    slope_seed = (y[-1] - y[0]) / (x[-1] - x[0])
    fit.SetParameters(y[0] - slope_seed * x[0], slope_seed)
    fit.SetParNames("intercept", "slope")
    fit.SetLineColor(ROOT.kRed + 1)
    fit.SetLineWidth(2)
    fit_result = graph.Fit(fit, "SQ")
    chi2 = fit.GetChisquare()
    fit_ndf = fit.GetNDF()
    # Keep all points in the fit, then extend the drawn line across the full
    # manually selected x-axis range for this configuration.
    fit.SetRange(*x_display_range)

    residuals = [
        yi - fit.Eval(xi)
        for xi, yi in zip(x, y)
    ]
    residual_sum_squares = sum(residual * residual for residual in residuals)
    ndf = len(points) - fit.GetNpar()
    residual_standard_error = (
        math.sqrt(residual_sum_squares / ndf) if ndf > 0 else None
    )
    y_mean = sum(y) / len(y)
    total_sum_squares = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else None
    )

    canvas = ROOT.TCanvas(f"canvas_{key}", "", 1000, 750)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.09)
    canvas.SetGrid()

    x_low, x_high = x_display_range
    y_low, y_high = (
        y_display_range if y_display_range is not None else padded_range(y)
    )
    graph.GetXaxis().SetLimits(x_low, x_high)
    graph.SetMinimum(y_low if y_display_range is not None else max(0.0, y_low))
    graph.SetMaximum(y_high)
    graph.GetXaxis().SetTitle(
        "Mean luminosity in stable intervals [#mub^{-1} s^{-1}]"
    )
    graph.GetYaxis().SetTitle("True AnyVetoed rate in stable intervals [Hz]")
    graph.GetXaxis().SetTitleSize(0.045)
    graph.GetYaxis().SetTitleSize(0.045)
    graph.GetXaxis().SetLabelSize(0.038)
    graph.GetYaxis().SetLabelSize(0.038)
    graph.GetYaxis().SetTitleOffset(1.35)
    graph.Draw("AP")
    fit.Draw("SAME")

    title = ROOT.TLatex()
    title.SetNDC(True)
    title.SetTextFont(42)
    title.SetTextSize(0.040)
    title.DrawLatex(
        0.13, 0.94, f"{display_configuration}: true rate vs luminosity"
    )

    box = ROOT.TPaveText(0.15, 0.7, 0.4, 0.9, "NDC")
    box.SetFillColor(ROOT.kWhite)
    box.SetFillStyle(1001)
    box.SetBorderSize(1)
    box.SetTextAlign(12)
    box.SetTextFont(42)
    box.SetTextSize(0.031)
    box.AddText("fit: p_{0} + p_{1} x")
    box.AddText(f"p_{{0}} = {fit.GetParameter(0):.5g} Hz")
    box.AddText(f"p_{{1}} = {fit.GetParameter(1):.5g}")
    box.AddText(f"#chi^{{2}} / ndf = {chi2:.2f} / {fit_ndf}")
    box.Draw()

    output = output_dir / f"luminosity_vs_true_trigger_rate_{key}_pyroot_fit.png"
    canvas.SaveAs(str(output))
    return output, {
        "configuration": configuration,
        "n_points": len(points),
        "fit_model": "p0 + p1 * luminosity",
        "fit_method": "unweighted chi-square with unit point errors",
        "point_errors": None,
        "effective_point_error_hz": 1.0,
        "fit_status": int(fit_result),
        "intercept_hz": fit.GetParameter(0),
        "slope_hz_per_ub_inv_s": fit.GetParameter(1),
        "residual_sum_squares_hz2": residual_sum_squares,
        "residual_standard_error_hz": residual_standard_error,
        "r_squared": r_squared,
        "chi2": chi2,
        "ndf": fit_ndf,
        "chi2_per_ndf": chi2 / fit_ndf if fit_ndf > 0 else None,
        "residual_ndf": ndf,
        "x_min": x_fit_min,
        "x_max": x_fit_max,
        "plot_x_min": x_low,
        "plot_x_max": x_high,
        "plot_y_min": (
            y_low if y_display_range is not None else max(0.0, y_low)
        ),
        "plot_y_max": y_high,
    }


def fit_configuration_groups(input_path, output_dir, *, skip_insufficient=False,
                             nonveto_x_range=(0, 20000), prescale_x_range=(0, 30000),
                             nonveto_y_range=(0, 4000), prescale_y_range=(0, 1600)):
    """Fit each configuration group without parsing command-line arguments."""
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetLineScalePS(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = {
        "nonveto": (nonveto_x_range, nonveto_y_range),
        "prescale": (prescale_x_range, prescale_y_range),
    }

    results = {}
    for key, configuration in CONFIGURATIONS.items():
        try:
            points = load_points(input_path, configuration)
            if points[0][0] == points[-1][0]:
                raise ValueError(f"{configuration}: luminosities must not all be equal")
        except ValueError as error:
            if not skip_insufficient:
                raise
            results[key] = {"status": "skipped", "reason": str(error)}
            print(f"Skipping luminosity fit: {error}")
            continue
        output, result = make_plot(
            points,
            key,
            configuration,
            output_dir,
            x_display_range=ranges[key][0],
            y_display_range=ranges[key][1],
        )
        results[key] = result
        print(f"Wrote {output}")

    result_path = output_dir / "luminosity_vs_true_trigger_rate_pyroot_fit.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {result_path}")
    return results


def main(args=None, skip_insufficient=False):
    args = arguments() if args is None else args
    fit_configuration_groups(
        args.input, args.output_dir, skip_insufficient=skip_insufficient,
        nonveto_x_range=args.nonveto_x_range, prescale_x_range=args.prescale_x_range,
        nonveto_y_range=args.nonveto_y_range, prescale_y_range=args.prescale_y_range,
    )


if __name__ == "__main__":
    main()
