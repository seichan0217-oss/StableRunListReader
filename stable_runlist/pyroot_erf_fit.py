#!/usr/bin/env python3
"""Error-function fits using PyROOT with selectable Poisson or chi-square statistics."""

import math
import numpy as np
import ROOT


ROOT.gROOT.SetBatch(True)


def _edge_expected(x_s, edge, baseline, amplitude, center_s, width_us):
    background = baseline
    argument = (x_s - center_s) * 1.0e6 / (math.sqrt(2.0) * width_us)
    turn = 0.5 * (1.0 + edge["direction"] * math.erf(argument))
    return max(background + amplitude * turn, 1.0e-12)


def _prepare_edges(x, y, edge_specs):
    all_x = np.asarray(x, dtype=float)
    all_y = np.asarray(y, dtype=float)
    if all_x.ndim != 1 or all_y.ndim != 1 or all_x.shape != all_y.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if not edge_specs:
        raise ValueError("at least one edge range is required")

    prepared = []
    for index, source in enumerate(edge_specs, start=1):
        edge = {
            "label": str(source.get("label", f"edge_{index}")),
            "center_guess": float(source["center_guess"]),
            "direction": int(source["direction"]),
            "fit_left": float(source["fit_left"]),
            "fit_right": float(source["fit_right"]),
        }
        if edge["direction"] not in (-1, 1):
            raise ValueError(f"edge {index}: direction must be +1 or -1")
        if (
            not np.isfinite(edge["center_guess"])
            or not np.isfinite(edge["fit_left"])
            or not np.isfinite(edge["fit_right"])
            or edge["fit_left"] >= edge["fit_right"]
        ):
            raise ValueError(f"edge {index}: invalid center or fit range")
        selected = (
            np.isfinite(all_x)
            & np.isfinite(all_y)
            & (all_y >= 0.0)
            & (all_x >= edge["fit_left"])
            & (all_x <= edge["fit_right"])
        )
        edge["indices"] = np.flatnonzero(selected)
        edge["fit_x"] = all_x[selected]
        edge["fit_y"] = all_y[selected]
        if len(edge["fit_x"]) < 6:
            raise ValueError(
                f"edge {index}: likelihood fit needs at least 6 bins; "
                f"selected {len(edge['fit_x'])}"
            )
        edge["bin_width_us"] = float(
            np.median(np.diff(edge["fit_x"])) * 1.0e6
        )
        edge["range_width_us"] = (
            edge["fit_right"] - edge["fit_left"]
        ) * 1.0e6

        side_bins = max(1, min(4, len(edge["fit_x"]) // 4))
        left_level = float(np.median(edge["fit_y"][:side_bins]))
        right_level = float(np.median(edge["fit_y"][-side_bins:]))
        if edge["direction"] == 1:
            baseline = left_level
            amplitude = right_level - left_level
        else:
            baseline = right_level
            amplitude = left_level - right_level
        edge["baseline_guess"] = max(baseline, 1.0e-6)
        edge["amplitude_guess"] = max(
            amplitude,
            float(np.ptp(edge["fit_y"])),
            1.0,
        )
        prepared.append(edge)
    return all_x, all_y, prepared


class _BinnedFitObjective:
    def __init__(self, edges, statistic):
        self.edges = edges
        self.statistic = statistic

    def __call__(self, parameters):
        width_us = parameters[0]
        objective = 0.0
        for edge_index, edge in enumerate(self.edges):
            offset = 1 + 3 * edge_index
            baseline = parameters[offset]
            amplitude = parameters[offset + 1]
            center_s = edge["center_guess"] + parameters[offset + 2] * 1.0e-6
            for x_s, observed in zip(edge["fit_x"], edge["fit_y"]):
                expected = _edge_expected(
                    float(x_s),
                    edge,
                    baseline,
                    amplitude,
                    center_s,
                    width_us,
                )
                if self.statistic == "poisson":
                    objective += expected
                    if observed > 0.0:
                        objective -= observed * math.log(expected)
                else:
                    objective += (observed - expected) ** 2 / expected
        return objective


def fit_shared_erf_likelihood(x, y, edge_specs, statistic="poisson"):
    """Fit edges with a shared width using Poisson likelihood or Pearson chi2."""
    statistic = str(statistic).lower()
    if statistic not in ("poisson", "chi2"):
        raise ValueError("statistic must be poisson or chi2")
    all_x, all_y, edges = _prepare_edges(x, y, edge_specs)
    parameter_count = 1 + 3 * len(edges)
    minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Migrad")
    if not minimizer:
        raise RuntimeError("ROOT could not create the Minuit2 Migrad minimizer")
    minimizer.SetMaxFunctionCalls(200_000)
    minimizer.SetMaxIterations(100_000)
    minimizer.SetTolerance(0.1)
    minimizer.SetStrategy(1)
    minimizer.SetPrintLevel(0)
    minimizer.SetErrorDef(0.5 if statistic == "poisson" else 1.0)

    minimum_bin_width = min(edge["bin_width_us"] for edge in edges)
    maximum_width = min(edge["range_width_us"] for edge in edges)
    initial_width = max(
        minimum_bin_width,
        float(np.median([edge["range_width_us"] for edge in edges])) / 6.0,
    )
    initial_width = min(initial_width, 0.9 * maximum_width)
    minimizer.SetLimitedVariable(
        0,
        "shared_width_us",
        initial_width,
        max(0.05 * initial_width, 0.01),
        max(minimum_bin_width / 100.0, 1.0e-4),
        maximum_width,
    )

    parameter_names = ["shared_width_us"]
    for edge_index, edge in enumerate(edges):
        offset = 1 + 3 * edge_index
        maximum_count = max(float(np.max(edge["fit_y"])), 1.0)
        normalization_limit = 100.0 * maximum_count + 1.0
        center_low_us = (
            float(edge["fit_x"][0]) - edge["center_guess"]
        ) * 1.0e6
        center_high_us = (
            float(edge["fit_x"][-1]) - edge["center_guess"]
        ) * 1.0e6
        prefix = f"edge{edge_index + 1}"
        parameter_names.extend(
            (
                f"{prefix}_baseline",
                f"{prefix}_amplitude",
                f"{prefix}_center_offset_us",
            )
        )
        minimizer.SetLimitedVariable(
            offset,
            parameter_names[offset],
            edge["baseline_guess"],
            max(math.sqrt(edge["baseline_guess"]), 0.1),
            0.0,
            normalization_limit,
        )
        minimizer.SetLimitedVariable(
            offset + 1,
            parameter_names[offset + 1],
            edge["amplitude_guess"],
            max(math.sqrt(edge["amplitude_guess"]), 0.1),
            0.0,
            normalization_limit,
        )
        minimizer.SetLimitedVariable(
            offset + 2,
            parameter_names[offset + 2],
            0.0,
            max(edge["bin_width_us"] / 5.0, 0.01),
            center_low_us,
            center_high_us,
        )

    objective = _BinnedFitObjective(edges, statistic)
    # Keep this Functor alive until Minimize and Hesse have both completed.
    functor = ROOT.Math.Functor(objective, parameter_count)
    minimizer.SetFunction(functor)
    success = bool(minimizer.Minimize())
    hesse_success = bool(minimizer.Hesse())
    status = int(minimizer.Status())
    covariance_status = int(minimizer.CovMatrixStatus())
    if status not in (0, 3):
        raise RuntimeError(
            f"PyROOT Minuit2 {statistic} fit failed: "
            f"success={success}, status={status}, edm={minimizer.Edm():.6g}"
        )

    values = np.asarray(
        [float(minimizer.X()[i]) for i in range(parameter_count)]
    )
    errors = np.asarray(
        [float(minimizer.Errors()[i]) for i in range(parameter_count)]
    )
    covariance = np.asarray(
        [
            [float(minimizer.CovMatrix(i, j)) for j in range(parameter_count)]
            for i in range(parameter_count)
        ]
    )

    deviance = 0.0
    pearson_chi2 = 0.0
    fit_results = []
    for edge_index, edge in enumerate(edges):
        offset = 1 + 3 * edge_index
        baseline = values[offset]
        amplitude = values[offset + 1]
        center_s = edge["center_guess"] + values[offset + 2] * 1.0e-6
        fitted_y = []
        for x_s, observed in zip(edge["fit_x"], edge["fit_y"]):
            expected = _edge_expected(
                float(x_s),
                edge,
                baseline,
                amplitude,
                center_s,
                values[0],
            )
            fitted_y.append(expected)
            pearson_chi2 += (observed - expected) ** 2 / expected
            if observed > 0.0:
                deviance += 2.0 * (
                    expected
                    - observed
                    + observed * math.log(observed / expected)
                )
            else:
                deviance += 2.0 * expected
        fit_results.append(
            {
                "label": edge["label"],
                "direction": edge["direction"],
                "position_s": float(center_s),
                "position_error_s": float(errors[offset + 2] * 1.0e-6),
                "transition_width_s": float(values[0] * 1.0e-6),
                "transition_width_error_s": float(errors[0] * 1.0e-6),
                "baseline": float(baseline),
                "baseline_error": float(errors[offset]),
                "amplitude": float(amplitude),
                "amplitude_error": float(errors[offset + 1]),
                "fit_start_s": float(edge["fit_x"][0]),
                "fit_stop_s": float(edge["fit_x"][-1]),
                "fit_bins": int(len(edge["fit_x"])),
                "fit_x": edge["fit_x"],
                "fit_y": np.asarray(fitted_y),
            }
        )

    ndf = int(sum(len(edge["fit_x"]) for edge in edges) - parameter_count)
    fit_statistic = deviance if statistic == "poisson" else pearson_chi2
    summary = {
        "backend": "PyROOT Minuit2",
        "statistic": statistic,
        "statistic_label": "deviance" if statistic == "poisson" else "chi2",
        "method": (
            "binned Poisson" if statistic == "poisson" else "Pearson chi-square"
        ),
        "transition_width_s": float(values[0] * 1.0e-6),
        "transition_width_error_s": float(errors[0] * 1.0e-6),
        "minimum_objective": float(minimizer.MinValue()),
        "deviance": float(deviance),
        "chi2": float(pearson_chi2),
        "ndf": ndf,
        "deviance_ndf": float(deviance / ndf) if ndf > 0 else float("nan"),
        "chi2_ndf": float(pearson_chi2 / ndf) if ndf > 0 else float("nan"),
        "fit_statistic": float(fit_statistic),
        "fit_statistic_ndf": (
            float(fit_statistic / ndf) if ndf > 0 else float("nan")
        ),
        "fit_success": success,
        "hesse_success": hesse_success,
        "status": status,
        "covariance_status": covariance_status,
        "edm": float(minimizer.Edm()),
        "parameter_names": parameter_names,
        "parameter_values": values,
        "parameter_errors": errors,
        "covariance": covariance,
    }
    return all_x, all_y, edges, fit_results, summary


def fit_erf_edges_independent_likelihood(
    x, y, edge_specs, statistic="poisson"
):
    """Fit every edge independently with its own transition width."""
    if not edge_specs:
        raise ValueError("at least one edge range is required")

    prepared_edges = []
    fit_results = []
    fit_summaries = []
    all_x = None
    all_y = None
    for spec in edge_specs:
        edge_x, edge_y, edges, fits, summary = fit_shared_erf_likelihood(
            x, y, [spec], statistic=statistic
        )
        all_x = edge_x
        all_y = edge_y
        edge = edges[0]
        prepared_edges.append(edge)
        fit = dict(fits[0])
        boundary_tolerance_s = 0.51 * edge["bin_width_us"] * 1.0e-6
        fit.update(
            {
                "deviance": summary["deviance"],
                "ndf": summary["ndf"],
                "deviance_ndf": summary["deviance_ndf"],
                "chi2": summary["chi2"],
                "chi2_ndf": summary["chi2_ndf"],
                "fit_statistic": summary["fit_statistic"],
                "fit_statistic_ndf": summary["fit_statistic_ndf"],
                "statistic": summary["statistic"],
                "statistic_label": summary["statistic_label"],
                "minuit_status": summary["status"],
                "covariance_status": summary["covariance_status"],
                "edm": summary["edm"],
                "fit_converged": (
                    summary["fit_success"] and summary["status"] == 0
                ),
                "center_at_boundary": (
                    fit["position_s"] - fit["fit_start_s"]
                    <= boundary_tolerance_s
                    or fit["fit_stop_s"] - fit["position_s"]
                    <= boundary_tolerance_s
                ),
            }
        )
        fit_results.append(fit)
        fit_summaries.append(summary)

    total_deviance = float(sum(item["deviance"] for item in fit_summaries))
    total_chi2 = float(sum(item["chi2"] for item in fit_summaries))
    total_fit_statistic = float(
        sum(item["fit_statistic"] for item in fit_summaries)
    )
    total_ndf = int(sum(item["ndf"] for item in fit_summaries))
    combined_summary = {
        "backend": "PyROOT Minuit2",
        "statistic": statistic,
        "statistic_label": "deviance" if statistic == "poisson" else "chi2",
        "method": (
            "binned Poisson" if statistic == "poisson" else "Pearson chi-square"
        ),
        "independent_edges": True,
        "deviance": total_deviance,
        "chi2": total_chi2,
        "fit_statistic": total_fit_statistic,
        "ndf": total_ndf,
        "deviance_ndf": (
            total_deviance / total_ndf if total_ndf > 0 else float("nan")
        ),
        "chi2_ndf": total_chi2 / total_ndf if total_ndf > 0 else float("nan"),
        "fit_statistic_ndf": (
            total_fit_statistic / total_ndf
            if total_ndf > 0
            else float("nan")
        ),
    }
    return all_x, all_y, prepared_edges, fit_results, combined_summary


def draw_erf_fits_root(
    x,
    y,
    edges,
    fit_results,
    summary,
    output,
    title_prefix="Deadtime",
):
    """Draw the fitted dead-time edges with ROOT and save a PNG."""
    columns = 2
    rows = (len(edges) + columns - 1) // columns
    canvas = ROOT.TCanvas(
        "deadtime_erf_fit_canvas",
        "Independent error-function fits",
        1800,
        600 * rows,
    )
    canvas.Divide(columns, rows)
    keep_alive = []

    for index, (edge, fit) in enumerate(zip(edges, fit_results), start=1):
        pad = canvas.cd(index)
        pad.SetGrid(1, 1)
        padding = max(
            2.0 * edge["bin_width_us"] * 1.0e-6,
            0.25 * (edge["fit_right"] - edge["fit_left"]),
        )
        visible = (
            (x >= edge["fit_left"] - padding)
            & (x <= edge["fit_right"] + padding)
        )
        visible_x_ms = x[visible] * 1000.0
        visible_y = y[visible]
        x_min = float(np.min(visible_x_ms))
        x_max = float(np.max(visible_x_ms))
        y_max = max(float(np.max(visible_y)) * 1.25, 1.0)
        direction = "rising" if edge["direction"] == 1 else "falling"
        frame = pad.DrawFrame(
            x_min,
            0.0,
            x_max,
            y_max,
            (
                f"{title_prefix} {index} {direction};"
                "Time difference [ms];Count (linear scale)"
            ),
        )
        frame.GetYaxis().SetTitleOffset(1.25)


        points = ROOT.TGraph(int(np.count_nonzero(visible)))
        points.SetName(f"deadtime_edge_{index}_data")
        for point_index, (point_x, point_y) in enumerate(
            zip(visible_x_ms, visible_y)
        ):
            points.SetPoint(point_index, float(point_x), float(point_y))
        points.SetMarkerStyle(20)
        points.SetMarkerSize(0.75)
        points.SetMarkerColor(ROOT.kBlue + 1)
        points.Draw("P SAME")

        dense_x = np.linspace(edge["fit_left"], edge["fit_right"], 500)
        curve = ROOT.TGraph(len(dense_x))
        curve.SetName(f"deadtime_edge_{index}_fit")
        for point_index, point_x in enumerate(dense_x):
            expected = _edge_expected(
                float(point_x),
                edge,
                fit["baseline"],
                fit["amplitude"],
                fit["position_s"],
                fit["transition_width_s"] * 1.0e6,
            )
            curve.SetPoint(point_index, float(point_x * 1000.0), expected)
        curve.SetLineColor(ROOT.kRed + 1)
        curve.SetLineWidth(3)
        curve.Draw("L SAME")

        center_line = ROOT.TLine(
            fit["position_s"] * 1000.0,
            0.0,
            fit["position_s"] * 1000.0,
            y_max,
        )
        center_line.SetLineColor(ROOT.kRed + 1)
        center_line.SetLineStyle(2)
        center_line.Draw("same")

        midpoint_ms = 0.5 * (edge["fit_left"] + edge["fit_right"]) * 1000.0
        midpoint_line = ROOT.TLine(midpoint_ms, 0.0, midpoint_ms, y_max)
        midpoint_line.SetLineColor(ROOT.kGray + 2)
        midpoint_line.SetLineStyle(3)
        midpoint_line.Draw("same")

        legend = ROOT.TLegend(0.12, 0.57, 0.48, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.032)
        goodness_label = (
            "D" if fit["statistic"] == "poisson" else "#chi^{2}"
        )
        legend.AddEntry(
            curve,
            f"{goodness_label}/ndf = {fit['fit_statistic_ndf']:.2f}",
            "l",
        )
        legend.AddEntry(
            curve,
            (
                f"#sigma = {fit['transition_width_s'] * 1.0e3:.3f} "
                f"#pm {fit['transition_width_error_s'] * 1.0e3:.3f} ms"
            ),
            "",
        )
        legend.AddEntry(
            center_line,
            (
                f"#mu = {fit['position_s'] * 1000.0:.6f} "
                f"#pm {fit['position_error_s'] * 1000.0:.6f} ms"
            ),
            "",
        )
        legend.AddEntry(
            curve,
            f"A = {fit['amplitude']:.1f} #pm {fit['amplitude_error']:.1f}",
            "",
        )
        legend.AddEntry(
            curve,
            f"B = {fit['baseline']:.1f} #pm {fit['baseline_error']:.1f}",
            "",
        )
        legend.Draw()
        keep_alive.extend(
            (
                frame,
                points,
                curve,
                center_line,
                midpoint_line,
                legend,
            )
        )

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output))
    canvas.Close()
