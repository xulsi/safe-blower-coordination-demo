"""Run grid search and Safe Bayesian Optimization on every demand scenario."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List

from blower_model import BlowerModel
from network_model import NetworkModel
from optimizer import GridSearchOptimizer, SBOOptimizer
from simulation import SimulationRunner
from utils import load_config

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("optimizer").setLevel(logging.WARNING)


def _build_runner(config: Dict[str, Any], method: str) -> SimulationRunner:
    blower = BlowerModel(config.get("blower_model", {}))
    network = NetworkModel(config.get("network_model", {}))
    penalty = config.get("optimizer", {}).get("penalty", {"lambda_sw": 0.0, "lambda_ramp": 0.0})
    if method == "sbo":
        sbo_params = config.get("optimizer", {}).get("sbo", {})
        optimizer = SBOOptimizer(int(sbo_params.get("init_samples", 5)), sbo_params, penalty)
    else:
        speed_grid = config.get("optimizer", {}).get("grid_search", {}).get(
            "speed_grid", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        )
        optimizer = GridSearchOptimizer(speed_grid, penalty)
    return SimulationRunner(blower, network, optimizer, config)


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}f}"


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    config["num_blowers"] = int(config.get("num_blowers", 3))
    scenarios: List[Dict[str, Any]] = config.get("simulation", {}).get("demand_scenarios", [])

    print("Safe blower coordination — grid search vs Safe Bayesian Optimization")
    print(f"{'Scenario':<36} {'Method':<22} {'Saving %':>10} {'Runtime s':>10} {'Evals':>8} {'Power kW':>10}")
    print("-" * 100)

    rows = []
    for scenario in scenarios:
        name = str(scenario.get("scenario"))
        demand = float(scenario.get("demand"))
        for method in ("grid", "sbo"):
            runner = _build_runner(config, method)
            result = runner.run_simulation(demand)
            label = "Grid search (GS)" if method == "grid" else "Safe Bayesian Opt. (SBO)"
            print(
                f"{name:<36} {label:<22} {_fmt(result.get('energy_savings_percent')):>10} "
                f"{_fmt(result.get('runtime_seconds'), 2):>10} {int(result.get('evaluation_count', 0)):>8} "
                f"{_fmt(result.get('total_power')):>10}"
            )
            rows.append(
                {
                    "scenario": name,
                    "demand": demand,
                    "method": method,
                    "energy_savings_percent": result.get("energy_savings_percent"),
                    "runtime_seconds": result.get("runtime_seconds"),
                    "evaluation_count": result.get("evaluation_count"),
                    "total_power": result.get("total_power"),
                    "header_pressure": result.get("header_pressure"),
                    "optimal_configuration": result.get("optimal_configuration"),
                }
            )

    out_path = root / "comparison_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("-" * 100)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
