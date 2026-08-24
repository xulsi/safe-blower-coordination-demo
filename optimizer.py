"""
optimizer.py

This module defines the optimization layer for coordinating multiple parallel blowers.
It provides an abstract Optimizer class and two concrete implementations:
    - GridSearchOptimizer: uses an exhaustive grid search over candidate on/off configurations
      and discrete speed values.
    - SBOOptimizer: uses Safe Bayesian Optimization (SBO) based on a Gaussian Process surrogate
      and Expected Improvement (EI) acquisition function to adaptively sample the decision space.
      
Both optimizers interact with the blower and network models and use utility functions for penalty
evaluations and safety checks. The returned result contains the optimal configuration along with
performance metrics such as total power consumption, aggregated flow, header pressure, runtime,
evaluation count, and energy savings percentage.

Configuration parameters are read from config.yaml and passed via the config dict.
All default values are strongly typed.
"""

import time
import logging
from abc import ABC, abstractmethod
from itertools import product, combinations
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from utils import calculate_switch_penalty, calculate_ramp_penalty

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HIGH_PENALTY: float = 1e6
DEFAULT_PRESSURE_TOLERANCE: float = 5.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def required_active_blowers(demand: float, rated_capacity: float, num_blowers: int) -> int:
    """Smallest number of blowers that can carry the demand at rated flow."""
    n = int(np.ceil(demand / max(rated_capacity, 1e-9) - 1e-12))
    return int(min(max(n, 1), num_blowers))


def evaluate_configuration(
    candidate_on_off: List[int],
    candidate_speeds: List[float],
    blower_model: Any,
    network_model: Any,
    demand: float,
    lambda_sw: float,
    lambda_ramp: float,
    pressure_tol: float,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate one on/off + speed candidate against demand, safety, and header pressure."""
    num_blowers = len(candidate_on_off)
    active_indices = [i for i, state in enumerate(candidate_on_off) if state == 1 and candidate_speeds[i] > 0]
    details: Dict[str, Any] = {"on_off": list(candidate_on_off), "speeds": list(candidate_speeds)}
    if not active_indices:
        return HIGH_PENALTY, details

    speed_sum = sum(candidate_speeds[i] for i in active_indices)
    if speed_sum <= 1e-9:
        return HIGH_PENALTY, details

    # Load-share in proportion to speed (faster machines take more flow).
    flows = [0.0] * num_blowers
    for i in active_indices:
        flows[i] = demand * candidate_speeds[i] / speed_sum

    total_power = 0.0
    discharge_pressures: List[float] = []
    p_set = float(getattr(network_model, "header_pressure_setpoint", 51.0))

    for i in active_indices:
        try:
            performance = blower_model.compute_performance(candidate_speeds[i], flows[i])
        except Exception:
            return HIGH_PENALTY, details
        if not blower_model.is_safe(performance):
            return HIGH_PENALTY, details
        if performance["pressure"] + 1e-6 < (p_set - pressure_tol):
            return HIGH_PENALTY, details
        total_power += performance["power"]
        discharge_pressures.append(performance["pressure"])

    aggregated_flow = float(sum(flows))
    if hasattr(network_model, "header_pressure"):
        header_pressure = network_model.header_pressure(aggregated_flow, discharge_pressures)
    else:
        header_pressure = network_model.compute_system_pressure(aggregated_flow)

    if aggregated_flow + 1e-6 < demand:
        return HIGH_PENALTY, details
    if abs(header_pressure - p_set) > pressure_tol:
        return HIGH_PENALTY, details

    switching_penalty = 0.0
    ramping_penalty = 0.0
    for i in range(num_blowers):
        on = 1 if i in active_indices else 0
        switching_penalty += calculate_switch_penalty(0, on, lambda_sw)
        if on:
            ramping_penalty += calculate_ramp_penalty(0.0, candidate_speeds[i], lambda_ramp)

    objective = total_power + switching_penalty + ramping_penalty
    details.update(
        {
            "total_flow": aggregated_flow,
            "header_pressure": float(header_pressure),
            "total_power": float(total_power),
            "switching_penalty": float(switching_penalty),
            "ramping_penalty": float(ramping_penalty),
            "objective": float(objective),
        }
    )
    return float(objective), details

# ---------------------------------------------------------------------------
# Abstract Base Optimizer Class
# ---------------------------------------------------------------------------
class Optimizer(ABC):
    """
    Abstract Optimizer base class defining the common interface.
    """

    @abstractmethod
    def optimize(
        self, blower_model: Any, network_model: Any, demand: float, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize the blower and network operating configuration.

        Parameters:
            blower_model: Instance of BlowerModel.
            network_model: Instance of NetworkModel.
            demand (float): Total required airflow (D).
            config (dict): Configuration dictionary loaded from config.yaml.

        Returns:
            dict: A dictionary containing:
                - "optimal_configuration": { "on_off": List[int], "speeds": List[float] }
                - "total_power": float
                - "total_flow": float
                - "header_pressure": float
                - "objective": float
                - "energy_savings_percent": float
                - "runtime_seconds": float
                - "evaluation_count": int
        """
        pass

# ---------------------------------------------------------------------------
# GridSearchOptimizer Class
# ---------------------------------------------------------------------------
class GridSearchOptimizer(Optimizer):
    """
    Optimizer implementation using an exhaustive grid search strategy.
    """

    def __init__(self, grid: List[float], penalty_params: Dict[str, float]) -> None:
        """
        Initialize the GridSearchOptimizer.

        Parameters:
            grid (List[float]): Discrete speed grid (e.g., [0.60, 0.65, ..., 1.00]).
            penalty_params (Dict[str, float]): Dictionary with keys 'lambda_sw' and 'lambda_ramp'.
        """
        self.speed_grid: List[float] = [float(v) for v in grid]
        self.lambda_sw: float = _as_float(penalty_params.get("lambda_sw"), 0.0)
        self.lambda_ramp: float = _as_float(penalty_params.get("lambda_ramp"), 0.0)
        logger.info("GridSearchOptimizer initialized with speed grid: %s, lambda_sw: %.3f, lambda_ramp: %.3f",
                    str(self.speed_grid), self.lambda_sw, self.lambda_ramp)

    def _generate_on_off_configurations(self, num_blowers: int) -> List[List[int]]:
        """
        Generate candidate on/off configurations for the given number of blowers.
        By industrial practice, typically one or two blowers are active.
        For num_blowers >= 2, only include configurations with at least one and at most (num_blowers - 1) active,
        but also allow all active if necessary.

        Parameters:
            num_blowers (int): Number of blowers in the system.

        Returns:
            List[List[int]]: A list of binary lists representing on/off states.
        """
        configurations = []
        # Generate all binary vectors (excluding all off)
        for r in range(1, num_blowers + 1):
            # Optionally, restrict to configurations with r active blowers.
            for indices in combinations(range(num_blowers), r):
                candidate = [0] * num_blowers
                for idx in indices:
                    candidate[idx] = 1
                configurations.append(candidate)
        return configurations

    def _evaluate_candidate(
        self,
        candidate_on_off: List[int],
        candidate_speeds: List[float],
        blower_model: Any,
        network_model: Any,
        demand: float,
        pressure_tol: float = DEFAULT_PRESSURE_TOLERANCE,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a candidate configuration.

        For each blower:
            - If active (on_off=1), assign candidate speed and flow = demand / (number of active blowers)
            - If inactive, speed is set to 0 and flow = 0.
        Then compute performance metrics via blower_model and network_model.
        Penalties for switching (assumed previous state off, i.e. 0) and ramping (assumed previous speed 0) are added.

        Constraints:
            - Total aggregated flow >= demand.
            - Header pressure within [setpoint - pressure_tol, setpoint + pressure_tol].
            - Each active blower's operating point must be safe.

        Returns:
            Tuple[float, Dict[str, Any]]:
                - Objective value (lower is better). Infeasible candidates get HIGH_PENALTY.
                - Details dictionary with computed metrics: on_off, speeds, total_power, total_flow, header_pressure.
        """
        pressure_tol = float(getattr(network_model, "pressure_tolerance", pressure_tol))
        return evaluate_configuration(
            candidate_on_off,
            candidate_speeds,
            blower_model,
            network_model,
            demand,
            self.lambda_sw,
            self.lambda_ramp,
            pressure_tol,
        )

    def optimize(
        self, blower_model: Any, network_model: Any, demand: float, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform grid search optimization over candidate on/off configurations and speed combinations.

        Parameters:
            blower_model: Instance of BlowerModel.
            network_model: Instance of NetworkModel.
            demand (float): Required airflow demand.
            config (dict): Configuration dictionary.

        Returns:
            dict: Result dictionary with optimal configuration and evaluation metrics.
        """
        start_time: float = time.time()
        evaluation_count: int = 0
        best_objective: float = HIGH_PENALTY
        best_details: Dict[str, Any] = {}

        # Get number of blowers from config or default to 3.
        num_blowers: int = int(config.get("num_blowers", 3))
        rated_capacity: float = float(getattr(blower_model, "rated_capacity", 4600.0))
        n_active: int = required_active_blowers(demand, rated_capacity, num_blowers)
        on_off = [1] * n_active + [0] * (num_blowers - n_active)
        speed_grid: List[float] = self.speed_grid
        active_indices = list(range(n_active))

        for speeds_active in product(speed_grid, repeat=len(active_indices)):
            candidate_speeds = [0.0] * num_blowers
            for j, idx in enumerate(active_indices):
                candidate_speeds[idx] = float(speeds_active[j])
            objective, details = self._evaluate_candidate(
                on_off, candidate_speeds, blower_model, network_model, demand, DEFAULT_PRESSURE_TOLERANCE
            )
            evaluation_count += 1
            if objective < best_objective:
                best_objective = objective
                best_details = details.copy()

        runtime_seconds: float = time.time() - start_time

        if not best_details or best_objective >= HIGH_PENALTY / 10.0:
            logger.warning("Grid search found no feasible point for demand=%.1f", demand)
            best_details = {
                "on_off": on_off,
                "speeds": [0.0] * num_blowers,
                "total_power": float("nan"),
                "total_flow": float("nan"),
                "header_pressure": float("nan"),
            }

        # Compute energy savings percentage relative to a baseline.
        # For this example, we assume baseline is operating all blowers at full speed (1.0) with equal flow split.
        baseline_total_power: float = 0.0
        baseline_on_off: List[int] = [1] * num_blowers
        active_count_baseline: int = num_blowers
        flow_baseline: float = demand / active_count_baseline if active_count_baseline > 0 else 0.0
        blower_flows_baseline: List[float] = []
        for _ in range(num_blowers):
            perf = blower_model.compute_performance(1.0, flow_baseline)
            baseline_total_power += perf.get("power", 0.0)
            blower_flows_baseline.append(perf.get("flow", 0.0))
        baseline_energy = baseline_total_power

        best_power = best_details.get("total_power", baseline_energy)
        energy_savings_percent: float = 0.0
        if baseline_energy > 0 and best_power == best_power:  # not NaN
            energy_savings_percent = (1 - best_power / baseline_energy) * 100

        result: Dict[str, Any] = {
            "optimal_configuration": {
                "on_off": best_details.get("on_off", []),
                "speeds": best_details.get("speeds", []),
            },
            "total_power": best_details.get("total_power", np.nan),
            "total_flow": best_details.get("total_flow", np.nan),
            "header_pressure": best_details.get("header_pressure", np.nan),
            "objective": best_objective,
            "energy_savings_percent": energy_savings_percent,
            "runtime_seconds": runtime_seconds,
            "evaluation_count": evaluation_count,
        }

        logger.info("Grid Search Optimization completed in %.2f seconds with %d evaluations.",
                    runtime_seconds, evaluation_count)
        return result

# ---------------------------------------------------------------------------
# SBOOptimizer Class
# ---------------------------------------------------------------------------
class SBOOptimizer(Optimizer):
    """
    Optimizer implementation using Safe Bayesian Optimization (SBO).
    """

    def __init__(self, init_samples: int, gp_params: Dict[str, Any], penalty_params: Dict[str, float]) -> None:
        """
        Initialize the SBOOptimizer.

        Parameters:
            init_samples (int): Minimum number of warm-up samples.
            gp_params (Dict[str, Any]): Parameters for the Gaussian Process (e.g., kernel type).
            penalty_params (Dict[str, float]): Dictionary with keys 'lambda_sw' and 'lambda_ramp'.
        """
        self.init_samples: int = int(init_samples)
        self.lambda_sw: float = _as_float(penalty_params.get("lambda_sw"), 0.0)
        self.lambda_ramp: float = _as_float(penalty_params.get("lambda_ramp"), 0.0)
        self.early_stopping_threshold: float = _as_float(gp_params.get("early_stopping"), 1e-3)
        if self.early_stopping_threshold <= 0:
            self.early_stopping_threshold = 1e-3
        self.max_iter: int = int(_as_float(gp_params.get("max_iter"), 30))

        # Set up the Gaussian Process regressor with an RBF kernel.
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp: GaussianProcessRegressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        logger.info("SBOOptimizer initialized with %d warm-up samples, early stopping threshold %.5f, max_iter=%d.",
                    self.init_samples, self.early_stopping_threshold, self.max_iter)

    def _process_candidate(self, candidate: List[float], min_speed: float = 0.60) -> List[float]:
        """
        Process a candidate vector by thresholding: if a candidate speed is below min_speed, set it to 0 (blower off).

        Parameters:
            candidate (List[float]): Raw candidate speeds.
            min_speed (float): Minimum operating speed for an active blower.

        Returns:
            List[float]: Processed candidate speeds.
        """
        return [speed if speed >= min_speed else 0.0 for speed in candidate]

    def _evaluate_candidate(
        self,
        candidate: List[float],
        blower_model: Any,
        network_model: Any,
        demand: float,
        pressure_tol: float = DEFAULT_PRESSURE_TOLERANCE,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a candidate vector for SBO.

        The candidate vector represents speeds for each blower. A value below min_speed implies the blower is off.
        The on/off configuration is derived by: on_off = 1 if speed >= min_speed else 0.
        For each active blower, assign flow = demand / (number of active blowers).

        Parameters:
            candidate (List[float]): Raw candidate vector.
            blower_model: Instance of BlowerModel.
            network_model: Instance of NetworkModel.
            demand (float): Required airflow demand.
            pressure_tol (float): Allowed deviation from header pressure setpoint.

        Returns:
            Tuple[float, Dict[str, Any]]:
                - Objective value (float) [HIGH_PENALTY if infeasible].
                - Details dictionary with candidate configuration and computed metrics.
        """
        min_speed: float = 0.60
        processed_candidate = self._process_candidate(candidate, min_speed)
        on_off: List[int] = [1 if speed >= min_speed else 0 for speed in processed_candidate]
        pressure_tol = float(getattr(network_model, "pressure_tolerance", pressure_tol))
        return evaluate_configuration(
            on_off,
            processed_candidate,
            blower_model,
            network_model,
            demand,
            self.lambda_sw,
            self.lambda_ramp,
            pressure_tol,
        )

    def _expected_improvement(self, x: np.ndarray, X_sample: np.ndarray, Y_sample: np.ndarray, xi: float = 0.01) -> float:
        """
        Compute the Expected Improvement (EI) at point x.

        Parameters:
            x (np.ndarray): Candidate point (1D array).
            X_sample (np.ndarray): Sampled input points.
            Y_sample (np.ndarray): Sampled objective values.
            xi (float): Exploration-exploitation parameter.

        Returns:
            float: Expected Improvement value.
        """
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        mu = float(np.asarray(mu).reshape(-1)[0])
        sigma = float(np.asarray(sigma).reshape(-1)[0])
        if sigma <= 1e-12:
            return 0.0
        y_best = float(np.min(Y_sample))
        imp = y_best - mu - xi
        Z = imp / sigma
        ei = imp * float(norm.cdf(Z)) + sigma * float(norm.pdf(Z))
        return float(ei)

    def optimize(
        self, blower_model: Any, network_model: Any, demand: float, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform Safe Bayesian Optimization over the candidate vector space.

        The domain is defined for each blower as [0, 1.0] where values < 0.60 indicate off.
        Warm-up with a predefined number of random samples is performed, followed by iterative
        candidate selection based on the Expected Improvement (EI) acquisition function with safety filtering.

        Parameters:
            blower_model: Instance of BlowerModel.
            network_model: Instance of NetworkModel.
            demand (float): Required airflow demand.
            config (dict): Configuration dictionary.

        Returns:
            dict: Result dictionary with optimal configuration and evaluation metrics.
        """
        start_time: float = time.time()
        evaluation_count: int = 0
        rng = np.random.default_rng(42)

        num_blowers: int = int(config.get("num_blowers", 3))
        rated_capacity: float = float(getattr(blower_model, "rated_capacity", 4600.0))
        n_active: int = required_active_blowers(demand, rated_capacity, num_blowers)
        lower_bound: float = 0.0
        upper_bound: float = 1.0

        X_samples: List[np.ndarray] = []
        Y_samples: List[float] = []
        details_list: List[Dict[str, Any]] = []

        def random_candidate() -> List[float]:
            cand = [0.0] * num_blowers
            for i in range(n_active):
                cand[i] = float(rng.uniform(0.70, 1.00))
            return cand

        for _ in range(self.init_samples):
            candidate = random_candidate()
            obj, det = self._evaluate_candidate(candidate, blower_model, network_model, demand, DEFAULT_PRESSURE_TOLERANCE)
            evaluation_count += 1
            X_samples.append(np.array(self._process_candidate(candidate)))
            Y_samples.append(obj)
            details_list.append(det)

        X_sample = np.array(X_samples)
        Y_sample = np.array(Y_samples)

        try:
            self.gp.fit(X_sample, Y_sample)
        except Exception as exc:
            logger.warning("Initial GP fit failed (%s); continuing with random search.", exc)

        best_index = int(np.argmin(Y_sample))
        best_objective = Y_sample[best_index]
        best_details = details_list[best_index]

        # SBO iterative loop.
        iteration: int = 0
        while iteration < self.max_iter:
            iteration += 1
            # Generate candidate pool: sample 100 random candidates.
            candidate_pool = [np.array(random_candidate()) for _ in range(100)]
            # Compute EI for each candidate.
            eis = []
            for cand in candidate_pool:
                cand_processed = self._process_candidate(cand.tolist())
                x_cand = np.array(cand_processed)
                ei = self._expected_improvement(x_cand, X_sample, Y_sample)
                eis.append(ei)
            # Select candidate with maximum EI.
            max_index = int(np.argmax(eis))
            next_candidate = candidate_pool[max_index].tolist()

            # Evaluate the selected candidate.
            obj, det = self._evaluate_candidate(next_candidate, blower_model, network_model, demand, DEFAULT_PRESSURE_TOLERANCE)
            evaluation_count += 1

            # Update best if improved.
            if obj < best_objective:
                improvement = best_objective - obj
                best_objective = obj
                best_details = det.copy()
                logger.debug("Iteration %d: New best objective %.5f found (improvement=%.5f).", iteration, best_objective, improvement)
                # Check early stopping based on improvement threshold.
                if improvement < self.early_stopping_threshold:
                    logger.info("Early stopping triggered: improvement %.5f below threshold %.5f.", improvement, self.early_stopping_threshold)
                    break

            # Append new sample and re-fit GP.
            x_new = np.array(self._process_candidate(next_candidate))
            X_sample = np.vstack((X_sample, x_new))
            Y_sample = np.append(Y_sample, obj)
            details_list.append(det)
            try:
                self.gp.fit(X_sample, Y_sample)
            except Exception:
                pass

        runtime_seconds: float = time.time() - start_time

        # Compute energy savings relative to baseline.
        baseline_total_power: float = 0.0
        num_blowers_default = num_blowers
        baseline_flow = demand / num_blowers_default if num_blowers_default > 0 else 0.0
        for _ in range(num_blowers_default):
            perf_baseline = blower_model.compute_performance(1.0, baseline_flow)
            baseline_total_power += perf_baseline.get("power", 0.0)
        baseline_energy = baseline_total_power

        energy_savings_percent: float = 0.0
        best_power = best_details.get("total_power", baseline_energy)
        if baseline_energy > 0 and best_power == best_power:
            energy_savings_percent = (1 - best_power / baseline_energy) * 100

        result: Dict[str, Any] = {
            "optimal_configuration": {
                "on_off": best_details.get("on_off", []),
                "speeds": best_details.get("speeds", []),
            },
            "total_power": best_details.get("total_power", np.nan),
            "total_flow": best_details.get("total_flow", np.nan),
            "header_pressure": best_details.get("header_pressure", np.nan),
            "objective": best_objective,
            "energy_savings_percent": energy_savings_percent,
            "runtime_seconds": runtime_seconds,
            "evaluation_count": evaluation_count,
        }

        logger.info("SBO Optimization completed in %.2f seconds with %d evaluations.",
                    runtime_seconds, evaluation_count)
        return result
