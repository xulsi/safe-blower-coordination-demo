"""blower_model.py

Blower performance from a synthetic vendor-style map, affinity-law scaling,
and surge / overheat envelopes. Header pressure is produced at the map point
(Q, N); a VFD can only *reduce* speed, so an operating point is capable of
meeting the 51 kPa setpoint when predicted Δp >= setpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class BlowerModel:
    """Single-blower map with affinity-law off-design corrections."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.rated_capacity: float = _as_float(config.get("rated_capacity"), 4600.0)
        self.motor_power: float = _as_float(config.get("motor_power"), 78.0)
        self.speed_range = list(config.get("speed_range", [0.60, 1.00]))
        self.speed_step: float = _as_float(config.get("speed_step"), 0.05)

        poly_config: Dict[str, Any] = config.get("polynomial_fit", {}) or {}
        self.poly_target_mae: float = _as_float(poly_config.get("target_mae"), 0.91)
        self.poly_target_max_dev: float = _as_float(poly_config.get("target_max_dev"), 1.82)

        map_config: Dict[str, Any] = config.get("map", {}) or {}
        self.shutoff_pressure: float = _as_float(map_config.get("shutoff_pressure"), 80.0)
        self.rated_pressure: float = _as_float(map_config.get("rated_pressure"), 62.0)

        safety_config: Dict[str, Any] = config.get("safety", {}) or {}
        self.surge_flow_fraction: float = _as_float(safety_config.get("surge_flow_fraction"), 0.50)
        self.overheat_flow_fraction: float = _as_float(safety_config.get("overheat_flow_fraction"), 1.05)
        self.overheat_temperature: float = _as_float(safety_config.get("overheat_temperature"), 120.0)

        self.nominal_speed: float = 1.0
        self.density_ratio: float = 1.0
        self.ambient_temperature: float = 25.0

        logger.info(
            "BlowerModel initialized: Q_rated=%.0f Nm3, P_rated=%.0f kW, speed=%s",
            self.rated_capacity,
            self.motor_power,
            self.speed_range,
        )

    def delta_p0(self, flow_nominal: float) -> float:
        """Gauge pressure (kPa) at N=1 for an equivalent-flow operating point."""
        frac = float(np.clip(flow_nominal / max(self.rated_capacity, 1e-9), 0.0, 1.2))
        return self.shutoff_pressure + (self.rated_pressure - self.shutoff_pressure) * frac

    def power0(self, flow_nominal: float) -> float:
        """Shaft power (kW) at N=1. Roughly linear with flow on a turbo map."""
        frac = float(np.clip(flow_nominal / max(self.rated_capacity, 1e-9), 0.0, 1.2))
        return self.motor_power * frac

    def compute_performance(self, speed: float, flow: float) -> Dict[str, float]:
        """
        Map (actual flow Q, speed N) through affinity laws.

        Q0 = Q / N,  Δp = Δp0(Q0) * N^2,  P = P0(Q0) * N^3.
        The returned flow is the requested actual flow (not re-scaled).
        """
        if speed < self.speed_range[0] - 1e-9 or speed > self.speed_range[1] + 1e-9:
            raise ValueError(f"Candidate speed {speed} is outside {self.speed_range}.")
        if speed <= 1e-9:
            raise ValueError("Speed must be positive for an active blower.")

        flow_nominal = flow / speed
        pressure = self.delta_p0(flow_nominal) * (speed ** 2) * self.density_ratio
        power = self.power0(flow_nominal) * (speed ** 3) * self.density_ratio
        # Discharge temperature rises with compression ratio / speed (synthetic).
        temperature = self.ambient_temperature + 35.0 * speed + 0.004 * max(flow, 0.0)

        return {
            "pressure": float(pressure),
            "flow": float(flow),
            "power": float(power),
            "speed": float(speed),
            "temperature": float(temperature),
        }

    def apply_affinity(self, speed: float, performance: Dict[str, float]) -> Dict[str, float]:
        """Re-evaluate a nominal-speed point at a new speed, keeping map position."""
        flow_nominal = performance["flow"]
        return self.compute_performance(speed, flow_nominal * speed)

    def surge_flow(self, speed: float) -> float:
        return self.surge_flow_fraction * self.rated_capacity * speed

    def overheat_flow(self, speed: float) -> float:
        return self.overheat_flow_fraction * self.rated_capacity * speed

    def is_safe(self, operating_point: Dict[str, float]) -> bool:
        speed = float(operating_point.get("speed", self.nominal_speed))
        flow = float(operating_point.get("flow", 0.0))
        temperature = float(operating_point.get("temperature", self.ambient_temperature))
        if flow < self.surge_flow(speed) - 1e-6:
            return False
        if flow > self.overheat_flow(speed) + 1e-6:
            return False
        if temperature > self.overheat_temperature + 1e-6:
            return False
        return True

    def can_meet_setpoint(self, speed: float, flow: float, p_set: float, tol: float) -> bool:
        try:
            perf = self.compute_performance(speed, flow)
        except ValueError:
            return False
        return perf["pressure"] + 1e-6 >= (p_set - tol)
