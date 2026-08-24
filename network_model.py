"""
network_model.py

This module defines the NetworkModel class that simulates the air distribution network.
It computes the system (header) pressure drop based on a quadratic relation derived from the
Darcy–Weisbach equation and aggregates flows from multiple blowers.

Key Methods:
    - __init__(params: dict): Initializes network parameters from configuration.
    - compute_system_pressure(flow: float) -> float: Computes system pressure using:
          ∆p_sys = static_pressure + resistance_coefficient * (flow)^2
    - aggregate_blower_flow(blower_flows: list[float]) -> float: Aggregates individual blower flows.
    
Configuration Parameters (from config.yaml, under 'network_model'):
    - header_pressure_setpoint: Target header pressure (default: 51 kPa).
    - pipe_diameter: Pipe diameter in mm (default: 300 mm).
    - static_pressure: Fixed static pressure loss (∆p_static); if not provided, defaults to 0.0.
    - resistance_coefficient: Aggregate resistance coefficient (k); if not provided, defaults to 0.0.
    - sensitivity_analysis: Contains 'perturbation' parameter (default: 0.05 for +/- 5% variation).
    
All parameters are strongly typed and default values are assigned if configuration entries
are missing or not numeric.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class NetworkModel:
    """
    A class representing the air distribution network. It computes the system pressure drop
    using a quadratic relationship and aggregates flows from multiple blowers.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the NetworkModel with network configuration parameters.

        Parameters:
            params (Dict[str, Any]): Dictionary of network parameters expected to be provided
                                     from the configuration (config.yaml) under 'network_model'.
                                     Expected keys:
                                        - header_pressure_setpoint (numeric): Target header pressure (kPa).
                                        - pipe_diameter (numeric): Pipe diameter in mm.
                                        - static_pressure (numeric): Fixed static pressure loss.
                                        - resistance_coefficient (numeric): Coefficient 'k' for pressure drop.
                                        - sensitivity_analysis: dict containing 'perturbation' (default: 0.05).
        """
        try:
            self.header_pressure_setpoint: float = float(params.get("header_pressure_setpoint", 51))
        except (ValueError, TypeError):
            logger.warning("Invalid header_pressure_setpoint; using default value 51 kPa.")
            self.header_pressure_setpoint = 51.0

        try:
            self.pipe_diameter: float = float(params.get("pipe_diameter", 300))
        except (ValueError, TypeError):
            logger.warning("Invalid pipe_diameter; using default value 300 mm.")
            self.pipe_diameter = 300.0

        # Retrieve static_pressure; if not numeric, warn and set default value.
        static_pressure_raw = params.get("static_pressure", 0)
        try:
            self.static_pressure: float = float(static_pressure_raw)
        except (ValueError, TypeError):
            logger.warning("Static pressure value '%s' is not numeric. Using default value 0.0 kPa.", static_pressure_raw)
            self.static_pressure = 0.0

        # Retrieve resistance_coefficient; if not numeric, warn and set default value.
        resistance_coeff_raw = params.get("resistance_coefficient", 0)
        try:
            self.resistance_coefficient: float = float(resistance_coeff_raw)
        except (ValueError, TypeError):
            logger.warning("Resistance coefficient value '%s' is not numeric. Using default value 0.0.", resistance_coeff_raw)
            self.resistance_coefficient = 0.0

        # Sensitivity analysis parameters (optional)
        sensitivity_config = params.get("sensitivity_analysis", {})
        try:
            self.perturbation: float = float(sensitivity_config.get("perturbation", 0.05))
        except (ValueError, TypeError):
            logger.warning("Sensitivity perturbation is invalid. Using default value 0.05.")
            self.perturbation = 0.05

        try:
            self.pressure_tolerance: float = float(params.get("pressure_tolerance", 3.0))
        except (ValueError, TypeError):
            self.pressure_tolerance = 3.0

        logger.info(
            "NetworkModel initialized: p_set=%.1f kPa, D=%.0f mm, p_static=%.1f kPa, k=%.3e",
            self.header_pressure_setpoint,
            self.pipe_diameter,
            self.static_pressure,
            self.resistance_coefficient,
        )

    def compute_system_pressure(self, flow: float) -> float:
        """
        Compute the system pressure drop for a given total flow using the quadratic system curve.

        The mathematical relationship:
            ∆p_sys(flow) = static_pressure + resistance_coefficient * (flow)²

        Parameters:
            flow (float): Total airflow delivered by the blowers (in Nm³ or equivalent units).

        Returns:
            float: Computed system pressure (in kPa).
        """
        if not isinstance(flow, (int, float)):
            logger.error("Flow must be a numeric value, got type %s.", type(flow))
            raise TypeError("Flow must be a numeric value.")

        computed_pressure: float = self.static_pressure + self.resistance_coefficient * (flow ** 2)
        logger.debug("System curve pressure for Q=%.1f Nm3: %.2f kPa", flow, computed_pressure)
        return computed_pressure

    def header_pressure(self, flow: float, blower_discharge_pressures: Optional[list] = None) -> float:
        """
        Header pressure under VFD + pressure-loop control.

        The PLC holds the header near the 51 kPa setpoint whenever every active
        blower can produce at least that discharge pressure. Otherwise the
        uncontrolled system-curve value is returned.
        """
        system_pressure = self.compute_system_pressure(flow)
        if blower_discharge_pressures:
            min_discharge = min(blower_discharge_pressures)
            if min_discharge + 1e-6 >= (self.header_pressure_setpoint - self.pressure_tolerance):
                return float(self.header_pressure_setpoint)
            return float(min(min_discharge, system_pressure) if system_pressure else min_discharge)
        return float(system_pressure)

    def aggregate_blower_flow(self, blower_flows: List[float]) -> float:
        """
        Aggregate the individual blower flows from a list to produce the total system airflow.

        Parameters:
            blower_flows (List[float]): A list containing individual blower flow values.

        Returns:
            float: Total aggregated air flow.
        """
        if not isinstance(blower_flows, list):
            logger.error("blower_flows must be a list of numeric values.")
            raise TypeError("blower_flows must be a list of numeric values.")

        total_flow: float = 0.0
        for index, flow in enumerate(blower_flows):
            if not isinstance(flow, (int, float)):
                logger.error("Flow value at index %d is not numeric.", index)
                raise TypeError(f"Flow value at index {index} must be numeric.")
            total_flow += float(flow)

        logger.debug("Aggregated blower flow from %d units: %.2f Nm3", len(blower_flows), total_flow)
        return total_flow


# Optional testing section if run as a script
if __name__ == "__main__":
    # Example configuration dictionary for NetworkModel, simulating values from config.yaml.
    example_config: Dict[str, Any] = {
        "header_pressure_setpoint": 51,
        "pipe_diameter": 300,
        "static_pressure": 5.0,  # Example static pressure in kPa.
        "resistance_coefficient": 0.0001,  # Example resistance coefficient.
        "sensitivity_analysis": {
            "perturbation": 0.05
        }
    }

    # Instantiate the NetworkModel with the example configuration.
    network_model = NetworkModel(example_config)

    # Simulate aggregated flows from three blowers.
    blower_flows_example = [1500.0, 1600.0, 1550.0]  # in Nm³ (example data)
    total_flow = network_model.aggregate_blower_flow(blower_flows_example)
    system_pressure = network_model.compute_system_pressure(total_flow)

    print("Total Aggregated Flow: {:.2f} Nm³".format(total_flow))
    print("Computed System Pressure: {:.2f} kPa".format(system_pressure))
