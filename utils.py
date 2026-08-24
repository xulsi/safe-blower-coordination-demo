"""
utils.py

This module provides shared helper functions and utilities for the project,
including configuration loading, polynomial fitting routines, error metric
calculations, safety verifications, and penalty computations.

Functions:
    load_config(config_filepath: str) -> dict
    polynomial_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray
    evaluate_fit(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray,
                 target_mae: float = 0.91, target_max_dev: float = 1.82) -> dict
    is_safe_operating_point(operating_point: dict, safety_limits: dict) -> bool
    calculate_switch_penalty(previous_state: int, current_state: int, lambda_sw: float) -> float
    calculate_ramp_penalty(previous_speed: float, current_speed: float, lambda_ramp: float) -> float
    mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float
    root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float
    max_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float
"""

import os
import logging
from typing import Any, Dict

import numpy as np
import yaml

# Configure logging with level INFO (adjust as needed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def merge_dicts(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries. For each key in the default dict, if the key is not present in custom,
    it is added. If the value corresponding to the key is a dict in both, merge them recursively.
    
    Parameters:
        default (Dict[str, Any]): The default configuration dictionary.
        custom (Dict[str, Any]): The custom configuration dictionary loaded from file.
    
    Returns:
        Dict[str, Any]: The merged configuration dictionary.
    """
    for key, default_value in default.items():
        if key not in custom:
            custom[key] = default_value
        else:
            custom_value = custom.get(key)
            if isinstance(default_value, dict) and isinstance(custom_value, dict):
                custom[key] = merge_dicts(default_value, custom_value)
    return custom


def load_config(config_filepath: str) -> dict:
    """
    Load and parse the YAML configuration file.

    Parameters:
        config_filepath (str): The path to the YAML configuration file.

    Returns:
        dict: A nested dictionary with configuration values.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file exists but cannot be parsed.
    """
    if not os.path.exists(config_filepath):
        logging.error("Config file '%s' does not exist.", config_filepath)
        raise FileNotFoundError(f"Config file '{config_filepath}' not found.")

    with open(config_filepath, 'r') as file:
        try:
            config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logging.error("Error parsing YAML file: %s", e)
            raise e

    # Define default configuration based on the provided config.yaml details.
    default_config = {
        'blower_model': {
            'rated_capacity': 4600,         # Nm3
            'motor_power': 78,              # kW
            'speed_range': [0.60, 1.00],
            'speed_step': 0.05,
            'polynomial_fit': {
                'target_mae': 0.91,         # kPa
                'target_max_dev': 1.82      # kPa
            },
            'surge_flow_fraction': 0.50,
            'overheat_flow_fraction': 1.05,
            'overheat_temperature': 120.0,
        },
        'network_model': {
            'header_pressure_setpoint': 51,  # kPa
            'pipe_diameter': 300,            # mm
            'static_pressure': 45.0,
            'resistance_coefficient': 2.5e-7,
            'pressure_tolerance': 3.0,
            'sensitivity_analysis': {
                'perturbation': 0.05
            }
        },
        'optimizer': {
            'penalty': {
                'lambda_sw': 0.0,           # Default penalty weight for switching (should be set in config)
                'lambda_ramp': 0.0          # Default penalty weight for ramping (should be set in config)
            },
            'grid_search': {
                'speed_grid': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
            },
            'sbo': {
                'init_samples': 5,
                'early_stopping': 0.05,
                'max_iter': 30,
                'gp_kernel': "RBF",
                'acquisition': "Expected Improvement"
            }
        },
        'simulation': {
            'demand_scenarios': [],        # List of scenarios; if empty, must be provided in config.
            'evaluation_metrics': ['energy_savings_percent', 'runtime_seconds', 'evaluation_count']
        }
    }

    config = merge_dicts(default_config, config_data or {})
    logging.info("Configuration loaded and merged successfully.")
    return config


def polynomial_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Fit a polynomial of the specified degree to the data points.

    Parameters:
        x (np.ndarray): 1D array of independent variable data.
        y (np.ndarray): 1D array of dependent variable data.
        degree (int): Degree of the polynomial to fit.

    Returns:
        np.ndarray: Array of polynomial coefficients (highest degree first).
    
    Raises:
        TypeError: If x or y is not of type np.ndarray.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs x and y must be numpy.ndarray types.")
    coeffs = np.polyfit(x, y, degree)
    logging.debug("Polynomial coefficients computed: %s", coeffs)
    return coeffs


def evaluate_fit(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray,
                 target_mae: float = 0.91, target_max_dev: float = 1.82) -> dict:
    """
    Evaluate the quality of the polynomial fit by calculating error metrics.

    Parameters:
        x (np.ndarray): 1D array of independent variable data.
        y (np.ndarray): 1D array of true dependent variable data.
        coeffs (np.ndarray): Polynomial coefficients.
        target_mae (float): Target mean absolute error threshold.
        target_max_dev (float): Target maximum absolute deviation threshold.

    Returns:
        dict: Dictionary with keys:
            - 'mae': Mean Absolute Error
            - 'rmse': Root Mean Square Error
            - 'max_deviation': Maximum absolute deviation
            - 'fit_ok': Boolean flag if error metrics meet targets
            - 'coeffs': The used polynomial coefficients
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(coeffs, np.ndarray):
        raise TypeError("x, y, and coeffs must be numpy.ndarray types.")

    y_pred = np.polyval(coeffs, x)
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    max_dev = float(np.max(np.abs(y - y_pred)))
    fit_ok = (mae <= target_mae) and (max_dev <= target_max_dev)

    if not fit_ok:
        logging.warning("Polynomial fit did not meet target thresholds: MAE=%.3f (target %.3f), Max Dev=%.3f (target %.3f)",
                        mae, target_mae, max_dev, target_max_dev)
    else:
        logging.info("Polynomial fit meets the target error thresholds.")

    return {
        'mae': mae,
        'rmse': rmse,
        'max_deviation': max_dev,
        'fit_ok': fit_ok,
        'coeffs': coeffs
    }


def is_safe_operating_point(operating_point: dict, safety_limits: dict) -> bool:
    """
    Check if the operating point is within the defined safety limits.
    
    The operating_point dictionary is expected to include keys such as 'pressure', 'flow', and 'temperature'.
    The safety_limits dictionary should include 'surge_limit' and 'overheat_limit'. If these limits are not
    specified numerically, the check will log a debug message and assume the operating point is safe.

    Parameters:
        operating_point (dict): Dictionary with operating parameters (e.g., pressure, flow, temperature).
        safety_limits (dict): Dictionary with safety limits (e.g., surge_limit, overheat_limit).

    Returns:
        bool: True if the operating point is safe; False otherwise.
    """
    is_safe = True

    surge_limit = safety_limits.get('surge_limit')
    overheat_limit = safety_limits.get('overheat_limit')

    # Check surge safety if a numeric limit is provided; assume operating_point contains 'flow'
    if isinstance(surge_limit, (int, float)):
        flow = operating_point.get('flow')
        if flow is None:
            logging.error("Operating point is missing 'flow' key for surge check.")
            is_safe = False
        elif flow < surge_limit:
            logging.info("Operating point flow (%.3f) is below the surge safety limit (%.3f).", flow, surge_limit)
            is_safe = False
    else:
        logging.debug("Surge limit not specified numerically; skipping surge safety check.")

    # Check overheat safety if a numeric limit is provided; assume operating_point contains 'temperature'
    if isinstance(overheat_limit, (int, float)):
        temperature = operating_point.get('temperature')
        if temperature is None:
            logging.error("Operating point is missing 'temperature' key for overheat check.")
            is_safe = False
        elif temperature > overheat_limit:
            logging.info("Operating point temperature (%.3f) exceeds the overheat limit (%.3f).", temperature, overheat_limit)
            is_safe = False
    else:
        logging.debug("Overheat limit not specified numerically; skipping overheat safety check.")

    return is_safe


def calculate_switch_penalty(previous_state: int, current_state: int, lambda_sw: float) -> float:
    """
    Calculate the penalty for switching the blower on/off.

    Parameters:
        previous_state (int): Previous on/off state (0 for off, 1 for on).
        current_state (int): Current on/off state (0 for off, 1 for on).
        lambda_sw (float): Weight factor for switching penalty.

    Returns:
        float: Calculated penalty cost.
    """
    if not isinstance(previous_state, int) or not isinstance(current_state, int):
        raise TypeError("previous_state and current_state must be integers (0 or 1).")
    if not isinstance(lambda_sw, (int, float)):
        raise TypeError("lambda_sw must be a numeric value.")

    penalty = lambda_sw if previous_state != current_state else 0.0
    logging.debug("Switch penalty calculated: %.3f (Previous: %d, Current: %d, Lambda: %.3f)",
                  penalty, previous_state, current_state, lambda_sw)
    return float(penalty)


def calculate_ramp_penalty(previous_speed: float, current_speed: float, lambda_ramp: float) -> float:
    """
    Calculate the penalty cost for ramping the blower speed.

    Parameters:
        previous_speed (float): Previous normalized speed.
        current_speed (float): Current normalized speed.
        lambda_ramp (float): Weight factor for ramping penalty.

    Returns:
        float: Calculated penalty cost based on the absolute difference in speeds.
    """
    if not isinstance(previous_speed, (int, float)) or not isinstance(current_speed, (int, float)):
        raise TypeError("previous_speed and current_speed must be numeric values.")
    if not isinstance(lambda_ramp, (int, float)):
        raise TypeError("lambda_ramp must be a numeric value.")

    penalty = lambda_ramp * abs(current_speed - previous_speed)
    logging.debug("Ramp penalty calculated: %.3f (Previous Speed: %.3f, Current Speed: %.3f, Lambda: %.3f)",
                  penalty, previous_speed, current_speed, lambda_ramp)
    return float(penalty)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean absolute error between true and predicted values.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Mean absolute error.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy.ndarray types.")
    mae = float(np.mean(np.abs(y_true - y_pred)))
    logging.debug("Mean Absolute Error computed: %.3f", mae)
    return mae


def root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the root mean square error (RMSE) between true and predicted values.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Root mean square error.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy.ndarray types.")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    logging.debug("Root Mean Square Error computed: %.3f", rmse)
    return rmse


def max_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the maximum absolute deviation between true and predicted values.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Maximum absolute deviation.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy.ndarray types.")
    deviation = float(np.max(np.abs(y_true - y_pred)))
    logging.debug("Maximum Deviation computed: %.3f", deviation)
    return deviation
