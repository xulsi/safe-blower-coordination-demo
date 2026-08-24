"""
main.py

This is the main entry point of the simulation application. It performs the following steps:
  1. Parses command-line arguments for optimizer choice (Grid Search or SBO).
  2. Loads and parses the configuration from config.yaml.
  3. Instantiates the key simulation models: BlowerModel, NetworkModel, and the chosen Optimizer.
  4. Creates a SimulationRunner that integrates these models.
  5. Iterates over the defined demand scenarios, runs the simulation, and prints the results.
  
All configuration values are read from config.yaml with default values provided as needed.
"""

import argparse
import logging
import sys
from typing import Dict, Any

# Import required modules from the project
from utils import load_config
from blower_model import BlowerModel
from network_model import NetworkModel
from optimizer import GridSearchOptimizer, SBOOptimizer, Optimizer
from simulation import SimulationRunner

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments including optimizer type.
    """
    parser = argparse.ArgumentParser(
        description="Simulation-based optimization for energy-efficient and safe blower coordination."
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="grid",
        choices=["grid", "sbo"],
        help="Optimizer to use: 'grid' for GridSearch or 'sbo' for Safe Bayesian Optimization. (Default: grid)",
    )
    return parser.parse_args()

def main() -> None:
    """
    Main function that sets up the simulation using configuration from config.yaml,
    instantiates the simulation components, and runs simulations for each demand scenario.
    """
    # Parse command-line arguments
    args = parse_arguments()
    optimizer_choice: str = args.optimizer.lower()
    logger.info("Optimizer chosen: %s", optimizer_choice)

    # Load configuration from config.yaml (using strong typing and default values)
    config_filepath: str = "config.yaml"
    try:
        config: Dict[str, Any] = load_config(config_filepath)
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        sys.exit(1)

    # Extract and validate sections from configuration.
    blower_config: Dict[str, Any] = config.get("blower_model", {})
    network_config: Dict[str, Any] = config.get("network_model", {})
    optimizer_config: Dict[str, Any] = config.get("optimizer", {})
    simulation_config: Dict[str, Any] = config.get("simulation", {})

    # Set default for number of blowers if not provided in config.
    num_blowers: int = int(config.get("num_blowers", 3))
    config["num_blowers"] = num_blowers
    logger.info("Number of blowers set to: %d", num_blowers)

    # Instantiate BlowerModel with its configuration.
    try:
        blower_model: BlowerModel = BlowerModel(blower_config)
    except Exception as e:
        logger.error("Error instantiating BlowerModel: %s", e)
        sys.exit(1)

    # Instantiate NetworkModel with its configuration.
    try:
        network_model: NetworkModel = NetworkModel(network_config)
    except Exception as e:
        logger.error("Error instantiating NetworkModel: %s", e)
        sys.exit(1)

    # Retrieve penalty parameters for the optimizer with defaults.
    penalty_params: Dict[str, float] = optimizer_config.get("penalty", {"lambda_sw": 0.0, "lambda_ramp": 0.0})
    
    # Instantiate the chosen optimizer based on command-line argument.
    optimizer: Optimizer
    if optimizer_choice == "sbo":
        # Get SBO specific configuration (with defaults if not provided)
        sbo_params: Dict[str, Any] = optimizer_config.get("sbo", {
            "init_samples": 5,
            "early_stopping": 1e-3,
            "gp_kernel": "RBF",  # Not used directly as the GP is set up in SBOOptimizer
            "acquisition": "Expected Improvement",
            "max_iter": 50
        })
        init_samples: int = int(sbo_params.get("init_samples", 5))
        optimizer = SBOOptimizer(init_samples, sbo_params, penalty_params)
        logger.info("Using SBOOptimizer with %d warm-up samples.", init_samples)
    else:
        # Default to GridSearchOptimizer with speed grid from configuration.
        grid_search_config: Dict[str, Any] = optimizer_config.get("grid_search", {
            "speed_grid": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        })
        speed_grid = grid_search_config.get("speed_grid", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        if not isinstance(speed_grid, list):
            logger.warning("Speed grid is not a list; using default speed grid.")
            speed_grid = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        optimizer = GridSearchOptimizer(speed_grid, penalty_params)
        logger.info("Using GridSearchOptimizer with speed grid: %s", str(speed_grid))

    # Create SimulationRunner instance with BlowerModel, NetworkModel, and chosen Optimizer.
    try:
        simulation_runner: SimulationRunner = SimulationRunner(blower_model, network_model, optimizer, config)
    except Exception as e:
        logger.error("Error initializing SimulationRunner: %s", e)
        sys.exit(1)

    # Retrieve demand scenarios from the simulation configuration.
    demand_scenarios = simulation_config.get("demand_scenarios", [])
    if not isinstance(demand_scenarios, list) or len(demand_scenarios) == 0:
        # If not provided, use a default scenario.
        default_demand: float = 1000.0
        logger.warning("No demand scenarios provided in configuration; using default demand: %.2f", default_demand)
        demand_scenarios = [{"scenario": "default", "demand": default_demand}]

    # Iterate over each demand scenario, run simulation, and output the results.
    for scenario in demand_scenarios:
        scenario_name = str(scenario.get("scenario", "Unnamed Scenario"))
        try:
            demand_value = float(scenario.get("demand", 1000.0))
        except (ValueError, TypeError):
            logger.error("Invalid demand value for scenario '%s'. Skipping this scenario.", scenario_name)
            continue

        logger.info("Running simulation for scenario '%s' with demand: %.2f", scenario_name, demand_value)
        try:
            simulation_result: Dict[str, Any] = simulation_runner.run_simulation(demand_value)
        except Exception as e:
            logger.error("Simulation failed for scenario '%s': %s", scenario_name, e)
            continue

        # Print the simulation result details in a human-readable format.
        print("\nScenario: {}".format(scenario_name))
        optimal_config = simulation_result.get("optimal_configuration", {})
        print("Optimal Configuration: {}".format(optimal_config))
        total_power = simulation_result.get("total_power", float('nan'))
        print("Total Power Consumption: {:.2f} kW".format(total_power))
        total_flow = simulation_result.get("total_flow", float('nan'))
        print("Total Flow: {:.2f} Nm³".format(total_flow))
        header_pressure = simulation_result.get("header_pressure", float('nan'))
        print("Header Pressure: {:.2f} kPa".format(header_pressure))
        energy_savings_percent = simulation_result.get("energy_savings_percent", 0.0)
        print("Energy Savings Percent: {:.2f}%".format(energy_savings_percent))
        evaluation_count = simulation_result.get("evaluation_count", 0)
        print("Evaluations Count: {}".format(evaluation_count))
        runtime_seconds = simulation_result.get("runtime_seconds", 0.0)
        print("Runtime (s): {:.2f}".format(runtime_seconds))
        print("-" * 50)

if __name__ == "__main__":
    main()
