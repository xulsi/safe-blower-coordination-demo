"""
simulation.py

This module defines the SimulationRunner class, which integrates the BlowerModel,
NetworkModel, and an Optimizer instance (either GridSearchOptimizer or SBOOptimizer)
to run simulations based on a given air demand scenario. The runner measures runtime,
collects evaluation metrics, and returns the optimal configuration along with performance
metrics such as energy savings percentage and number of candidate evaluations.

Classes:
    SimulationRunner
"""

import time
import logging
import sys
from typing import Any, Dict

# Import configuration loader and model/optimizer classes from other modules.
from utils import load_config
from blower_model import BlowerModel
from network_model import NetworkModel
from optimizer import GridSearchOptimizer, SBOOptimizer, Optimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SimulationRunner:
    """
    The SimulationRunner class integrates the blower, network, and optimizer modules.
    
    It provides a run_simulation method that, for a given air demand value,
    invokes the optimizer routine, measures runtime, and returns a consolidated
    dictionary of simulation results.
    """
    def __init__(self, blower: BlowerModel, network: NetworkModel, optimizer: Optimizer, config: Dict[str, Any]) -> None:
        """
        Initialize the SimulationRunner with instances of BlowerModel, NetworkModel, 
        and Optimizer, as well as a global configuration dictionary.
        
        Parameters:
            blower (BlowerModel): An instance of the blower performance model.
            network (NetworkModel): An instance of the air distribution network model.
            optimizer (Optimizer): A concrete optimizer instance (e.g., GridSearchOptimizer or SBOOptimizer).
            config (Dict[str, Any]): Global configuration parameters (loaded from config.yaml).
        """
        if blower is None:
            raise ValueError("A valid BlowerModel instance is required.")
        if network is None:
            raise ValueError("A valid NetworkModel instance is required.")
        if optimizer is None:
            raise ValueError("A valid Optimizer instance is required.")
        if not isinstance(config, dict):
            raise ValueError("Configuration must be provided as a dictionary.")
        
        self.blower: BlowerModel = blower
        self.network: NetworkModel = network
        self.optimizer: Optimizer = optimizer
        self.config: Dict[str, Any] = config
        
        logger.info("SimulationRunner initialized with provided models and configuration.")

    def run_simulation(self, demand: float) -> Dict[str, Any]:
        """
        Run the simulation for the specified air demand.
        
        The method measures the runtime, calls the optimizer's optimize() method
        (which tests candidate configurations via the blower and network models),
        and aggregates simulation metrics such as optimal configuration, energy savings,
        evaluation count, and total runtime.
        
        Parameters:
            demand (float): The total required air demand (e.g., from a demand scenario).
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - "optimal_configuration": dict with keys "on_off" and "speeds"
                - "total_power": float, simulated optimal power consumption (kW)
                - "total_flow": float, aggregated blower flow (Nm³)
                - "header_pressure": float, computed header pressure (kPa)
                - "objective": float, the minimized objective function value
                - "energy_savings_percent": float, percent energy saving compared to baseline
                - "evaluation_count": int, total candidates evaluated during optimization
                - "runtime_seconds": float, simulation runtime in seconds
        """
        if not isinstance(demand, (int, float)):
            raise TypeError("Demand must be a numeric value.")
        
        logger.info("Starting simulation with demand: %.2f", demand)
        start_time: float = time.time()
        
        # Run optimizer's routine.
        result: Dict[str, Any] = self.optimizer.optimize(self.blower, self.network, demand, self.config)
        
        runtime_seconds: float = time.time() - start_time
        result["runtime_seconds"] = runtime_seconds
        
        logger.info("Simulation completed in %.2f seconds.", runtime_seconds)
        return result


if __name__ == "__main__":
    # Load configuration from the config.yaml file.
    config_filepath: str = "config.yaml"
    config: Dict[str, Any] = load_config(config_filepath)
    
    # Instantiate the BlowerModel using the blower_model configuration.
    blower_config: Dict[str, Any] = config.get("blower_model", {})
    blower_model = BlowerModel(blower_config)
    
    # Instantiate the NetworkModel using the network_model configuration.
    network_config: Dict[str, Any] = config.get("network_model", {})
    network_model = NetworkModel(network_config)
    
    # Determine number of blowers; default to 3 if not explicitly defined.
    num_blowers: int = int(config.get("num_blowers", 3))
    config["num_blowers"] = num_blowers  # Ensure this value is stored in the configuration.
    
    # Select the optimizer based on command-line arguments (default is GridSearchOptimizer).
    optimizer_choice: str = "grid"
    if len(sys.argv) > 1:
        optimizer_choice = sys.argv[1].lower()
    
    penalty_params: Dict[str, float] = config.get("optimizer", {}).get("penalty", {"lambda_sw": 0.0, "lambda_ramp": 0.0})
    optimizer: Optimizer
    if optimizer_choice == "sbo":
        # Use SBOOptimizer with parameters from config (fallback defaults provided).
        sbo_params: Dict[str, Any] = config.get("optimizer", {}).get("sbo", {"init_samples": 5, "early_stopping": 1e-3, "gp_kernel": "RBF", "acquisition": "Expected Improvement"})
        optimizer = SBOOptimizer(sbo_params.get("init_samples", 5), sbo_params, penalty_params)
        logger.info("Using SBOOptimizer for simulation.")
    else:
        # Default to GridSearchOptimizer with speed grid from configuration.
        grid_search_config: Dict[str, Any] = config.get("optimizer", {}).get("grid_search", {"speed_grid": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]})
        speed_grid = grid_search_config.get("speed_grid", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        optimizer = GridSearchOptimizer(speed_grid, penalty_params)
        logger.info("Using GridSearchOptimizer for simulation.")
    
    # Instantiate the SimulationRunner with the created model instances.
    simulation_runner = SimulationRunner(blower_model, network_model, optimizer, config)
    
    # Retrieve demand scenarios from the simulation configuration.
    simulation_config: Dict[str, Any] = config.get("simulation", {})
    demand_scenarios = simulation_config.get("demand_scenarios", [])
    
    # If no scenarios defined, use a default demand scenario.
    if not demand_scenarios:
        default_demand: float = 1000.0  # Representative default demand if none provided.
        demand_scenarios = [{"scenario": "default", "demand": default_demand}]
    
    # Run simulation for each demand scenario and print the results.
    for scenario in demand_scenarios:
        scenario_name: str = scenario.get("scenario", "Unnamed Scenario")
        try:
            scenario_demand: float = float(scenario.get("demand", 1000.0))
        except (ValueError, TypeError):
            logger.error("Invalid demand value for scenario '%s'. Skipping this scenario.", scenario_name)
            continue
        
        logger.info("Running simulation for scenario: %s with demand: %.2f", scenario_name, scenario_demand)
        simulation_result: Dict[str, Any] = simulation_runner.run_simulation(scenario_demand)
        
        # Output simulation result details.
        print("\nScenario: {}".format(scenario_name))
        optimal_config: Dict[str, Any] = simulation_result.get("optimal_configuration", {})
        print("Optimal Configuration: {}".format(optimal_config))
        print("Total Power Consumption: {:.2f} kW".format(simulation_result.get("total_power", float('nan'))))
        print("Total Flow: {:.2f} Nm³".format(simulation_result.get("total_flow", float('nan'))))
        print("Header Pressure: {:.2f} kPa".format(simulation_result.get("header_pressure", float('nan'))))
        print("Energy Savings Percent: {:.2f}%".format(simulation_result.get("energy_savings_percent", 0.0)))
        print("Evaluations Count: {}".format(simulation_result.get("evaluation_count", 0)))
        print("Runtime (s): {:.2f}".format(simulation_result.get("runtime_seconds", 0.0)))
        print("-" * 50)
