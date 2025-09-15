"""
Run ticket queue simulation experiments
"""
import random
import numpy as np
from simulator import TicketQueueSimulator
from config import NORMAL_SCENARIO, PEAK_SCENARIO
from models import JobType


def run_experiment(scenario, num_replications=30, simulation_time=3600):
    """Run multiple replications of a scenario"""
    print(f"Running {num_replications} replications...")
    
    results = []
    for i in range(num_replications):
        random.seed(42 + i)  # Different seed per replication
        
        sim = TicketQueueSimulator(scenario)
        metrics = sim.run(simulation_time)
        results.append(metrics)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_replications} replications")
    
    return results


def analyze_results(results, scenario_name):
    """Analyze and print results"""
    print(f"\n=== {scenario_name.upper()} SCENARIO RESULTS ===")
    
    # Main metrics
    W_values = [r.get_W() for r in results]
    X_values = [r.get_X() for r in results]
    rho_values = [r.get_rho() for r in results]
    L_values = [r.get_L() for r in results]
    timeout_rates = [r.get_timeout_rate() for r in results]
    
    print(f"W (avg response time): {np.mean(W_values):.2f} ± {np.std(W_values):.2f} seconds")
    print(f"X (throughput): {np.mean(X_values):.2f} ± {np.std(X_values):.2f} jobs/second")
    print(f"ρ (utilization): {np.mean(rho_values):.3f} ± {np.std(rho_values):.3f}")
    print(f"L (avg queue length): {np.mean(L_values):.1f} ± {np.std(L_values):.1f}")
    print(f"Timeout rate: {np.mean(timeout_rates):.2f}% ± {np.std(timeout_rates):.2f}%")
    
    # By job type
    print("\nBy Job Type:")
    for job_type in JobType:
        type_values = []
        for r in results:
            w_type = r.get_W_by_type(job_type)
            if w_type > 0:  # Only if jobs of this type were processed
                type_values.append(w_type)
        
        if type_values:
            print(f"  {job_type.value}: {np.mean(type_values):.2f} ± {np.std(type_values):.2f} seconds")
    
    return {
        'W': np.mean(W_values),
        'X': np.mean(X_values), 
        'rho': np.mean(rho_values),
        'timeout_rate': np.mean(timeout_rates)
    }


def compare_scenarios(normal_stats, peak_stats):
    """Compare scenarios"""
    print(f"\n=== SCENARIO COMPARISON ===")
    print(f"Degradation Factor (W_peak/W_normal): {peak_stats['W']/normal_stats['W']:.2f}")
    print(f"Utilization Growth (ρ_peak/ρ_normal): {peak_stats['rho']/normal_stats['rho']:.2f}")
    print(f"Timeout Impact: {peak_stats['timeout_rate'] - normal_stats['timeout_rate']:.2f}%")


if __name__ == "__main__":
    print("Ticket Queue Simulation")
    print("======================")
    
    # Run normal scenario
    print("\n1. Running NORMAL scenario...")
    normal_results = run_experiment(NORMAL_SCENARIO, num_replications=30)
    normal_stats = analyze_results(normal_results, "normal")
    
    # Run peak scenario  
    print("\n2. Running PEAK scenario...")
    peak_results = run_experiment(PEAK_SCENARIO, num_replications=30)
    peak_stats = analyze_results(peak_results, "peak")
    
    # Compare scenarios
    compare_scenarios(normal_stats, peak_stats)
    
    print("\nSimulation completed!")