"""
Run ticket queue simulation experiments
"""
import random
import numpy as np
from scipy import stats
from simulator import TicketQueueSimulator
from config import SCENARIO
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


def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval using t-distribution"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def analyze_results(results, scenario_name):
    """Analyze and print results with confidence intervals"""
    print(f"\n=== {scenario_name.upper()} SCENARIO RESULTS ===")

    # Main metrics
    W_values = [r.get_W() for r in results]
    X_values = [r.get_X() for r in results]
    rho_values = [r.get_rho() for r in results]
    L_values = [r.get_L() for r in results]
    timeout_rates = [r.get_timeout_rate() for r in results]

    # Calculate confidence intervals
    W_ci = confidence_interval(W_values)
    X_ci = confidence_interval(X_values)
    rho_ci = confidence_interval(rho_values)
    L_ci = confidence_interval(L_values)
    timeout_ci = confidence_interval(timeout_rates)

    print(f"W (avg response time): {np.mean(W_values):.2f} ± {np.std(W_values):.2f} seconds")
    print(f"  95% CI: [{W_ci[0]:.2f}, {W_ci[1]:.2f}]")
    print(f"X (throughput): {np.mean(X_values):.2f} ± {np.std(X_values):.2f} jobs/second")
    print(f"  95% CI: [{X_ci[0]:.2f}, {X_ci[1]:.2f}]")
    print(f"ρ (utilization): {np.mean(rho_values):.3f} ± {np.std(rho_values):.3f}")
    print(f"  95% CI: [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")
    print(f"L (avg queue length): {np.mean(L_values):.1f} ± {np.std(L_values):.1f}")
    print(f"  95% CI: [{L_ci[0]:.1f}, {L_ci[1]:.1f}]")
    print(f"Timeout rate: {np.mean(timeout_rates):.2f}% ± {np.std(timeout_rates):.2f}%")
    print(f"  95% CI: [{timeout_ci[0]:.2f}%, {timeout_ci[1]:.2f}%]")

    # By job type with confidence intervals
    print("\nBy Job Type:")
    for job_type in JobType:
        type_values = []
        for r in results:
            w_type = r.get_W_by_type(job_type)
            if w_type > 0:  # Only if jobs of this type were processed
                type_values.append(w_type)

        if type_values and len(type_values) > 1:
            type_ci = confidence_interval(type_values)
            print(f"  {job_type.value}: {np.mean(type_values):.2f} ± {np.std(type_values):.2f} seconds")
            print(f"    95% CI: [{type_ci[0]:.2f}, {type_ci[1]:.2f}]")

    return {
        'W': np.mean(W_values),
        'X': np.mean(X_values),
        'rho': np.mean(rho_values),
        'timeout_rate': np.mean(timeout_rates),
        'W_ci': W_ci,
        'X_ci': X_ci,
        'rho_ci': rho_ci,
        'timeout_ci': timeout_ci
    }



if __name__ == "__main__":
    print("Ticket Queue Simulation")
    print("======================")

    # Run simulation with real system parameters
    print("\nRunning simulation with real system parameters...")
    results = run_experiment(SCENARIO, num_replications=30)
    analyze_results(results, "Real System")

    print("\nSimulation completed!")