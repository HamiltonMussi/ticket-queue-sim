"""
Analyze real system data to extract parameters and validate distributions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime

def load_and_clean_data():
    """Load and preprocess the validation data"""
    df = pd.read_csv('validation_data.csv')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert timestamps from milliseconds to seconds
    df['createdAt'] = df['createdAt'] / 1000
    df['startedAt'] = df['startedAt'] / 1000
    df['finishedAt'] = df['finishedAt'] / 1000
    df['processedTime'] = df['processedTime'] / 1000  # Convert to seconds

    # Calculate inter-arrival times
    df = df.sort_values('createdAt')
    df['inter_arrival'] = df['createdAt'].diff()

    # Calculate waiting time (time in queue)
    df['waiting_time'] = df['startedAt'] - df['createdAt']

    # Calculate response time (total time in system)
    df['response_time'] = df['finishedAt'] - df['createdAt']

    return df

def analyze_arrival_process(df):
    """Analyze arrival process and estimate λ"""
    print("=== ARRIVAL PROCESS ANALYSIS ===")

    # Remove NaN inter-arrival times (first job)
    inter_arrivals = df['inter_arrival'].dropna()

    # Calculate arrival rate
    total_time = df['createdAt'].max() - df['createdAt'].min()
    total_jobs = len(df)
    lambda_estimate = total_jobs / total_time

    print(f"Total observation time: {total_time:.2f} seconds")
    print(f"Total jobs: {total_jobs}")
    print(f"Estimated λ (arrival rate): {lambda_estimate:.4f} jobs/second")
    print(f"Average inter-arrival time: {inter_arrivals.mean():.2f} seconds")

    # Test if inter-arrivals follow exponential distribution
    print(f"\nExponential distribution test:")
    print(f"Theoretical mean (1/λ): {1/lambda_estimate:.2f} seconds")
    print(f"Observed mean: {inter_arrivals.mean():.2f} seconds")

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(inter_arrivals, lambda x: stats.expon.cdf(x, scale=1/lambda_estimate))
    print(f"KS test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
    print(f"Exponential fit: {'GOOD' if ks_p > 0.05 else 'POOR'}")

    return lambda_estimate, inter_arrivals

def analyze_job_distribution(df):
    """Analyze job type distribution and estimate π"""
    print(f"\n=== JOB TYPE DISTRIBUTION ===")

    job_counts = df['jobName'].value_counts()
    job_proportions = df['jobName'].value_counts(normalize=True)

    print("Job type distribution:")
    for job_type in job_counts.index:
        count = job_counts[job_type]
        proportion = job_proportions[job_type]
        print(f"  {job_type}: {count} jobs ({proportion:.3f})")

    return job_proportions

def analyze_service_times(df):
    """Analyze service times by job type and estimate μ"""
    print(f"\n=== SERVICE TIME ANALYSIS ===")

    service_rates = {}

    for job_type in df['jobName'].unique():
        job_data = df[df['jobName'] == job_type]
        service_times = job_data['processedTime']

        print(f"\n{job_type.upper()} jobs:")
        print(f"  Count: {len(service_times)}")
        print(f"  Mean service time: {service_times.mean():.3f} seconds")
        print(f"  Std deviation: {service_times.std():.3f} seconds")
        print(f"  Min: {service_times.min():.3f}s, Max: {service_times.max():.3f}s")
        print(f"  Median: {service_times.median():.3f}s")
        print(f"  Coefficient of variation: {service_times.std()/service_times.mean():.3f}")

        # Estimate service rate (μ)
        mu_estimate = 1 / service_times.mean()
        service_rates[job_type] = mu_estimate
        print(f"  Estimated μ: {mu_estimate:.4f} jobs/second")

        # Test exponential distribution
        ks_stat, ks_p = stats.kstest(service_times, lambda x: stats.expon.cdf(x, scale=service_times.mean()))
        print(f"  Exponential fit: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
        print(f"  Distribution fit: {'GOOD' if ks_p > 0.05 else 'POOR'}")

        # Test normal distribution
        ks_stat_norm, ks_p_norm = stats.kstest(service_times,
                                              lambda x: stats.norm.cdf(x, loc=service_times.mean(), scale=service_times.std()))
        print(f"  Normal fit: statistic={ks_stat_norm:.4f}, p-value={ks_p_norm:.4f}")
        print(f"  Normal fit: {'GOOD' if ks_p_norm > 0.05 else 'POOR'}")

        # Test gamma distribution
        try:
            alpha, loc, beta = stats.gamma.fit(service_times, floc=0)
            ks_stat_gamma, ks_p_gamma = stats.kstest(service_times,
                                                    lambda x: stats.gamma.cdf(x, alpha, scale=beta))
            print(f"  Gamma fit: α={alpha:.2f}, β={beta:.3f}, p-value={ks_p_gamma:.4f}")
            print(f"  Gamma fit: {'GOOD' if ks_p_gamma > 0.05 else 'POOR'}")
        except:
            print("  Gamma fit: Failed")

        # Test shifted exponential (exponential + constant)
        min_time = service_times.min()
        shifted_times = service_times - min_time
        if shifted_times.max() > 0:
            ks_stat_shifted, ks_p_shifted = stats.kstest(shifted_times,
                                                        lambda x: stats.expon.cdf(x, scale=shifted_times.mean()))
            print(f"  Shifted Exponential fit: shift={min_time:.3f}s, p-value={ks_p_shifted:.4f}")
            print(f"  Shifted Exponential fit: {'GOOD' if ks_p_shifted > 0.05 else 'POOR'}")

        # Test exponential with location parameter
        try:
            loc_exp, scale_exp = stats.expon.fit(service_times)
            ks_stat_loc_exp, ks_p_loc_exp = stats.kstest(service_times,
                                                         lambda x: stats.expon.cdf(x, loc=loc_exp, scale=scale_exp))
            print(f"  Exponential(loc) fit: loc={loc_exp:.3f}, scale={scale_exp:.3f}, p-value={ks_p_loc_exp:.4f}")
            print(f"  Exponential(loc) fit: {'GOOD' if ks_p_loc_exp > 0.05 else 'POOR'}")
        except:
            print("  Exponential(loc) fit: Failed")

    return service_rates

def create_validation_plots(df, lambda_est, service_rates):
    """Create plots to validate distributions"""
    print(f"\n=== CREATING VALIDATION PLOTS ===")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution Validation Plots', fontsize=16)

    # 1. Inter-arrival times
    inter_arrivals = df['inter_arrival'].dropna()
    axes[0,0].hist(inter_arrivals, bins=20, density=True, alpha=0.7, label='Observed')
    x = np.linspace(0, inter_arrivals.max(), 100)
    axes[0,0].plot(x, stats.expon.pdf(x, scale=1/lambda_est), 'r-', label=f'Exponential (λ={lambda_est:.4f})')
    axes[0,0].set_title('Inter-arrival Times')
    axes[0,0].set_xlabel('Time (seconds)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()

    # 2. Job type distribution
    job_counts = df['jobName'].value_counts()
    axes[0,1].bar(job_counts.index, job_counts.values)
    axes[0,1].set_title('Job Type Distribution')
    axes[0,1].set_xlabel('Job Type')
    axes[0,1].set_ylabel('Count')

    # 3. Overall service times
    overall_service = df['processedTime']
    axes[0,2].hist(overall_service, bins=20, density=True, alpha=0.7, label='Observed')

    # Test shifted exponential for overall
    overall_min = overall_service.min()
    overall_shifted = overall_service - overall_min
    overall_scale_shifted = overall_shifted.mean()

    x = np.linspace(overall_service.min(), overall_service.max(), 100)

    # Shifted exponential (good fit)
    overall_lambda_shifted = 1 / overall_scale_shifted
    axes[0,2].plot(x, stats.expon.pdf(x - overall_min, scale=overall_scale_shifted), 'r-',
                   label=f'Exponential (λ={overall_lambda_shifted:.2f}, shift={overall_min:.3f}s)')

    # Test overall shifted exponential fit
    ks_stat_shifted, ks_p_shifted = stats.kstest(overall_shifted,
                                                 lambda x: stats.expon.cdf(x, scale=overall_scale_shifted))

    axes[0,2].set_xlabel('Time (seconds)')
    axes[0,2].set_ylabel('Density')
    axes[0,2].legend()

    # 4-5. Service times by job type (expire and purchase separately)
    job_types = df['jobName'].unique()
    colors = ['blue', 'green']
    positions = [(1,0), (1,1)]  # Bottom row

    for i, (job_type, color, pos) in enumerate(zip(job_types[:2], colors, positions)):
        job_data = df[df['jobName'] == job_type]['processedTime']
        axes[pos].hist(job_data, bins=15, density=True, alpha=0.7, color=color, label='Observed')

        # Plot theoretical shifted exponential
        min_time = job_data.min()
        shifted_data = job_data - min_time
        scale_shifted = shifted_data.mean()

        x = np.linspace(job_data.min(), job_data.max(), 100)

        # Shifted exponential (good fit)
        lambda_shifted = 1 / scale_shifted
        axes[pos].plot(x, stats.expon.pdf(x - min_time, scale=scale_shifted), 'r-',
                      label=f'Exponential (λ={lambda_shifted:.2f}, shift={min_time:.3f}s)')

        axes[pos].set_title(f'{job_type} Service Times')
        axes[pos].set_xlabel('Time (seconds)')
        axes[pos].set_ylabel('Density')
        axes[pos].legend()

    # Hide the empty subplot (bottom-right would be [1,2])
    axes[1,2].set_visible(False)

    plt.tight_layout()
    plt.savefig('distribution_validation.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'distribution_validation.png'")

def generate_config_parameters(lambda_est, job_proportions, service_rates):
    """Generate updated configuration parameters"""
    print(f"\n=== SUGGESTED CONFIG PARAMETERS ===")

    print(f"# Based on real system data analysis")
    print(f"REAL_SYSTEM_SCENARIO = {{")
    print(f"    'lambda_total': {lambda_est:.4f},  # jobs per second")
    print(f"")
    print(f"    # Job distribution (π)")

    for job_type, proportion in job_proportions.items():
        print(f"    'pi_{job_type}': {proportion:.3f},")

    print(f"")
    print(f"    # Service rates (μ) - jobs per second")

    for job_type, mu in service_rates.items():
        print(f"    'mu_{job_type}': {mu:.4f},  # {1/mu:.1f}s average")

    print(f"}}")

def main():
    """Main analysis function"""
    print("Real System Data Analysis")
    print("========================")

    # Load data
    df = load_and_clean_data()
    print(f"Loaded {len(df)} jobs from validation data")

    # Analyze arrival process
    lambda_est, inter_arrivals = analyze_arrival_process(df)

    # Analyze job distribution
    job_proportions = analyze_job_distribution(df)

    # Analyze service times
    service_rates = analyze_service_times(df)

    # Create validation plots
    create_validation_plots(df, lambda_est, service_rates)

    # Generate config parameters
    generate_config_parameters(lambda_est, job_proportions, service_rates)

    return df, lambda_est, job_proportions, service_rates

if __name__ == "__main__":
    df, lambda_est, job_proportions, service_rates = main()