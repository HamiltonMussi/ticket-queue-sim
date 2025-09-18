"""
Analyze real system data to extract parameters and validate distributions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def load_and_clean_data():
    """Load and preprocess the validation data"""
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / 'validation_data.csv')
    df.columns = df.columns.str.strip()

    # Convert timestamps from milliseconds to seconds
    df['createdAt'] = df['createdAt'] / 1000
    df['startedAt'] = df['startedAt'] / 1000
    df['finishedAt'] = df['finishedAt'] / 1000
    df['processedTime'] = df['processedTime'] / 1000

    # Calculate inter-arrival times
    df = df.sort_values('createdAt')
    df['inter_arrival'] = df['createdAt'].diff()

    return df

def analyze_arrival_process(df):
    """Analyze arrival process and estimate λ"""
    total_time = df['createdAt'].max() - df['createdAt'].min()
    total_jobs = len(df)
    lambda_estimate = total_jobs / total_time

    print(f"Arrival Analysis:")
    print(f"  λ (arrival rate): {lambda_estimate:.4f} jobs/second")
    print(f"  Total jobs: {total_jobs}")

    return lambda_estimate

def analyze_job_distribution(df):
    """Analyze job type distribution"""
    job_proportions = df['jobName'].value_counts(normalize=True)

    print(f"\nJob Distribution:")
    for job_type, proportion in job_proportions.items():
        print(f"  {job_type}: {proportion:.3f}")

    return job_proportions

def analyze_service_times(df):
    """Analyze service times and estimate parameters"""
    service_rates = {}

    print(f"\nService Time Analysis:")
    for job_type in df['jobName'].unique():
        job_data = df[df['jobName'] == job_type]
        service_times = job_data['processedTime']

        # Shifted exponential parameters
        min_time = service_times.min()
        shifted_times = service_times - min_time
        scale_shifted = shifted_times.mean()
        lambda_shifted = 1 / scale_shifted

        service_rates[job_type] = {
            'shift': min_time,
            'lambda': lambda_shifted,
            'mu_original': 1 / service_times.mean()  # For compatibility
        }

        print(f"  {job_type}: shift={min_time:.3f}s, λ={lambda_shifted:.2f}")

    return service_rates

def create_validation_plots(df, lambda_est, service_rates):
    """Create clean validation plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution Validation Plots', fontsize=16)

    # 1. Inter-arrival times
    inter_arrivals = df['inter_arrival'].dropna()
    axes[0,0].hist(inter_arrivals, bins=20, density=True, alpha=0.7, label='Observed')
    x = np.linspace(0, inter_arrivals.max(), 100)
    axes[0,0].plot(x, stats.expon.pdf(x, scale=1/lambda_est), 'r-',
                   label=f'Exponential (λ={lambda_est:.4f})')
    axes[0,0].set_title('Inter-arrival Times')
    axes[0,0].set_xlabel('Time (seconds)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()

    # 2. Job type distribution
    job_counts = df['jobName'].value_counts()
    bars = axes[0,1].bar(job_counts.index, job_counts.values)

    # Add value labels on top of bars
    for bar, value in zip(bars, job_counts.values):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{value}', ha='center', va='bottom', fontweight='bold')

    # Adjust y-axis to accommodate labels
    axes[0,1].set_ylim(0, max(job_counts.values) * 1.1)

    axes[0,1].set_title('Job Type Distribution')
    axes[0,1].set_xlabel('Job Type')
    axes[0,1].set_ylabel('Count')

    # 3. Overall service times
    overall_service = df['processedTime']
    axes[0,2].hist(overall_service, bins=20, density=True, alpha=0.7, label='Observed')

    overall_min = overall_service.min()
    overall_shifted = overall_service - overall_min
    overall_scale_shifted = overall_shifted.mean()
    overall_lambda_shifted = 1 / overall_scale_shifted

    x = np.linspace(overall_service.min(), overall_service.max(), 100)
    axes[0,2].plot(x, stats.expon.pdf(x - overall_min, scale=overall_scale_shifted), 'r-',
                   label=f'Exponential (λ={overall_lambda_shifted:.2f}, shift={overall_min:.3f}s)')

    axes[0,2].set_title('Overall Service Times')
    axes[0,2].set_xlabel('Time (seconds)')
    axes[0,2].set_ylabel('Density')
    axes[0,2].legend()

    # 4-5. Service times by job type
    job_types = df['jobName'].unique()
    colors = ['blue', 'green']
    positions = [(1,0), (1,1)]

    for job_type, color, pos in zip(job_types[:2], colors, positions):
        job_data = df[df['jobName'] == job_type]['processedTime']
        axes[pos].hist(job_data, bins=15, density=True, alpha=0.7, color=color, label='Observed')

        # Get parameters from service_rates
        shift = service_rates[job_type]['shift']
        lambda_val = service_rates[job_type]['lambda']
        scale = 1 / lambda_val

        x = np.linspace(job_data.min(), job_data.max(), 100)
        axes[pos].plot(x, stats.expon.pdf(x - shift, scale=scale), 'r-',
                      label=f'Exponential (λ={lambda_val:.2f}, shift={shift:.3f}s)')

        axes[pos].set_title(f'{job_type} Service Times')
        axes[pos].set_xlabel('Time (seconds)')
        axes[pos].set_ylabel('Density')
        axes[pos].legend()

    # Hide empty subplot
    axes[1,2].set_visible(False)

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / 'distribution_validation.png', dpi=300, bbox_inches='tight')
    print(f"\nPlots saved as 'outputs/distribution_validation.png'")

def generate_config_parameters(lambda_est, job_proportions, service_rates):
    """Generate clean configuration parameters"""
    print(f"\n" + "="*50)
    print(f"EXTRACTED PARAMETERS FOR CONFIG.PY")
    print(f"="*50)

    print(f"REAL_SYSTEM_SCENARIO = {{")
    print(f"    'lambda_total': {lambda_est:.4f},")
    print(f"")

    for job_type, proportion in job_proportions.items():
        print(f"    'pi_{job_type}': {proportion:.3f},")

    print(f"")
    for job_type, params in service_rates.items():
        mu_val = params['mu_original']
        print(f"    'mu_{job_type}': {mu_val:.4f},  # {1/mu_val:.3f}s average")

    print(f"}}")

def main():
    """Main analysis function"""
    print("Real System Data Analysis")
    print("="*25)

    # Load and analyze data
    df = load_and_clean_data()
    lambda_est = analyze_arrival_process(df)
    job_proportions = analyze_job_distribution(df)
    service_rates = analyze_service_times(df)

    # Generate outputs
    create_validation_plots(df, lambda_est, service_rates)
    generate_config_parameters(lambda_est, job_proportions, service_rates)

if __name__ == "__main__":
    main()
