"""
Configuration parameters for simulation based on real system data
"""

# Real System Parameters (extracted from validation_data.csv)
SCENARIO = {
    # Arrival rate (λ) - based on observed data
    'lambda_total': 0.0315,  # jobs per second

    # Job distribution (π) - observed proportions
    'pi_purchase': 0.350,
    'pi_expire': 0.650,
    'pi_refund': 0.000,     # Not observed in sample
    'pi_chargeback': 0.000, # Not observed in sample

    # Service rates (μ) - estimated from observed service times
    'mu_purchase': 4.3904,  # 0.228s average
    'mu_expire': 6.9593,    # 0.144s average
    'mu_refund': 0.067,     # Fallback value (not observed)
    'mu_chargeback': 0.05,  # Fallback value (not observed)
}