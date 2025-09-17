"""
Configuration parameters for simulation scenarios
"""

# Real System Parameters (extracted from validation_data.csv)
REAL_SYSTEM_SCENARIO = {
    # Arrival rate (λ) - based on observed data
    'lambda_total': 0.0315,  # jobs per second

    # Job distribution (π) - observed proportions
    'pi_purchase': 0.350,
    'pi_expire': 0.650,
    'pi_refund': 0.000,     # Not observed in sample
    'pi_chargeback': 0.000, # Not observed in sample

    # Service rates (μ) - estimated from observed service times
    'mu_purchase': 4.3904,  # 0.23s average
    'mu_expire': 6.9593,    # 0.14s average
    'mu_refund': 0.067,     # Fallback value (not observed)
    'mu_chargeback': 0.05,  # Fallback value (not observed)
}

# Scenario 1: Normal Operation (based on real system)
NORMAL_SCENARIO = {
    # Arrival rate (λ)
    'lambda_total': 0.0315,  # jobs per second (real system rate)

    # Job distribution (π) - real system proportions
    'pi_purchase': 0.350,
    'pi_expire': 0.650,
    'pi_refund': 0.000,
    'pi_chargeback': 0.000,

    # Service rates (μ) - real system rates
    'mu_purchase': 4.3904,  # 0.23s average
    'mu_expire': 6.9593,    # 0.14s average
    'mu_refund': 0.067,     # Not used
    'mu_chargeback': 0.05,  # Not used
}

# Scenario 2: Peak Load (stress test scenario)
PEAK_SCENARIO = {
    # Higher arrival rate (10x normal load)
    'lambda_total': 0.315,  # 10x real system rate

    # Same job distribution as real system
    'pi_purchase': 0.350,
    'pi_expire': 0.650,
    'pi_refund': 0.000,
    'pi_chargeback': 0.000,

    # Same service rates as real system
    'mu_purchase': 4.3904,  # 0.23s average
    'mu_expire': 6.9593,    # 0.14s average
    'mu_refund': 0.067,     # Not used
    'mu_chargeback': 0.05,  # Not used
}