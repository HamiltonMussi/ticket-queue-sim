"""
Configuration parameters for simulation scenarios
"""

# Scenario 1: Normal Operation
NORMAL_SCENARIO = {
    # Arrival rate (λ)
    'lambda_total': 1.0,  # jobs per second
    
    # Job distribution (π) - must sum to 1.0
    'pi_purchase': 0.60,
    'pi_expire': 0.25,
    'pi_refund': 0.12,
    'pi_chargeback': 0.03,
    
    # Service rates (μ) - jobs per second
    'mu_purchase': 0.1,     # 10s average
    'mu_expire': 0.5,       # 2s average  
    'mu_refund': 0.067,     # 15s average
    'mu_chargeback': 0.05,  # 20s average
}

# Scenario 2: Peak Load
PEAK_SCENARIO = {
    # Higher arrival rate
    'lambda_total': 10.0,  # 10x normal load
    
    # More purchases during peak
    'pi_purchase': 0.70,
    'pi_expire': 0.20,
    'pi_refund': 0.08,
    'pi_chargeback': 0.02,
    
    # Same service rates
    'mu_purchase': 0.1,
    'mu_expire': 0.5,
    'mu_refund': 0.067,
    'mu_chargeback': 0.05,
}