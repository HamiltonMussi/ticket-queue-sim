"""
Configuration parameters for simulation based on real system data
"""

# Normal scenario (baseline from real system)
NORMAL_SCENARIO = {
    # System capacity
    'num_workers': 200,

    # Arrival process
    'lambda_total': 0.0315,

    # Job distribution
    'pi_purchase': 0.350,
    'pi_expire': 0.650,
    'pi_refund': 0.000,
    'pi_chargeback': 0.000,

    # Shifted Exponential service times: time = shift + Exponential(rate)
    'shift_purchase': 0.134,
    'rate_purchase': 10.66,
    'shift_expire': 0.087,
    'rate_expire': 17.64,
    'shift_refund': 0.0,
    'rate_refund': 15.0,
    'shift_chargeback': 0.0,
    'rate_chargeback': 20.0,
}

# Extreme degradation with 200 workers (system overloaded but has capacity)
EXTREME_DEGRADATION_200W = {
    # System capacity
    'num_workers': 200,

    # Arrival process (20x normal traffic)
    'lambda_total': 0.63,

    # Job distribution (more purchases during peak)
    'pi_purchase': 0.350,
    'pi_expire': 0.650,
    'pi_refund': 0.000,
    'pi_chargeback': 0.000,

    # Extremely slow service times (50x degradation)
    'shift_purchase': 6.7,      # 6.7s minimum (DB timeouts)
    'rate_purchase': 0.21,      # High variability
    'shift_expire': 4.3,        # 4.3s minimum
    'rate_expire': 0.35,        # High variability
    'shift_refund': 0.0,
    'rate_refund': 15.0,
    'shift_chargeback': 0.0,
    'rate_chargeback': 20.0,
}

# Extreme degradation with 1 worker (bottleneck scenario)
EXTREME_DEGRADATION_1W = {
    # System capacity (severe bottleneck)
    'num_workers': 1,

    # Arrival process (same high traffic)
    'lambda_total': 0.63,

    # Job distribution
    'pi_purchase': 0.350,
    'pi_expire': 0.650,
    'pi_refund': 0.000,
    'pi_chargeback': 0.000,

    # Same degraded service times
    'shift_purchase': 6.7,
    'rate_purchase': 0.21,
    'shift_expire': 4.3,
    'rate_expire': 0.35,
    'shift_refund': 0.0,
    'rate_refund': 15.0,
    'shift_chargeback': 0.0,
    'rate_chargeback': 20.0,
}

# Default scenario (for backward compatibility)
SCENARIO = NORMAL_SCENARIO