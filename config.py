"""
Configuration parameters for simulation based on real system data
"""

SCENARIO = {
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