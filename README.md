# E-commerce ticket order queue simulation
This project develops a simulation for an e-commerce ticket order system that uses a queue. 

## Real System Architecture
- The queue in the original e-commerce system analyzed is implemented with the use of BullMQ and Redis. 
- There is a single queue to store all of the "Orders" jobs. 
- The possible jobs are: 
    - purchase
    - refund
    - chargeback
    - expire
- There are 200 workers
- Discipline: FIFO (first-in-first-out)

## System Modelling
- The random variables analyzed are:
    - Arrival Rate (the total number of jobs that arrive in a given time period)
    - Service Rate (average time that it takes for a job to be processed)
    - Job Distribution (percentage that each job type represents in the total)
- Using a small sample of 100 jobs from the real system, the following distributions could be obtained:
![Distribution validation](/distribution_validation.png)
- Due to the small sample size, the distribution classification doesn't show a very good fit, but it can be estimated that the Arrival Rate and Service Rate follow an exponential distribution.
- Also, the sample does not have any occurrences of refund and chargeback jobs. Therefore, these job types will be considered irrelevant for this study. As can be seen, 65% of the jobs were expire, and 35% were purchase.

## Performance Metrics
### Main Metrics
- **W** = average response time (seconds) - total time in system (waiting + processing)
- **X** = throughput (completed jobs per second)
- **ρ** = average worker utilization (0 to 1)
- **L** = average queue length

### Metrics by Job Type
- **W_i** = average response time for job type i (purchase, expire, refund, chargeback)
- **X_i** = throughput for job type i (completed jobs per second)

### Variability Metrics
- **σ_W** = standard deviation of response times
- **P95_W** = 95th percentile of response times
- **P99_W** = 99th percentile of response times  
- **CV_W** = coefficient of variation (σ_W / W)

### Comparative Metrics (between scenarios)
- **Degradation Factor** = W_peak / W_normal
- **Utilization Growth** = ρ_peak / ρ_normal
- **Timeout Impact** = Timeout_Rate_peak - Timeout_Rate_normal

## Simulation Model
### Model Classification
- **M/G/200**: Markovian arrivals, General service times, 200 servers
- **Single queue**: all job types share the same FIFO queue
- **Mixed job types**: 4 different job classes with different service rates

### Arrival Process
- **Job arrivals**: Poisson process with rate λ (jobs per second)
- **Inter-arrival times**: Exponentially distributed with parameter λ
- **Job type selection**: each arriving job is assigned a type using Multinomial distribution with probabilities [π_purchase, π_expire, π_refund, π_chargeback]

### Service Process
- **Service times**: Shifted Exponential distribution per job type
    - purchase jobs: shift + Exponential(rate_purchase)
    - expire jobs: shift + Exponential(rate_expire)
    - refund jobs: shift + Exponential(rate_refund)
    - chargeback jobs: shift + Exponential(rate_chargeback)
- **Workers**: Configurable number of identical servers processing jobs in parallel
- **Queue discipline**: FIFO (first-in-first-out)

### System Components
- **Job Generator**: creates jobs according to Poisson(λ) and assigns types using π distribution
- **Queue**: unlimited capacity FIFO buffer storing waiting jobs
- **Worker Pool**: configurable number of parallel workers that process jobs according to their type-specific service rates
- **Metrics Collector**: continuously collects performance data during simulation

### Data Collection
- **Job-level data**: arrival time, start time, finish time, job type, timeout status
- **System-level samples**: queue length and worker utilization collected every 10 seconds
- **Aggregated metrics**: calculated at the end of each simulation run

## Simulation Scenarios

### 1. Normal Scenario (Baseline)
- **Workers**: 200
- **Arrival rate**: 0.0315 jobs/s (real system data)
- **Service times**: Real system distribution (0.17s average)
  - Purchase: 0.134s + Exp(10.66) = 0.228s average
  - Expire: 0.087s + Exp(17.64) = 0.144s average
- **Purpose**: Baseline performance under normal conditions
- **Expected results**: Very low utilization, minimal response times

### 2. Extreme Degradation - 200 Workers
- **Workers**: 200 (same capacity)
- **Arrival rate**: 0.63 jobs/s (20x normal traffic)
- **Service times**: ~50x slower (8.7s average, simulating DB overload)
  - Purchase: 6.7s + Exp(0.21) = 11.5s average
  - Expire: 4.3s + Exp(0.35) = 7.2s average
- **Purpose**: Test system resilience under severe performance degradation
- **Expected results**: High response times but stable throughput

### 3. Extreme Degradation - 1 Worker
- **Workers**: 1 (severe capacity constraint)
- **Arrival rate**: 0.63 jobs/s (same high traffic)
- **Service times**: ~50x slower (same degradation)
  - Purchase: 6.7s + Exp(0.21) = 11.5s average
  - Expire: 4.3s + Exp(0.35) = 7.2s average
- **Purpose**: Demonstrate system collapse under capacity bottleneck
- **Expected results**: System instability, infinite queue growth, very high response times

## Results

### Performance Comparison

| Scenario | Workers | Arrival Rate | W (Response Time) | X (Throughput) | ρ (Utilization) | L (Queue Length) |
|----------|---------|--------------|-------------------|----------------|-----------------|------------------|
| Normal | 200 | 0.032 jobs/s | 0.17s | 0.031 jobs/s | 0.000 | 0.0 |
| Extreme (200w) | 200 | 0.630 jobs/s | 8.63s | 0.627 jobs/s | 0.027 | 0.0 |
| Extreme (1w) | 1 | 0.630 jobs/s | 1483s | 0.120 jobs/s | 0.999 | 925.9 |

### Response Time by Job Type

| Scenario | Purchase Time | Expire Time | Weighted Average |
|----------|---------------|-------------|------------------|
| Normal | 0.23s | 0.14s | 0.17s |
| Extreme (200w) | 11.42s | 7.15s | 8.63s |
| Extreme (1w) | 1467s | 1491s | 1483s |

### Degradation Analysis

| Metric | Normal to Extreme (200w) | Normal to Extreme (1w) | Extreme (200w) to Extreme (1w) |
|--------|-------------------------|----------------------|------------------------------|
| Response Time Factor | 50.8x slower | 8724x slower | 171.9x slower |
| Throughput Factor | 20.2x higher | 3.9x higher | 0.19x (collapse) |
| Utilization Factor | - | - | 37.0x higher |

### Key Insights

1. **Performance vs. Capacity**: 50x service degradation is manageable with adequate capacity (200 workers)
2. **Capacity Bottleneck**: Same degradation with insufficient capacity (1 worker) causes complete system collapse
3. **Queue Formation**: High capacity prevents queue buildup even under extreme degradation
4. **Utilization**: System can operate efficiently at low utilization but fails when saturated with only 1 worker
5. **Throughput**: Adequate capacity maintains target throughput despite performance degradation

