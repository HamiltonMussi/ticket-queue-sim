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
- Timeout: the default is 160s and the max is 600s
- Retry: exponential backoff (base 1s)

## Defined Parameters
### Arrival Rate (λ)
- it is the total amount of jobs that arrive in a determined amount of time
### Service Rate (μ)
- average time that it takes for a job to be processed
- μ_purchase, μ_expire, μ_refund, μ_chargeback
### Jobs Distribution (π)
- the percentage that each possible job corresponds in the total
- π_purchase, π_expire, π_refund, π_chargeback

## Performance Metrics
### Main Metrics
- **W** = average response time (seconds) - total time in system (waiting + processing)
- **X** = throughput (completed jobs per second)
- **ρ** = average worker utilization (0 to 1)
- **L** = average queue length

### Metrics by Job Type
- **W_i** = average response time for job type i (purchase, expire, refund, chargeback)
- **X_i** = throughput for job type i (completed jobs per second)

### Reliability Metrics
- **Timeout Rate** = percentage of jobs that exceeded 160s processing time

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
- **Service times**: Exponentially distributed per job type
    - purchase jobs: μ_purchase (average 1/μ_purchase seconds)
    - expire jobs: μ_expire (average 1/μ_expire seconds)
    - refund jobs: μ_refund (average 1/μ_refund seconds)
    - chargeback jobs: μ_chargeback (average 1/μ_chargeback seconds)
- **Workers**: 200 identical servers processing jobs in parallel
- **Queue discipline**: FIFO (first-in-first-out)

### System Components
- **Job Generator**: creates jobs according to Poisson(λ) and assigns types using π distribution
- **Queue**: unlimited capacity FIFO buffer storing waiting jobs
- **Worker Pool**: 200 parallel workers that process jobs according to their type-specific service rates
- **Timeout Detection**: jobs that exceed 160s processing time are marked as timed out
- **Metrics Collector**: continuously collects performance data during simulation

### Data Collection
- **Job-level data**: arrival time, start time, finish time, job type, timeout status
- **System-level samples**: queue length and worker utilization collected every 10 seconds
- **Aggregated metrics**: calculated at the end of each simulation run