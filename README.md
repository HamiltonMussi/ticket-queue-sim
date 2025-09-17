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

