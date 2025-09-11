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