"""
Ticket queue simulator - M/G/200 system
"""
import simpy
import random
from models import Job, JobType, Metrics


class TicketQueueSimulator:
    def __init__(self, config):
        self.config = config
        self.completed_jobs = []
        self.queue_samples = []
        self.worker_busy_time = 0
        
    def job_generator(self, env, queue):
        """Generate jobs: Poisson(λ) arrival, Multinomial(π) type selection"""
        while True:
            # When does next job arrive? Exponential(λ)
            inter_arrival = random.expovariate(self.config['lambda_total'])
            yield env.timeout(inter_arrival)
            
            # What type of job? Multinomial(π)
            job_type = random.choices(
                [JobType.PURCHASE, JobType.EXPIRE, JobType.REFUND, JobType.CHARGEBACK],
                weights=[
                    self.config['pi_purchase'],
                    self.config['pi_expire'], 
                    self.config['pi_refund'],
                    self.config['pi_chargeback']
                ]
            )[0]
            
            job = Job(job_type=job_type, arrival_time=env.now)
            queue.put(job)
    
    def worker(self, env, queue, workers_resource):
        """Worker process: FIFO queue discipline"""
        while True:
            job = yield queue.get()
            
            with workers_resource.request() as req:
                yield req
                job.start_time = env.now
                
                # Service time: Shifted Exponential based on job type
                # time = shift + Exponential(rate)
                shift_params = {
                    JobType.PURCHASE: self.config['shift_purchase'],
                    JobType.EXPIRE: self.config['shift_expire'],
                    JobType.REFUND: self.config['shift_refund'],
                    JobType.CHARGEBACK: self.config['shift_chargeback']
                }

                rate_params = {
                    JobType.PURCHASE: self.config['rate_purchase'],
                    JobType.EXPIRE: self.config['rate_expire'],
                    JobType.REFUND: self.config['rate_refund'],
                    JobType.CHARGEBACK: self.config['rate_chargeback']
                }

                shift = shift_params[job.job_type]
                rate = rate_params[job.job_type]
                exponential_component = random.expovariate(rate)
                service_time = shift + exponential_component
                start_time = env.now
                yield env.timeout(service_time)
                
                job.finish_time = env.now
                self.worker_busy_time += (env.now - start_time)
                
                # Timeout detection: > 160s
                if job.service_time > 160.0:
                    job.timed_out = True
                
                self.completed_jobs.append(job)
    
    def monitor(self, env, queue):
        """Collect system samples every 10 seconds"""
        while True:
            self.queue_samples.append((env.now, len(queue.items)))
            yield env.timeout(10.0)
    
    def run(self, simulation_time=3600):
        """Run simulation"""
        env = simpy.Environment()
        queue = simpy.Store(env)
        workers = simpy.Resource(env, capacity=200)
        
        # Start processes
        env.process(self.job_generator(env, queue))
        env.process(self.monitor(env, queue))
        
        # Start 200 workers
        for i in range(200):
            env.process(self.worker(env, queue, workers))
        
        # Run simulation
        env.run(until=simulation_time)
        
        # Return metrics
        return Metrics(
            completed_jobs=self.completed_jobs,
            queue_samples=self.queue_samples,
            worker_busy_time=self.worker_busy_time,
            simulation_time=simulation_time
        )