"""
Data models for ticket queue simulation
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict


class JobType(Enum):
    PURCHASE = "purchase"
    EXPIRE = "expire"
    REFUND = "refund"
    CHARGEBACK = "chargeback"


@dataclass
class Job:
    job_type: JobType
    arrival_time: float
    start_time: float = None
    finish_time: float = None
    timed_out: bool = False
    
    @property
    def response_time(self) -> float:
        """W_i - total time in system"""
        if self.finish_time is None:
            return 0.0
        return self.finish_time - self.arrival_time
    
    @property
    def service_time(self) -> float:
        if self.start_time is None or self.finish_time is None:
            return 0.0
        return self.finish_time - self.start_time


@dataclass
class Metrics:
    """Simulation metrics as defined in README"""
    completed_jobs: List[Job]
    queue_samples: List[tuple]  # (time, queue_length)
    worker_busy_time: float
    simulation_time: float
    num_workers: int = 200
    
    def get_W(self) -> float:
        """Average response time"""
        if not self.completed_jobs:
            return 0.0
        return sum(job.response_time for job in self.completed_jobs) / len(self.completed_jobs)
    
    def get_X(self) -> float:
        """Throughput (jobs per second)"""
        return len(self.completed_jobs) / self.simulation_time
    
    def get_rho(self) -> float:
        """Worker utilization"""
        return self.worker_busy_time / (self.num_workers * self.simulation_time)
    
    def get_L(self) -> float:
        """Average queue length"""
        if not self.queue_samples:
            return 0.0
        return sum(length for _, length in self.queue_samples) / len(self.queue_samples)
    
    def get_timeout_rate(self) -> float:
        """Percentage of jobs that timed out"""
        if not self.completed_jobs:
            return 0.0
        timed_out = sum(1 for job in self.completed_jobs if job.timed_out)
        return (timed_out / len(self.completed_jobs)) * 100
    
    def get_W_by_type(self, job_type: JobType) -> float:
        """Average response time by job type"""
        jobs = [job for job in self.completed_jobs if job.job_type == job_type]
        if not jobs:
            return 0.0
        return sum(job.response_time for job in jobs) / len(jobs)