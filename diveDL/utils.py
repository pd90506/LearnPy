import time
import numpy as np
import math

class Timer:
    """Record mulitple runing times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer"""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """Return the average time"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time"""
        return sum(self.times)
    
    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)