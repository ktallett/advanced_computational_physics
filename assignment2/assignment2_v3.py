#!/usr/bin/env python3

# MIT Licensed

# Python to calculate the multidimensional normal using Monte Carlo
# Using MPI reduce and SUM to parallelize the process.
# Run with 'mpirun -n 4 python assignment2_v3.py'

# Import statements

import math
import time
from mpi4py import MPI
import numpy as np


# Start of class
class MonteCarloSimulation:
    def __init__(self, num_simulations, num_variables, global_seed=None):
        # Set class properties
        self.num_simulations = num_simulations
        self.num_variables = num_variables
        self.global_seed = global_seed
        # Set properties based on the number of cores being used
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    # Generates samples from seed, dimension, and function
    def _worker(self, seed, dimension, function, *args):
        """
        This function creates the samples for the simulation
        """
        np.random.seed(seed)
        # Generate samples using inverse sampling
        samples = self._inverse_sampling(dimension)
        evaluations = function(samples, *args)
        local_mean = np.mean(evaluations)
        local_var = np.var(evaluations)
        return samples, local_mean, local_var
    
    # Inverse sampling function
    def _inverse_sampling(self, dimension):
        """
        Generate samples using inverse sampling
        """
        samples = []
        for _ in range(self.num_simulations // self.size):
            # Generate samples using inverse sampling for each dimension
            sample = np.random.uniform(0, 1, dimension)
            samples.append(sample)
        return np.array(samples)

    # Calculates errors of the integral
    def _compute_error(self, integral_values):
        """
        This function calculates the errors of the integral
        """
        global_integral = np.zeros_like(integral_values)
        self.comm.Allreduce(integral_values, global_integral, op=MPI.SUM)
        global_integral /= self.size
        # Compute error
        squared_diff = (integral_values - global_integral) ** 2
        mean_squared_diff = np.mean(squared_diff)
        error = np.sqrt(mean_squared_diff / (self.size - 1))
        return error
    
    # Integrate function
    def run_integration(self, dimension, function, *args):
        """
        This function calculates the integral using MPI reduce to parallelize the integration
        """
        local_seed = np.random.randint(0, 10000) if self.global_seed is None else self.global_seed + self.rank
        samples, local_mean, local_var = self._worker(local_seed, dimension, function, *args)
        # Set global variable matrices
        global_samples = np.zeros_like(samples)
        global_mean = np.zeros_like(local_mean)
        global_var = np.zeros_like(local_var)
        self.comm.Allreduce(samples, global_samples, op=MPI.SUM)
        self.comm.Allreduce(local_mean, global_mean, op=MPI.SUM)
        self.comm.Allreduce(local_var, global_var, op=MPI.SUM)
        global_samples /= self.size
        global_mean /= self.size
        global_var /= self.size
        error = self._compute_error(local_mean)
        return global_samples, global_mean, global_var, error

# Gaussian function to compute
def normal_pdf(x_vals, mu_mv, sigma):
    """
    Gaussian function
    """
    # Calculating the coefficient of the integral
    coefficient = 1 / (math.sqrt((2 * math.pi) ** len(mu_mv) * np.linalg.det(sigma)))
    # Making sure the dimensions are the same size
    centered_samples = x_vals - mu_mv.reshape(1, -1)
    # Calculating the exponent of the integral
    exponent = -0.5 * np.sum(np.dot(centered_samples, np.linalg.inv(sigma)) * centered_samples, axis=1)
    # Combining the two to make the complete equation
    pdf_val = coefficient * np.exp(exponent)
    return pdf_val

# Using the class and functions in the main function
def main():

    """
    Main function to utilize the class features 
    to calculate the integral of the Gaussian distribution function
    """
    start_time = time.time()
    # Number of samples
    num_simulations = 1000000
    # EDIT NUMBER OF DIMENSIONS
    num_variables = 6
    global_seed = 12
    mc_sim = MonteCarloSimulation(num_simulations, num_variables, global_seed)
    # EDIT: MEAN VECTOR SHIFT DISTRIBUTION
    mu_mv = np.zeros(num_variables)
    # EDIT: SPREAD
    sigma = np.diag([1, 2, 3, 4, 5, 6])
    # Evaluate integral, mean, variance, and error
    samples, mean, variance, error = mc_sim.run_integration(num_variables, normal_pdf, mu_mv, sigma)
    # Prints results if back at global
    if mc_sim.rank == 0:
        print("Worker Mean:", "%.4f"%mean)
        print("Worker Variance:", "%.4f"%variance)
        print("Integral of Normal Distribution:", "%.4f"%np.mean(samples))
        print("Error in Integral:", "%.4f"%error)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time Taken:", time_taken)

# Check to see if the main program is being executed
if __name__ == "__main__":
    main()
