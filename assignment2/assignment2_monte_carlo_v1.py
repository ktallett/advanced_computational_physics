#!/usr/bin/env python3

# MIT Licensed

# Python to calculate the normal using the monte carlo using MPI to paralellize the process.

# Import statements

import math
import numpy as np
from mpi4py import MPI

# Functions

# Gaussian function to compute

def normal_pdf(x, mu, sigma):
  	
  	# Calculating the coefficient of the integral
    coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)
    
    # Calculating the exponent of the integral
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    
    # Combining the two to make the complete equation
    pdf_val = coefficient * math.exp(exponent)
    
    # Returning the numerical answer
    return pdf_val
    
# Monte Carlo Method to estimate the normal

def monte_carlo(no_of_samples, mu, sigma):

	samples = np.random.normal(my, sigma, no_of_samples):
	
	# Computes the probability density function
	pdf_vals = np.array([normal_pdf(sample, mu, sigma) for sample in samples])
	
	# Calcualtes the estimated value for the calculated pdf values
	estimated_val = np.mean(pdf_vals)
	
	# Returns the estimated value
	return estimated_val


if __name__ == '__main__':


	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	
	no_of_samples = 100000
	total_no_of_samples = size * no_of_samples
	
	# Calculate an estimate on a single core
	local_estimate = monte_carlo(num_samples_per_process, 0, 1)
	
	# Gathers all estimates and calculates the average of all cores processes
	all_estimates = comm.gather(local_estimate, root=0)
	
	if rank == 0:
		final_estimate = np.mean(all_estimates)