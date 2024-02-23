#!/usr/bin/env python3

# MIT Licensed

# Python to calculate the normal using the monte carlo using MPI to paralellize the process.

# Run with 'mpirun -n 4 python assignment2_monte_carlo_v1.py'

# Import statements

# Module providing timing capabilities. 
import time
import math
import numpy as np
from mpi4py import MPI

# Functions

# Gaussian function to compute

def normal_pdf(x, mu, sigma):
  	
  	# Calculating the coefficient of the integral
    coefficient = 1 / (math.sqrt((2 * math.pi) ** len(mu) * np.linalg.det(sigma)))
    
    # Calculating the exponent of the integral
    exponent = -0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu))
    
    # Combining the two to make the complete equation
    pdf_val = coefficient * math.exp(exponent)
    
    # Returning the numerical answer
    return pdf_val
    
# Monte Carlo Method to estimate the normal

def monte_carlo(no_of_samples, mu, sigma):

	samples = np.random.multivariate_normal(mu, sigma, no_of_samples)
	
	# Computes the probability density function
	pdf_vals = np.array([normal_pdf(sample, mu, sigma) for sample in samples])
	
	# Calcualtes the estimated value for the calculated pdf values
	estimated_val = np.mean(pdf_vals)
	
	# Calculate estimated variance
	estimated_var = np.var(pdf_vals)
	
	# Returns the estimated value
	return estimated_val, estimated_var


# Start the timer
start_time = time.time()

if __name__ == '__main__':


	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	
	# Dimensions
	dimensions = [1, 6]
	
	# Number of samples per dimension
	no_of_samples = 100000
	
	# Identity matrixes
	sigmas = [np.eye(d) for d in dimensions]
	
	# Zero Mean vector for each dimension
	locations = [np.zeros(d) for d in dimensions]
	
	local_estimates = []
	
	for (dimension, sigma, location) in zip(dimensions, sigmas, locations):
		total_no_of_samples = size * no_of_samples
		local_estimate, local_estimate_var = monte_carlo(total_no_of_samples, location, sigma)
		local_estimates.append((local_estimate, local_estimate_var))
		
	
	
	if rank == 0:
		sum_estimates = []
		
		for i in range(1, size):
			received_estimates = comm.recv(source=i)
			sum_estimates.extend(received_estimates)
		
		for dimension, estimates in zip(dimensions_sum_estimates):
			print("Dimension: ", dimension)
			for i, (estimated_val, estimated_var) in enumerate(estimates):
				total_no_of_samples_total = no_of_samples * size
				error_val = np.sqrt(estimated_var / total_no_of_samples_total)
				print("Parameter Combo: ", i+1)
				print("Estimated integral :", estimated_val)
				print("Estimated variance :", estimated_var)
				print("Error bars :", error_val)
	else:
		comm.send(local_estimates, dest=0)
		# print("Time Taken:", time_taken, "sec")

		# Calculate time taken
		# end_time = time.time()
		# time_taken = end_time - start_time	
		