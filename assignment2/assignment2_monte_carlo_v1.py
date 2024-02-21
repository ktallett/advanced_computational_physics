# Import libraries

import math
import numpy as np

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