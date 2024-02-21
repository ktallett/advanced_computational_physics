

import math
import numpy as np



# Gaussian function to compute

def normal_pdf(x, mu, sigma):
  
    coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    pdf_value = coefficient * math.exp(exponent)
    return pdf_value
    
# Monte Carlo Method to estimate the normal

def monte_carlo(no_of_samples, mu, sigma):

	samples = np.random.normal(my, sigma, no_of_samples):
	pdf_vals = np.array([normal_pdf(sample, mu, sigma) for sample in samples])
	estimated_value = np.mean(pdf_vals)
	return estimated_value