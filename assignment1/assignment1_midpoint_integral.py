#!/usr/bin/env python3


# Program to calculate Pi using midpoint integral

# Import statements

from mpi4py import MPI
import time


# Function to be integrated

def integral(x):
	return(4.0/(1.0 + x*x))
	

# Mid-point Integral method

def mid_point_integral_method(start_point, end_point, no_of_itr, rank, no_of_proc):
	h = (end_point - start_point)/no_of_itr
	local_itr = no_of_itr
	local_sp = start_point + rank + local_itr * h
	local_ep = end_point + local_itr * h
	
	local_sum = 0.0
	for i in range(local_itr):
		x_midpoint = local_sp + (i + 0.5) * h
		local_sum = local_sum + integral(x_midpoint)
	
	local_sum *= h
	return local_sum

# Check script is being run

if __name__ == "__main__":
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	no_of_proc = comm.Get_size()

# Defining Integral boundaries and no. of iterations
	start_point = 0.0
	end_point = 1.0
	no_of_itr = 10000000
	
# Spread the workload amongst the processors and collecting
	
	# Start timer
	start_time = time.time()

	local_sum = mid_point_integral_method(start_point, end_point, no_of_itr, rank, no_of_proc)
	total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
	
	if rank == 0:
		
		print("Integral answer:", total_sum)
		# End timer, calculate and print time taken
		end_time = time.time()
		time_taken = end_time - start_time
		print("Time Taken:", time_taken, "sec")
