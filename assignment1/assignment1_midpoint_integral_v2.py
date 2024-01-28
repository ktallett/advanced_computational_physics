#!/usr/bin/env python3

# Python script to calculate Pi utilising MPI using a mid-point integral

# Run with 'mpirun -n 8 python assignment1_midpoint_integral_v2.py'


# Import statements
from mpi4py import MPI
import time


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

# Function to be integrated
def f(x):
    return 4.0 / (1.0 + x * x)

# Mid-point Integral method
def mid_point_integral_method(start, end, num_intervals, h):
    local_sum = 0.0
    for i in range(start, end):
        x_midpoint = (i + 0.5) * h
        local_sum += f(x_midpoint)
    return local_sum

# Define starting and end point of integral
start_point = 0.0
end_point = 1.0

# Define number of intervals
num_intervals = 1000000  

# Width of each subinterval
h = (end_point - start_point) / num_intervals

# Spread intervals equally to different processors // ensuring it is an integer number

local_n = num_intervals // p

# Defining integration range each process is responsible for

local_start = my_rank * local_n
local_end = (my_rank + 1) * local_n

# Run mid-point integral function
local_integral = mid_point_integral_method(local_start, local_end, num_intervals, h)

total_integral = 0.0

# Spread the workload amongst the processors using send and recieve
start_time = time.time()

if my_rank == 0:
    total_integral = local_integral
    for source in range(1, p):
        integral = comm.recv(source=source)
        total_integral += integral
else:
    comm.send(local_integral, dest=0)

if my_rank == 0:
    pi_approx = total_integral / num_intervals
    print("Approximation of Pi:", pi_approx)
    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time Taken:", time_taken, "sec")



# End parallelisation
MPI.Finalize()
