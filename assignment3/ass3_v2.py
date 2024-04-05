#!/usr/bin/env python3

# MIT Licensed

# Python code to solve the Laplace equation using a random walk method
# Run with 'mpirun -n 4 assignment3_ass3_v2.py'
# Edit n-tasks in job script for 8 and 16 to match 8 and 16 respectively

# Import statements
import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Start of class
class RandomWalkSolver:
    def __init__(self, num_simulations, grid_shape, region):
        # Set class properties
        self.num_simulations = num_simulations
        self.grid_shape = grid_shape
        self.region = region
        self.dx = (region[1] - region[0]) / grid_shape[0]
        self.dy = (region[3] - region[2]) / grid_shape[1]
        # Set properties based on the number of cores being used
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def is_inside_region(self, point):
        """
        Check if current point is inside the region
        """
        x, y = point
        x_min, x_max, y_min, y_max = self.region
        return x_min <= x <= x_max and y_min <= y <= y_max

    def random_walk(self, start_point):
        """
        Performing the random walk around the grid
        """
        current_point = start_point
        while self.is_inside_region(current_point):
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            current_point = (current_point[0] + dx * self.dx, current_point[1] + dy * self.dy)
        return current_point

    def solve_laplace_equation_parallel(self):
        """
        Solve the laplace equation
        """
        subdomain_shape = (self.grid_shape[0] // self.size, self.grid_shape[1])
        subdomain_start_row = self.rank * subdomain_shape[0]
        subdomain_end_row = min((self.rank + 1) * subdomain_shape[0], self.grid_shape[0])

        local_potential_grid = np.zeros(subdomain_shape)
        for i in range(subdomain_start_row, subdomain_end_row):
            for j in range(self.grid_shape[1]):
                end_points = []
                for _ in range(self.num_simulations):
                    end_point = self.random_walk((i * self.dx, j * self.dy))
                    end_points.append(end_point)
                local_potential_grid[i - subdomain_start_row, j] = self.calculate_mean_potential(end_points)

        global_potential_grid = np.zeros(self.grid_shape)
        self.comm.Allgather(local_potential_grid, global_potential_grid)

        return global_potential_grid

    def calculate_mean_potential(self, end_points):
        """
        Calculate mean potential to use in Laplace equation
        """
        
        potential = 0
        for end_point in end_points:
            # end point is the distance from the origin
            potential += np.sqrt(end_point[0] ** 2 + end_point[1] ** 2)
        return potential / len(end_points)
        
    def calculate_greens_function_error(self, num_trials):
        """
        Calculating error of green's function
        """
        
        greens_function_values = []
        for _ in range(num_trials):
            potential_grid = self.solve_laplace_equation_parallel()
            x_index = np.argmin(np.abs(np.linspace(self.region[0], self.region[1], self.grid_shape[0]) - 0))
            y_index = np.argmin(np.abs(np.linspace(self.region[2], self.region[3], self.grid_shape[1]) - 0))
            greens_function = potential_grid[x_index, y_index]
            greens_function_values.append(greens_function)
        greens_function_values = np.array(greens_function_values)
        mean_greens_function = np.mean(greens_function_values)
        std_dev_greens_function = np.std(greens_function_values)
        return mean_greens_function, std_dev_greens_function
    
    def plot_exit_probability(self, exit_probability, save_path):
        """
        Plot the probability of exiting at each point and save as an image
        """
        # Create a 2D array with the scalar value repeated to match the grid shape
        exit_probabilities = np.full(self.grid_shape, exit_probability)
        
        plt.imshow(exit_probabilities, origin='lower', extent=self.region, cmap='viridis')
        plt.colorbar(label='Exit Probability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Exit Probability Distribution')
        plt.savefig(save_path)
        plt.close()
def main():
    # Define region as (x_min, x_max, y_min, y_max)
    region = (0, 10, 0, 10)  
    grid_shape = (10, 10)
    # Number of random walks per grid point    
    num_simulations = 10  

    random_walk_solver = RandomWalkSolver(num_simulations, grid_shape, region)

    # Solve Laplace equation in parallel
    potential_grid = random_walk_solver.solve_laplace_equation_parallel()
    
    # Calculate exit probabilities
    exit_probabilities = np.mean(np.where(potential_grid == 0, 1, 0))

    # Plot exit probability distribution
    save_path = "exit_probabilities.png"
    random_walk_solver.plot_exit_probability(exit_probabilities, save_path)
    
    # Calculate Average Potential
    total_potential = 0
    num_points = 0
    
    # Iterate over grid points and sum potential 
    for i in range(len(potential_grid)):
        for j in range(len(potential_grid[0])):
            x = i
            y = j
            if region[0] <= x <= region[1] and region[2] <= y <= region[3]:
                total_potential += potential_grid[i, j]
                num_points += 1

    average_potential = total_potential / num_points
    
    print("Average potential: ", average_potential)

    specified_point = (0, 2.5) 
    
    # Print the specified point
    if random_walk_solver.rank == 0:
        print("Specified point:", specified_point)

    # Convert specified point to grid indices
    x_index = np.argmin(np.abs(np.linspace(region[0], region[1], grid_shape[0]) - specified_point[0]))
    y_index = np.argmin(np.abs(np.linspace(region[2], region[3], grid_shape[1]) - specified_point[1]))

    # Calculate Green's function value at the center point
    greens_function = potential_grid[x_index, y_index]
    
    # Calculate error estimate of the Green's function
    mean_greens_function, std_dev_greens_function = random_walk_solver.calculate_greens_function_error(num_simulations)
    greens_function_error = std_dev_greens_function / np.sqrt(num_simulations)
    
    if random_walk_solver.rank == 0:
        print("Green's function value at specified point:", greens_function)
        print("Error estimate of the Green's function:", greens_function_error)

if __name__ == "__main__":
    main()
