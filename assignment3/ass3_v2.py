#!/usr/bin/env python3

# MIT Licensed

# Python code to solve the Laplace equation using a random walk method
# Run with 'mpirun -n 4 ass3_v2.py'

# Import statements
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

# Start of class
class RandomWalkSolver:
    def __init__(self, num_simulations, grid_shape, region, boundary_charges):
        self.num_simulations = num_simulations
        self.grid_shape = grid_shape
        self.region = region
        self.boundary_charges = boundary_charges
        self.dx = (region[1] - region[0]) / grid_shape[0]
        self.dy = (region[3] - region[2]) / grid_shape[1]
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
        Performing the random walk around the grid with bias towards lower potentials
        """
        current_point = start_point
        while self.is_inside_region(current_point):
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            potential_here = self.calculate_mean_potential([current_point])
            dx_potential = self.calculate_mean_potential([(current_point[0] + dx * self.dx, current_point[1])])
            dy_potential = self.calculate_mean_potential([(current_point[0], current_point[1] + dy * self.dy)])
            dx_diff = potential_here - dx_potential
            dy_diff = potential_here - dy_potential
            probabilities = [1, 1, 1]  
            if dx != 0:
                probabilities[dx + 1] += dx_diff
            if dy != 0:
                probabilities[dy + 1] += dy_diff
            # Remove negative probability
            probabilities = [max(0, p) for p in probabilities]
            # Normalize probabilities
            probabilities_sum = sum(probabilities)
            probabilities = [p / probabilities_sum for p in probabilities]
            # Choose the direction based on  probabilities
            direction = np.random.choice([-1, 0, 1], p=probabilities)
            if dx != 0 and direction == dx:
                current_point = (current_point[0] + dx * self.dx, current_point[1])
            elif dy != 0 and direction == dy:
                current_point = (current_point[0], current_point[1] + dy * self.dy)
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
        Calculate mean potential
        """
        potential = 0
        for end_point in end_points:
            x, y = end_point
            for side, charge in self.boundary_charges.items():
                if side == "left" and x == self.region[0]:
                    potential += charge
                elif side == "right" and x == self.region[1]:
                    potential += charge
                elif side == "bottom" and y == self.region[2]:
                    potential += charge
                elif side == "top" and y == self.region[3]:
                    potential += charge
            # Add contribution from distance
            potential += np.sqrt(x ** 2 + y ** 2)
        return potential / len(end_points) 
    def calculate_greens_function_error(self, num_trials):
        """
        Calculating error of greens function
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

# Using the class and functions in the main function
def main():
    """
    Main function to utilize the class features 
    to calculate the Green Function, error and potential
    """
    # x_min, x_max, y_min, y_max
    region = (0, 10, 0, 10)  
    grid_shape = (10, 10)
    # Number of random walks per grid point    
    num_simulations = 10  
    ### EDIT: BOUNDARY CHARGES
    boundary_charges = {"left": 1, "right": 1, "top": 1, "bottom": 1}
    random_walk_solver = RandomWalkSolver(num_simulations, grid_shape, region, boundary_charges)
    # Create a potential grid using laplace equation
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
    ### EDIT: Change point its evaluated at
    specified_point = (5, 5) 
    # Print the specified point
    if random_walk_solver.rank == 0:
        print("Specified point:", specified_point)

    # Convert specified point to grid indices
    x_index = np.argmin(np.abs(np.linspace(region[0], region[1], grid_shape[0]) - specified_point[0]))
    y_index = np.argmin(np.abs(np.linspace(region[2], region[3], grid_shape[1]) - specified_point[1]))
    # Calculate Green's function
    greens_function = potential_grid[x_index, y_index]
    # Calculate error estimate of the Green's function
    mean_greens_function, std_dev_greens_function = random_walk_solver.calculate_greens_function_error(num_simulations)
    greens_function_error = std_dev_greens_function / np.sqrt(num_simulations)
    if random_walk_solver.rank == 0:
        print("Green's function value at specified point:", greens_function)
        print("Error estimate of the Green's function:", greens_function_error)
if __name__ == "__main__":
    main()
