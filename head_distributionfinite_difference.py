# CEE 250B Homework #7
# Ivan Thieu
# 12/02/2018

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================Functions==============================
# Generate the b matrix for a specific row
def generate_b_mat(time_iteration_num):
    b_mat = b_mat_co * df.iloc[time_iteration_num, :] # Multiply with head at i,j
    b_mat.iloc[0] -= head_left  # Account for constant head in left row
    b_mat.iloc[-1] -= head_right  # Account for constant head in last row
    b_mat.iloc[pump_left_pos] += pump_factor  # Account for pumping at 25m
    b_mat.iloc[pump_right_pos] += pump_factor  # Account for pumping at 75m
    return b_mat.tolist()

# Define constants
specific_storage = 1
k = 10
pump_rate = 1
length = 100
time = 1000 # Ensure it reaches steady state by setting high run time
head_left = 8
head_right = 10
pump_left = 25
pump_right = 75

# Define delta x and t
delta_x = 0.1
delta_t = 0.1

# Create the x and t nodes that we will solve for
x_points = np.arange(delta_x, length, delta_x)
t_points = np.arange(0, time + delta_t, delta_t)

# Set up the df to record the head values at each x and t node
df = pd.DataFrame(columns=x_points, index=t_points)

# Solve the initial conditions (Assume linear relationship)
slope, intercept = np.polyfit([0, length], [head_left, head_right], 1)[:] # Get slope and y intercept
df.iloc[0] = slope*x_points + intercept # Calculate points at time 0 and input results into df

# Calculations for the matrix to solve system of linear equations
a_mat_co = -1*(2+specific_storage*delta_x**2/(k*delta_t)) # b term in the A matrix
pump_factor = pump_rate/delta_x*delta_x**2/k # Additional term in pumping locations
b_mat_co = -1*(specific_storage*delta_x**2/(k*delta_t)) # Coefficient in the B matrix

# Get the index of the pumps in the x_points array
pump_left_pos = int(pump_left/delta_x - 1)
pump_right_pos = int(pump_right/delta_x - 1)

# Create A matrix
a_matrix = (np.diag([1]*(len(x_points) - 1), -1) +
           np.diag([a_mat_co]*len(x_points), 0) +
           np.diag([1]*(len(x_points) - 1), 1))

# Use data from current row in dataframe to fill in next row
for i in range(0, len(t_points) - 1):
    print(i) # Generate some sort of response while running
    df.iloc[i + 1] = np.linalg.solve(a_matrix, generate_b_mat(i))

# Output results to a txt file to view easily
file_path = r"C:\Users\ivant\Desktop\CEE250B\Homework7\cee250b_output_.txt" # Desired location of output
np.savetxt(file_path,
           df.values,
           delimiter='\t',
           header="\t".join(map(str, df.columns.values.tolist())))

# Calculate position of desired times
desired_days = [0, 3, 20, 100, 1000]
desired_pos = [int(i/delta_t) for i in desired_days]

# Create a plot for each desired day
for i in desired_pos:
    plt.figure()
    plt.plot(x_points, df.iloc[i])
    plt.title("Head distribution for t = " + str(int(i*delta_t)) + " Day(s)")
    plt.xlabel("Length (m)")
    plt.ylabel("Head (m)")
    plt.show()