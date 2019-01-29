# Ivan Thieu
# 01/23/19
# CEE 250D Homework #2

# Import libraries
import cvxpy as cp
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

#============= Problem 1b (Solve Dual Problem) ===========
# Define constraint functions
# 6y1 + 2y2 >= 9
def f1(y1):
    return (9 - 6*y1)/2

# y1 + 4y2 >= 6
def f2(y1):
    return (6 - y1)/4

# y1 + y2 >= 3
def f3(y1):
    return 3 - y1

# Check a pair of (x, y) coordinates to ensure that it meets all the constraints
def check_constraints(y1, y2):
    if (6*y1 + 2*y2 >= 9 and
        y1 + 4*y2 >= 6 and
        y1 + y2 >= 3):
        return True
    else:
        return False

# The below functions are the same functions as above but solving for y1 given y2
# It will be used for filling in the infeasible regions
# 6y1 + 2y2 >= 9 - Solve for y1 given y2
def f1_2(y2):
    return (9 - 2*y2)/6

# y1 + 4y2 >= 6 - Solve for y1 given y2
def f2_2(y2):
    return 6 - 4*y2

# y1 + y2 >= 3 - Solve for y1 given y2
def f3_2(y2):
    return 3 - y2

# Objective Function of Dual Problem
def objective_func(extreme_point):
    y1, y2 = extreme_point
    return 2*y1 + 4*y2

# Define y1
y1_arr = np.arange(-1, 50, 1)

# Define portion of the A matrix for each function
# This is to solve for the intersection points for the graph
# E.g. Get intersection of 6y1 + 2y2 = 9 and 1y1 + 4y2 = 6
f1_a = [6, 2]
f2_a = [1, 4]
f3_a = [1, 1]

# Define portion of b matrix for each function
f1_b = 9
f2_b = 6
f3_b = 3

# Solve for X matrix for each combination of functions (Get intersections)
f1_f2_x, f1_f2_y = np.round(np.linalg.solve([f1_a, f2_a], [f1_b, f2_b]), 5)
f1_f3_x, f1_f3_y = np.round(np.linalg.solve([f1_a, f3_a], [f1_b, f3_b]), 5)
f2_f3_x, f2_f3_y = np.round(np.linalg.solve([f2_a, f3_a], [f2_b, f3_b]), 5)

# Generate list of x and y coordinates that intersected
intersection_x = [f1_f2_x, f1_f3_x, f2_f3_x]
intersection_y = [f1_f2_y, f1_f3_y, f2_f3_y]

# Get only the intersection points that meet all the constraints
# These are the extreme points
extreme_points = [(x, y) for (x, y) in list(zip(intersection_x, intersection_y))
                  if check_constraints(x, y)]

# Populate x and y coordinates for extreme points
extreme_x, extreme_y = list(zip(*extreme_points)) # Unzip the list for [x1, x2], [y1, y2] instead of [(x1, y1), (x2, y2)]

# Calculate the Min function for each extreme point and choose the min
pot_sols = [objective_func(extreme) for extreme in extreme_points]
dual_min = min(pot_sols)

# Point corresponding to solution of dual problem
dual_solution_point = extreme_points[pot_sols.index(dual_min)]
print("Min W (Dual Problem)\n%.2f" %dual_min)
print("\nOptimal solution (y1, y2) for dual problem\n"
      "(%.2f, %.2f)" %dual_solution_point)

# Plot y1 and the corresponding y2 values for each constraint
plt.plot(y1_arr, f1(y1_arr), color="black", label="constraint1")
plt.plot(y1_arr, f2(y1_arr), color="black", label="constraint2")
plt.plot(y1_arr, f3(y1_arr), color="black", label="constraint3")

# Add limits to make the valid range more visible
upper_y1_bound = 3.0
lower_y1_bound = -0.5

upper_y2_bound = 3.0
lower_y2_bound = -0.5

plt.xlim([lower_y1_bound, upper_y1_bound])
plt.ylim([lower_y2_bound, upper_y2_bound])

# Fill in the feasible region
feasible_reg, = plt.fill([f1_2(upper_y2_bound), upper_y1_bound, upper_y1_bound, f2_f3_x, f1_f3_x],
                 [upper_y2_bound, upper_y2_bound, f2(upper_y1_bound), f2_f3_y, f1_f3_y],
                 alpha=.25, color="black")

# Hatch the infeasible regions
# Only defining one of the plots as a variable for the legend since they all have the same hatch pattern
infeasible, = plt.fill([lower_y1_bound, f1_2(lower_y2_bound), f1_2(upper_y2_bound), lower_y1_bound],
                       [lower_y2_bound, lower_y2_bound, upper_y2_bound, upper_y2_bound],
                       hatch="///", alpha=0)
plt.fill([lower_y1_bound, upper_y1_bound, upper_y1_bound, lower_y1_bound],
         [lower_y2_bound, lower_y2_bound, f2(upper_y1_bound), f2(lower_y1_bound)],
         hatch="///", alpha=0)
plt.fill([lower_y1_bound, upper_y1_bound, upper_y1_bound, lower_y1_bound],
         [lower_y2_bound, lower_y2_bound, f3(upper_y1_bound), f3(lower_y1_bound)],
         hatch="///", alpha=0)

# Plot the extreme points
pot_points = plt.scatter(extreme_x, extreme_y, s=100, color="black")

# Mark the solution
min_point = plt.scatter(dual_solution_point[0], dual_solution_point[1], s=150, marker="*", color="grey")

# Labels
plt.title("Graphical Solution for Dual Problem")
plt.xlabel(r"$y_1$")
plt.ylabel(r"$y_2$")
plt.legend(handles=[feasible_reg, infeasible, pot_points, min_point],
           labels=["Feasible Region", "Infeasible Region", "Extreme Points", "Optimal Solution"],
           loc="upper right")

# Show formulas on the graph to make it clearer since graph prints in black and white
plt.text(0.75, 2.75, r"$6y_1 + 2y_2 = 9$", fontsize=10, bbox={"facecolor":"white"})
plt.text(-0.15, 2.75, r"$y_1 + 4y_2 = 6$", fontsize=10, bbox={"facecolor":"white"})
plt.text(-0.35, 1.35, r"$y_1 + y_2 = 3$", fontsize=10, bbox={"facecolor":"white"})

# Show coordinates of extreme points in the graph
for x, y in extreme_points:
    plt.text(x + 0.05, y + 0.05, "(%.2f, %.2f)" %(x, y))

# Show the plot
# Block=false prevents the code from getting paused from the graph
# Remove it to view the graph and pause the rest of the code
plt.show(block=False)

#============= Problem 1c (Computational Solution to Maximization Problem) ===========
# Define givens
n = 3 # Number of variables to solve for
a_matrix = np.array([[6, 1, 1],
                     [2, 4, 1]])
b_matrix = np.transpose([2, 4])
c_transpose = np.array([[9, 6, 3]])

# Set up problem
x_matrix = cp.Variable(n)
objective = cp.Maximize(c_transpose*x_matrix)
constraints = [a_matrix*x_matrix <= b_matrix, x_matrix >= 0]
prob = cp.Problem(objective, constraints)

# Solve problem
result = prob.solve()
print("\nOptimal Solution for Primal\n"
      "max Z = %.2f\n"
      "x1: %.2f\n"
      "x2: %.2f\n"
      "x3: %.2f" %tuple(np.append(objective.value, x_matrix.value)))

#============= Problem 2 (Minimum Capacity) ===========
# Define functions to get the constraint values
# Flood Reservoir Constraint (Min amount of volume to leave open in case of flood)
def flood_reservoir(reserve_data, inflow_data, yr_index):
    return np.array(reserve_data + inflow_data.iloc[yr_index, :])

# Min Storage Constraint (Min amount of volume to leave in the reservoir)
def min_storage(storage_data, inflow_data, yr_index):
    return np.array(storage_data - inflow_data.iloc[yr_index, :])

# Water Supply Constraint (Min amount of water that needs to be released)
# Treating t-1 as previous month
# For Jan, Year 1, it will use inflow from Dec, Year 1
def water_supply(flow_data, inflow_data, month_index, yr_index):
    return flow_data[month_index] - inflow_data.iloc[yr_index, month_index - 1]

# Max Release Constraint (Max amount of water that can be released due to channel)
def max_release(release_data, inflow_data, month_index, yr_index):
    return release_data[month_index] - inflow_data.iloc[yr_index, month_index - 1]

# Function to generate the constraints.
# Using data that get defined later
def generate_constraints(number_years):
    capacity_constraints = []
    for yr in range(0, number_years):
        capacity_constraints += [capacity - b >= flood_reservoir(data_month_constraints["V(t+1)"], data_inflow, yr),
                                b >= min_storage(data_month_constraints["Smin (t+1)"], data_inflow, yr)] + \
                               [b[mon - 1] - b[mon] >=
                                water_supply(data_month_constraints["q (t)"], data_inflow, mon, yr)
                                for mon in range(0, num_months)] + \
                               [b[mon - 1] - b[mon] <=
                                max_release(data_month_constraints["f(t)"], data_inflow, mon, yr)
                                for mon in range(0, num_months)]
    return capacity_constraints

# Process and clean the data
# Import the data
file_path = r"C:\Users\ivant\Desktop\CEE250D\Homeworks\Homework2\Data Set for HW #2 (2019).xlsx"
data = pd.read_excel(file_path)

# Split into inflow data (varies on month and year). Set the index to be the year
data_inflow = data.dropna().set_index("year")

# Split into monthly constraints (varies only on months)
split_point = data.loc[data["year"] == "Month"].index[0]
data_month_constraints = data.iloc[split_point:, :] # Get rows after split point and all columns
data_month_constraints.columns = data_month_constraints.iloc[0].tolist() # Reset the headers to be the first row
data_month_constraints = data_month_constraints.iloc[1:].set_index("Month") # Drop the duplicate row and set index
data_month_constraints.dropna(axis=1, inplace=True) # Drop the columns that are not defined

# Define variables/matrices to solve for
num_months = len(data_month_constraints.index)
num_years = len(data_inflow.index)

# Variables to solve using Linear Programming
capacity = cp.Variable()
b = cp.Variable(num_months)

# Objective Function
obj = cp.Minimize(capacity)

# Constraints
constraints_capacity = generate_constraints(num_years)

# Set up problem
prob = cp.Problem(obj, constraints=constraints_capacity)

# Solve
prob.solve()

# Print out the solutions
print("\nMin Capacity:\n%.2f" %capacity.value)
print("\nB values:")
for b_j in range(0, len(b.value)):
    print("b" + str(b_j) + ": %.2f" %b[b_j].value)

# Check Constraints
month_index = [data_month_constraints.index]

# B(j-1) - B(j). Roll by 1 to get B(j-1)
b_diff = np.roll(b.value, 1) - b.value

# Max Release constraint
release_const = []
for year in range(num_years):
    release_const.append(b_diff + np.roll(data_inflow.iloc[year, :], 1))
print(DataFrame(np.amax(np.array(release_const), axis=0), index=month_index, columns=["Max Release Constraint"]))

# Storage Constraints (St+1)
s_t1 = b.value + data_inflow.iloc[:, :]
print(DataFrame(np.amin(np.array(s_t1), axis=0), index=month_index, columns=["Storage Constraint"]))

# Flood Control Constraint (C - St+1)
flood_control = capacity.value - s_t1
print(DataFrame(np.amin(np.array(flood_control), axis=0), index=month_index, columns=["Flood Control Constraint"]))

# Min Release constraint
print(DataFrame(np.amin(np.array(release_const), axis=0), index=month_index, columns=["Min Release Constraint"]))