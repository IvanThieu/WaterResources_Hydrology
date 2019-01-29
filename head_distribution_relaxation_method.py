# ==================Ivan Thieu=====================
# ==================10/22/18=======================
# ==================CEE250B Homework #3============

# Problem 1 - Relaxation Method

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom functions

# Generate points along one direction
def generate_points(length, delta):
    num_steps = int(length / delta) + 1
    return np.linspace(0, length, num_steps)

# Generate all of the coordinates as a list of tuples
def generate_coordinates(x_pos_range, y_pos_range):
    return [(x, y) for y in y_pos_range for x in x_pos_range]

# Check if the coordinate has a value from the boundary conditions or if it is outside the aquifer
def boundary_check(coordinates):
    (x, y) = coordinates

    if x == 0 and y < aq_height:
        return left_bound_name
    elif x == aq_len and y <= cut_off[1]:
        return right_bound_name
    elif y == 0:
        return bot_psi
    elif ((x >= cut_off[0] and y >= cut_off[1]) or y >= aq_height) and not (x > cut_off[0] and y > cut_off[1]):
        return top_psi
    elif x > cut_off[0] and y > cut_off[1]: # Set the cutoff portion to NaN
        return np.nan
    else: # Everything else is an interior node
        return interior_node_name

# Initialize the psi value with initial guess if it not a boundary value
def initialize_psi(row_data):
    boundary_value = row_data[df_boundaries_col_name] # Check the value in the boundaries column for each row

    # If boundary condition is not a number/already defined from boundary_check, make interior node initial guess
    try:
        float(boundary_value)
        return boundary_value
    except ValueError:
        return initial_guess

# Set the previous sai column as the current sai column
def set_previous_sai(data_set):
    data_set[df_previous_sai_col_name] = data_set[df_current_sai_col_name]
    return

# Find current psi for psi_n
def find_psi_n(data_set, coordinate_value):
    # Prevent looking past boundaries when looking up coordinate values
    if coordinate_value[0] > cut_off[0] and coordinate_value[1] > cut_off[1]: # Both overshoot
        coordinate_value = (0, aq_height) # Set the coordinate to one of the top psi coordinates
    elif coordinate_value[0] > aq_len: # Length overshoots
        coordinate_value[0] = aq_len
    elif coordinate_value[1] > aq_height: # Height overshoots
        coordinate_value[1] = aq_height

    # Look up the sai value for the corresponding coordinate
    return data_set.loc[data_set[df_coordinates_col_name] == coordinate_value, df_current_sai_col_name].values[0]

# Calculate the psi value
def calculate_psi(coordinates, boundary_cond):
    (psi_coord_x, psi_coord_y) = coordinates

    # Forward method (Psi = Psi2)
    if boundary_cond == left_bound_name:
        psi2_coord = (psi_coord_x + delta_x, psi_coord_y)
        psi = find_psi_n(df, psi2_coord)

    # Backwards method (Psi = Psi4)
    elif boundary_cond == right_bound_name:
        psi4_coord = (psi_coord_x - delta_x, psi_coord_y)
        psi = find_psi_n(df, psi4_coord)

    # Relaxation method (Reference formula)
    elif boundary_cond == interior_node_name:
        # Coordinates for the psis
        psi1_coord = (psi_coord_x, psi_coord_y - delta_y)
        psi2_coord = (psi_coord_x + delta_x, psi_coord_y)
        psi3_coord = (psi_coord_x, psi_coord_y + delta_y)
        psi4_coord = (psi_coord_x - delta_x, psi_coord_y)

        # Actual psi_n values
        psi1 = find_psi_n(df, psi1_coord)
        psi2 = find_psi_n(df, psi2_coord)
        psi3 = find_psi_n(df, psi3_coord)
        psi4 = find_psi_n(df, psi4_coord)

        # Calculate psi
        psi = ((delta_x/delta_y)**2 * psi1 + psi2 + (delta_x/delta_y)**2 * psi3 + psi4) / (2 + 2*(delta_x/delta_y)**2)

    # If it is a boundary condition/defined
    else:
        psi = boundary_cond

    return  psi

# Define parameters (ft)
aq_len = 170.0 # Length of entire aquifer
aq_len1 = 80.0 # Left part of aquifer (Before cutoff)
aq_height1 = 40.0 # Height1 of aquifer (Before cutoff)
aq_height2 = 30.0 # Height2 of aquifer (After cutoff)
aq_height = aq_height1 + aq_height2 # Total height
cut_off = (aq_len1, aq_height1) # Cut off area (no stream flow after this point)

# Head dimensions (ft)
h0 = 100.0
h1 = 50.0

# K (ft/day)
k = 5.0

# Stopping criteria
stop_error_num = 0.10

# List to store the max errors in each iteration
max_error_list = []

# Define size of each node cell
delta_x = 2.0
delta_y = 2.0

# Define psi at the boundaries along with initial guess
bot_psi = 1.0
top_psi = 100.0
initial_guess = 50.0

# Data frame Column Names
df_coordinates_col_name = "Coordinates"
df_previous_sai_col_name = "Previous Sai"
df_current_sai_col_name = "Current Sai"
df_error_col_name = "Error"
df_boundaries_col_name = "Boundaries"

# Boundary values
left_bound_name = "Left"
right_bound_name = "Right"
interior_node_name = "Interior"

# Generate coordinates
aq_x = generate_points(aq_len, delta_x)
aq_y = generate_points(aq_height, delta_y)
aq_coord = generate_coordinates(aq_x, aq_y)

# Create the data frame
df = pd.DataFrame(data= {df_coordinates_col_name: aq_coord, df_previous_sai_col_name: np.nan,
                         df_current_sai_col_name: np.nan, df_error_col_name: np.nan},
                  columns=[df_coordinates_col_name, df_previous_sai_col_name, df_current_sai_col_name,
                           df_error_col_name])
df.set_index(df.index) # Set the index to make it easier to view row numbers

# Add boundary conditions to data frame
df[df_boundaries_col_name] = df[df_coordinates_col_name].map(lambda coordinates:
                                             boundary_check(coordinates))

# Initialize the current psi and previous sai columns
df[df_current_sai_col_name] = df.apply(initialize_psi, axis= 1) # Initialize
set_previous_sai(df) # Set previous sai as current sai

# Using for loop instead of while loop
for num_iterations in range(0, 500):
    # Calculate the psi. Use the most updated value once we get it
    for i in range(0, len(df.index)):
        psi_coord = df.at[i, df_coordinates_col_name] # Psi coordinates
        row_bound_cond = df.at[i, df_boundaries_col_name] # Boundary conditions
        psi_val = calculate_psi(psi_coord, row_bound_cond) # Calculate psi
        df.at[i, df_current_sai_col_name] = psi_val # Update psi value and keep looping

    # Error check
    df[df_error_col_name] = abs(df[df_current_sai_col_name] - df[df_previous_sai_col_name])

    # Take the max of the errors for only the interior nodes because that is what we are trying to find
    max_error = df[
        df[df_boundaries_col_name] == interior_node_name] \
        [df_error_col_name].max()
    max_error_list.append(max_error) # Append the result to our max error list
    print(max_error) # To have the script provide some sort of feedback while running

    # Continue and Stop Criteria
    if max_error > stop_error_num:
        set_previous_sai(df)
    else:
        print(num_iterations) # Output the number of iterations it took
        break

# df.to_csv("df_output.csv") # Output the data frame to view results

# Plot the error distribution
plt.plot(max_error_list, linewidth= 2.0, color= "k")
plt.title("Maximum Absolute Error Per Iteration")
plt.ylabel("Maximum Absolute Error")
plt.xlabel("Iteration Number")
plt.show()

# Contour plotting
num_streamlines = 10 # Define number of streamlines to display
X, Y = np.meshgrid(aq_x, aq_y) # Create X and Y matrix
Z = df[df_current_sai_col_name].values.reshape(X.shape) # Reshape the streamline values to fit the matrix
plt.contour(X, Y, Z, num_streamlines, colors="black") # Plot
plt.title("Streamline Map")
plt.xlabel("Length (ft)")
plt.ylabel("Height (ft)")
plt.xlim(0, 170)
plt.ylim(0, 70)
plt.axis("scaled")
plt.locator_params(axis="x", nbins= 17) # Set the number of ticks for both x and y axis
plt.locator_params(axis="y", nbins= 8)
plt.show()
