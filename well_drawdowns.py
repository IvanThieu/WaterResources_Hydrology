# Ivan Thieu
# 11/16/18

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# Problem 2 Infinite Series
# Define constants
Q = 650*0.133681 # Q in ft3/min
S = 10**-4
T = 10**4*0.133681/24/60 # T in ft2/min
r1_sq = 1600
r2_sq = 6400
r3_sq = 46400
r4_sq = 41600
time = [.1, .2, .5, 1, 10, 100, 500, 1000] # Times we are interested in

# Calculate u given an r2 and time
def calc_u(ri_2, time_i):
    return ri_2 * S/ (4*T*time_i)

# Calculate well function given a u value
def calc_wu(u):
    val = -0.5772 - np.log(u) # np.log instead of math.log to prevent type error
    for i in range(1, 101):
        val = val - u**i/(i*math.factorial(i)) if i % 2 == 0 else val + u**i/(i*math.factorial(i))
    return val

# Calculate draw down given the well functions
def calc_s(wu_1, wu_2, wu_3, wu_4):
    return Q/(4*math.pi*T) * (wu_1 - wu_2 - wu_3 + wu_4)

# Initialize dataframe
df = pd.DataFrame(data= {"time": time, "w_u1": np.nan, "w_u2": np.nan,
                         "w_u3": np.nan, "w_u4": np.nan, "s": np.nan},
                  columns= ["time", "w_u1", "w_u2", "w_u3", "w_u4", "s"])

# Calculate the u and w(u) values
df.w_u1 = calc_wu(calc_u(r1_sq, df.time))
df.w_u2 = calc_wu(calc_u(r2_sq, df.time))
df.w_u3 = calc_wu(calc_u(r3_sq, df.time))
df.w_u4 = calc_wu(calc_u(r4_sq, df.time))
df.s = calc_s(df.w_u1, df.w_u2, df.w_u3, df.w_u4)

# Print out data frame to view all of the values
# print(df)

# Use more time intervals for graphing for a smooth plot
expanded_df = pd.DataFrame(data= {"time": np.arange(.1, 1000, .1), "w_u1": np.nan, "w_u2": np.nan,
                         "w_u3": np.nan, "w_u4": np.nan, "s": np.nan},
                  columns= ["time", "w_u1", "w_u2", "w_u3", "w_u4", "s"])
expanded_df.w_u1 = calc_wu(calc_u(r1_sq, expanded_df.time))
expanded_df.w_u2 = calc_wu(calc_u(r2_sq, expanded_df.time))
expanded_df.w_u3 = calc_wu(calc_u(r3_sq, expanded_df.time))
expanded_df.w_u4 = calc_wu(calc_u(r4_sq, expanded_df.time))
expanded_df.s = calc_s(expanded_df.w_u1, expanded_df.w_u2, expanded_df.w_u3, expanded_df.w_u4)

# Plot
plt.figure()
plt.scatter(df.time, df.s) # Scatter plot
plt.semilogx(expanded_df.time, expanded_df.s) # Semilog plot
plt.grid(which= "both")
plt.title("Drawdown over Time")
plt.xlabel("Time (Min)")
plt.ylabel("Drawdown (ft)")
plt.show()

# Problem 3 Theis Solution
# Input data and set up dataframe
r = 824
time = [3, 5, 8, 12, 20, 24, 30, 38, 47, 50, 60, 70, 80, 90, 100, 130, 160, 200, 206,
        320, 380, 500]
s = [.3, .7, 1.3, 2.1, 3.2, 3.6, 4.1, 4.7, 5.1, 5.3, 5.7, 6.1, 6.3, 6.7, 7, 7.5, 8.3, 8.5,
     9.2, 9.7, 10.2, 10.9]
df = pd.DataFrame(data= {"time": time, "r2/t": np.nan, "s": s}, columns= ["time", "r2/t", "s"])
df["r2/t"] = r**2 / df.time

# Plot scatter plot in log log scale
plt.figure()
plt.scatter(df["r2/t"], df.s)
plt.xscale("log")
plt.yscale("log")
plt.title("Drawdown over $\\frac{r^2}{t}$")
plt.xlabel("$\\frac{r^2}{t}$ ($\\frac{ft^2}{min}$)")
plt.ylabel("s (ft)")
plt.ylim(.01, 20)
plt.xlim(10, 10**6)
plt.show()

# Problem 4 Walton's Type Curve
# Set up dataframe
time = [2.53, 3.11, 8.33, 11.63, 15.73, 19.93, 25.03, 31.73, 39.73, 47.73, 52.03,
        63.73, 79.73, 95.73, 103.73, 120.10]
s = [0.124, 0.145, 0.258, 0.289, 0.323, 0.341, 0.370, 0.390, 0.415, 0.430, 0.439,
     0.459, 0.477, 0.493, 0.502, 0.510]
df = pd.DataFrame(data={"time": time, "s": s}, columns= ["time", "s"])

# Plot
plt.figure()
plt.scatter(df.time, df.s, s=2)
plt.xscale("log")
plt.yscale("log")
plt.title("Drawdown over Time")
plt.xlabel("Time (min)")
plt.ylabel("Drawdown (ft)")
plt.ylim(.01, 10)
plt.xlim(0.1, 10**4)
plt.show()

# Problem 4 Walton's Type Curve (Well #2)
# Set up dataframe
time = [8.15, 11.40, 15.00, 19.00, 27.00, 40.00, 55.00, 63.00, 79.00, 95.00]
s = [0.063, 0.089, 0.110, 0.130, 0.161, 0.182, 0.197, 0.207, 0.213, 0.218]
df = pd.DataFrame(data= {"time": time, "s": s}, columns= ["time", "s"])

# Plot
plt.figure()
plt.scatter(df.time, df.s, s=2)
plt.xscale("log")
plt.yscale("log")
plt.title("Drawdown over Time")
plt.xlabel("Time (min)")
plt.ylabel("Drawdown (ft)")
plt.ylim(.01, 10)
plt.xlim(0.1, 10**4)
plt.show()

# Problem 5 Jacob's Method
# Set up dataframe
time = list(map(lambda x: x*60, [1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24])) # Convert to min
s = [0.6, 1.4, 2.4, 2.9, 3.3, 4.0, 5.2, 6.2, 7.5, 9.1, 10.5]
df = pd.DataFrame(data= {"time": time, "s": s}, columns= ["time", "s"])

# Plot
plt.figure()
plt.scatter(df.time, df.s)
plt.xscale("log")
plt.title("Drawdown over Time (Semi log)")
plt.xlabel("Time (min)")
plt.ylabel("Drawdown (ft)")

# Best fit line
best_fit = np.polyfit(np.log(df.time), df.s, 1) # Best fit slope and intercept using least squares
plt.plot(df.time, best_fit[0]*np.log(df.time) + best_fit[1])
plt.text(225, 7.5, "s = " + str(round(float(best_fit[0]), 2)) + "ln(t) " +
         str(round(float(best_fit[1]), 2)))
plt.show()

# Getting drawdown for 1 log cycle (t = 100 and t = 1000 min)
# np.log and math.log are natural logs. Using natural log and log10 does not affect the final answers
s_100 = best_fit[0] * math.log(100) + best_fit[1]
s_1000 = best_fit[0] * math.log(1000) + best_fit[1]
print(s_100)
print(s_1000)

# Getting time at drawdown = 0 to solve for S
t_0 = math.exp((0 - best_fit[1])/best_fit[0])
print(t_0)