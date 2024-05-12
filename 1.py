import numpy as np
import matplotlib.pyplot as plt

# Coefficients
k1 = -2.685e-11
k2 = 1.539e-07
k3 = -0.0003261
k4 = 1

# Define the cubic function
def cubic_function(x):
    return k1*x**3 + k2*x**2 + k3*x + k4

# Generate x values
x_values = np.linspace(-500000, 500000, 400)  # Adjust the range and density as needed

# Compute y values
y_values = cubic_function(x_values)

# Plotting the function
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Cubic Function')
plt.title('Graph of the Cubic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.show()
