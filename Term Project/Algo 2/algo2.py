import time
import random
import string
import matplotlib.pyplot as plt

def lcs_recursive(X, Y, m, n):
    """Finds the length of the longest common subsequence using a pure recursive approach."""
    if m == 0 or n == 0:
        return 0
    elif X[m - 1] == Y[n - 1]:
        return 1 + lcs_recursive(X, Y, m - 1, n - 1)
    else:
        return max(lcs_recursive(X, Y, m, n - 1), lcs_recursive(X, Y, m - 1, n))

# Function to generate random test cases
def generate_random_string(length):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Performance testing
input_sizes = list(range(1, 21))  # Small sizes due to exponential complexity
runtimes = []

for size in input_sizes:
    X = generate_random_string(size)
    Y = generate_random_string(size)

    start_time = time.time()
    lcs_recursive(X, Y, len(X), len(Y))
    end_time = time.time()

    runtimes.append(end_time - start_time)

# Plot the results
plt.plot(input_sizes, runtimes, marker='o', linestyle='-')
plt.xlabel("Input Length (n)")
plt.ylabel("Time (seconds)")
plt.title("Runtime of Recursive LCS Algorithm")
plt.savefig("algo2.png")
plt.show()

