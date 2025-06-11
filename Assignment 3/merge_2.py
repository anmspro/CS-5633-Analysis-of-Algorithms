"""
Sorting Algorithms Implementation for HW 3

- Quicksort (with optimizations: insertion sort for small inputs, randomized pivoting, median-of-three pivoting)
- Radix Sort (with best base selection)
- Insertion Sort
- Performance analysis and visualization
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
# Re-define sorting algorithms
def insertion_sort(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quicksort(arr, low=0, high=None, cutoff=10):
    if high is None:
        high = len(arr) - 1
    if low < high:
        if high - low < cutoff:
            insertion_sort(arr, low, high)
        else:
            pi = partition(arr, low, high)
            quicksort(arr, low, pi - 1, cutoff)
            quicksort(arr, pi + 1, high, cutoff)

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
    
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr, base=10):
    max_element = max(arr)
    exp = 1
    while max_element // exp > 0:
        counting_sort(arr, exp)
        exp *= base

def main():
    # Test sorting algorithms on random inputs
    num_tests = 10
    sizes = [random.randint(400, 5000) for _ in range(num_tests)]  # Array sizes between 500 and 5000
    quicksort_times = []
    radix_sort_times = []

    for size in sizes:
        arr = [random.randint(0, 20000) for _ in range(size)]
        arr1, arr2 = arr.copy(), arr.copy()

        # Measure Quicksort execution time
        start = time.perf_counter()
        quicksort(arr1, cutoff=10)
        end = time.perf_counter()
        quicksort_times.append(end - start)

        # Measure Radix Sort execution time
        start = time.perf_counter()
        radix_sort(arr2, base=100)
        end = time.perf_counter()
        radix_sort_times.append(end - start)

    # Calculate averages
    avg_quicksort_time = np.mean(quicksort_times)
    avg_radix_sort_time = np.mean(radix_sort_times)

    print(f"Average Quicksort Time: {avg_quicksort_time:.6f} s")
    print(f"Average Radix Sort Time: {avg_radix_sort_time:.6f} s")
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.bar(["Quicksort", "Radix Sort"], [avg_quicksort_time, avg_radix_sort_time])
    plt.ylabel("Average Execution Time (s)")
    plt.title("Comparison of Sorting Algorithm Execution Times")
    plt.savefig("Average")

if __name__ == "__main__":
    main()

def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)

# Function to generate random arrays
def generate_random_array(n):
    return [random.randint(0, 20000) for _ in range(n)]

# Define small input sizes to test
input_sizes = list(range(0, 5000, 5))  # Small inputs from 5 to 1000
insertion_times = []
quicksort_times = []

for size in input_sizes:
    arr1 = generate_random_array(size)
    arr2 = arr1.copy()

    # Measure Insertion Sort Time
    start = time.time()
    insertion_sort(arr1)
    insertion_times.append(time.time() - start)

    # Measure Quicksort Time
    start = time.time()
    quicksort(arr2)
    quicksort_times.append(time.time() - start)

# Find the cutoff point where Insertion Sort becomes faster
cutoff_index = np.argmax(np.array(insertion_times) < np.array(quicksort_times))
best_cutoff = input_sizes[cutoff_index]

print(f"Best Cutoff Point: {best_cutoff}")

# Plot comparison
plt.figure(figsize=(8, 5))
plt.plot(input_sizes, insertion_times, marker='o', linestyle='-', label="Insertion Sort")
plt.plot(input_sizes, quicksort_times, marker='s', linestyle='-', label="Pure Quicksort")
plt.axvline(best_cutoff, color='red', linestyle='dashed', label=f"Best Cutoff: {best_cutoff}")
plt.xlabel("Input Size (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Finding the Cutoff Between Insertion Sort and Quicksort")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("1")
plt.show()

# Function to generate random integers
def generate_random_array(n):
    return [random.randint(0, 20000) for _ in range(n)]

# Experimenting with different bases for Radix Sort

bases = [10, 50, 100, 256, 512, 1024]  # Different radix bases
radix_times = []

for base in bases:
    arr = generate_random_array(10000)  # Fixed input size
    start = time.time()
    radix_sort(arr, base)
    radix_times.append(time.time() - start)

# Plot comparison of Radix Sort bases
plt.figure(figsize=(8, 5))
plt.plot(bases, radix_times, marker='o', linestyle='-', label="Radix Sort Execution Time")
plt.xlabel("Radix Base")
plt.ylabel("Execution Time (seconds)")
plt.title("Finding the Best Base for Radix Sort")
plt.legend()
plt.grid(True)
plt.savefig("2")


# Identify the best base for Radix Sort
best_radix_base = bases[np.argmin(radix_times)]
best_radix_base
# Re-define necessary quicksort functions as execution state was reset
def randomized_partition(arr, low, high):
    """ Randomized Partition for Quicksort """
    rand_pivot = random.randint(low, high)
    arr[high], arr[rand_pivot] = arr[rand_pivot], arr[high]
    return partition(arr, low, high)

def median_of_three_partition(arr, low, high):
    """ Median-of-three Partition for Quicksort """
    mid = (low + high) // 2
    pivots = [(arr[low], low), (arr[mid], mid), (arr[high], high)]
    pivots.sort()
    median_index = pivots[1][1]
    arr[high], arr[median_index] = arr[median_index], arr[high]
    return partition(arr, low, high)

def quicksort(arr, low, high, cutoff):
    """ Quicksort with insertion sort optimization for small inputs """
    if high - low + 1 <= cutoff:
        insertion_sort(arr[low:high + 1])
        return
    if low < high:
        pivot = partition(arr, low, high)
        quicksort(arr, low, pivot - 1, cutoff)
        quicksort(arr, pivot + 1, high, cutoff)

def quicksort_random(arr, low, high, cutoff):
    """ Quicksort with Randomized Pivoting """
    if high - low + 1 <= cutoff:
        insertion_sort(arr[low:high + 1])
        return
    if low < high:
        pivot = randomized_partition(arr, low, high)
        quicksort_random(arr, low, pivot - 1, cutoff)
        quicksort_random(arr, pivot + 1, high, cutoff)

def quicksort_median(arr, low, high, cutoff):
    """ Quicksort with Median-of-Three Pivoting """
    if high - low + 1 <= cutoff:
        insertion_sort(arr[low:high + 1])
        return
    if low < high:
        pivot = median_of_three_partition(arr, low, high)
        quicksort_median(arr, low, pivot - 1, cutoff)
        quicksort_median(arr, pivot + 1, high, cutoff)

# Comparing different variations of Quicksort

input_sizes = [100, 500, 1000, 5000, 10000, 20000]  # Different input sizes
qs_times = []  # Standard Quicksort times
qs_random_times = []  # Randomized Pivot Quicksort times
qs_median_times = []  # Median-of-Three Quicksort times

best_cutoff = 20  # Optimal cutoff found earlier

for size in input_sizes:
    arr1 = [random.randint(0, 20000) for _ in range(size)]
    arr2 = arr1.copy()
    arr3 = arr1.copy()

    # Measure execution time for Standard Quicksort
    start = time.time()
    quicksort(arr1, 0, len(arr1) - 1, best_cutoff)
    qs_times.append(time.time() - start)

    # Measure execution time for Randomized Pivot Quicksort
    start = time.time()
    quicksort_random(arr2, 0, len(arr2) - 1, best_cutoff)
    qs_random_times.append(time.time() - start)

    # Measure execution time for Median-of-Three Pivot Quicksort
    start = time.time()
    quicksort_median(arr3, 0, len(arr3) - 1, best_cutoff)
    qs_median_times.append(time.time() - start)

# Plot comparison of Quicksort variations
plt.figure(figsize=(8, 5))
plt.plot(input_sizes, qs_times, marker='o', linestyle='-', label="Standard Quicksort")
plt.plot(input_sizes, qs_random_times, marker='s', linestyle='-', label="Randomized Pivot Quicksort")
plt.plot(input_sizes, qs_median_times, marker='^', linestyle='-', label="Median-of-Three Quicksort")
plt.xlabel("Input Size (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Comparison of Quicksort Variations")
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig("3")

# Identify the best Quicksort variation
qs_variations = ["Standard", "Randomized Pivot", "Median-of-Three"]
qs_best = qs_variations[np.argmin([qs_times[-1], qs_random_times[-1], qs_median_times[-1]])]
qs_best
# Generate random array function
def generate_random_array(n):
    return [random.randint(0, 20000) for _ in range(n)]

# Define input sizes for small and large cases
small_input_sizes = [10, 20, 50, 100, 200, 256]  # Small input sizes
large_input_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # Large input sizes (log scale)

# Store execution times
insertion_times_small, quicksort_times_small, radix_times_small = [], [], []
insertion_times_large, quicksort_times_large, radix_times_large = [], [], []

best_cutoff = 20  # Optimal cutoff found earlier
best_radix_base = 256  # Best base for Radix Sort

# Measure execution times for small inputs
for size in small_input_sizes:
    arr1 = generate_random_array(size)
    arr2 = arr1.copy()
    arr3 = arr1.copy()

    start = time.time()
    insertion_sort(arr1)
    insertion_times_small.append(time.time() - start)

    start = time.time()
    quicksort_median(arr2, 0, len(arr2) - 1, best_cutoff)
    quicksort_times_small.append(time.time() - start)

    start = time.time()
    radix_sort(arr3, best_radix_base)
    radix_times_small.append(time.time() - start)

# Measure execution times for large inputs
for size in large_input_sizes:
    arr1 = generate_random_array(size)
    arr2 = arr1.copy()
    arr3 = arr1.copy()

    start = time.time()
    insertion_sort(arr1)
    insertion_times_large.append(time.time() - start)

    start = time.time()
    quicksort_median(arr2, 0, len(arr2) - 1, best_cutoff)
    quicksort_times_large.append(time.time() - start)

    start = time.time()
    radix_sort(arr3, best_radix_base)
    radix_times_large.append(time.time() - start)

# Plot comparison for small inputs
plt.figure(figsize=(8, 5))
plt.plot(small_input_sizes, insertion_times_small, marker='o', linestyle='-', label="Insertion Sort")
plt.plot(small_input_sizes, quicksort_times_small, marker='s', linestyle='-', label="Median-of-Three Quicksort")
plt.plot(small_input_sizes, radix_times_small, marker='^', linestyle='-', label="Radix Sort (Base 256)")
plt.xlabel("Input Size (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Sorting Performance on Small Inputs")
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig("4")
# Plot comparison for large inputs (log scale)
plt.figure(figsize=(8, 5))
plt.plot(large_input_sizes, insertion_times_large, marker='o', linestyle='-', label="Insertion Sort")
plt.plot(large_input_sizes, quicksort_times_large, marker='s', linestyle='-', label="Median-of-Three Quicksort")
plt.plot(large_input_sizes, radix_times_large, marker='^', linestyle='-', label="Radix Sort (Base 256)")
plt.xscale("log", base=2)
plt.xlabel("Input Size (n) (log scale)")
plt.ylabel("Execution Time (seconds)")
plt.title("Sorting Performance on Large Inputs")
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig("5")