import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_random_numbers(n, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0, 1, n)

def ks_test(data):
    d, p_value = stats.kstest(data, 'uniform')
    return d, p_value

def chi_square_test(data, bins=10):
    observed, bin_edges = np.histogram(data, bins=bins)
    expected = len(data) / bins
    chi_square_stat = ((observed - expected) ** 2 / expected).sum()
    p_value = stats.chi2.sf(chi_square_stat, df=bins-1)
    return chi_square_stat, p_value

def plot_histogram(data, bins=10):
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title('Histogram of Generated Random Numbers')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# Generate random numbers
n = 1000
random_numbers = generate_random_numbers(n, seed=42)

# Perform K-S Test
d, ks_p_value = ks_test(random_numbers)
print(f"K-S Test: D-statistic = {d:.4f}, p-value = {ks_p_value:.4f}")

# Perform Chi-Square Test
chi_square_stat, chi_p_value = chi_square_test(random_numbers)
print(f"Chi-Square Test: Chi-Square Statistic = {chi_square_stat:.4f}, p-value = {chi_p_value:.4f}")

# Plot histogram
plot_histogram(random_numbers)
