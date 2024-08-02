import numpy as np
from tabulate import tabulate
from scipy.stats import chisquare

np.random.seed(42)
random_numbers = np.random.uniform(0, 1, 1000)

num_bins = 10

observed_frequencies, _ = np.histogram(random_numbers, bins=num_bins, range=(0, 1))
expected_frequencies = np.ones(num_bins) * (len(random_numbers) / num_bins)
chi_statistic, p_value = chisquare(observed_frequencies, expected_frequencies)

print(f"Chi-Square Statistic: {chi_statistic}")
print(f"P-value: {p_value}")

alpha = 0.05
status = 'Pass' if p_value > alpha else 'Fail'
print(f"Test Status: {status}")

table = [
    ['Bin', 'Range', 'Observed Frequency', 'Expected Frequency']
]

for i, (obs, exp) in enumerate(zip(observed_frequencies, expected_frequencies)):
    low = i / num_bins
    high = (i + 1) / num_bins
    table.append([i + 1, f'({low:.1f}, {high:.1f})', obs, exp])

print(tabulate(table, headers='firstrow', tablefmt='grid'))
