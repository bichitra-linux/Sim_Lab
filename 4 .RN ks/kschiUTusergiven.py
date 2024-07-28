import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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
    plt.title('Histogram of Input Random Numbers')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def main():
    # User input for random numbers
    print("Enter your list of random numbers separated by spaces:")
    user_input = input()
    random_numbers = list(map(float, user_input.split()))
    
    # Perform K-S Test
    d, ks_p_value = ks_test(random_numbers)
    print(f"K-S Test: D-statistic = {d:.4f}, p-value = {ks_p_value:.4f}")

    # Perform Chi-Square Test
    chi_square_stat, chi_p_value = chi_square_test(random_numbers)
    print(f"Chi-Square Test: Chi-Square Statistic = {chi_square_stat:.4f}, p-value = {chi_p_value:.4f}")

    # Plot histogram
    plot_histogram(random_numbers)

if __name__ == "__main__":
    main()
