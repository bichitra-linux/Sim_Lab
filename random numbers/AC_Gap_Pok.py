import numpy as np
from scipy.stats import chi2
import itertools

def load_random_numbers(filename):
    with open(filename, 'r') as file:
        numbers = [float(line.strip()) for line in file]
    return numbers

def autocorrelation_test(numbers, lag, alpha):
    n = len(numbers)
    mean = np.mean(numbers)
    numerator = sum((numbers[i] - mean) * (numbers[i + lag] - mean) for i in range(n - lag))
    denominator = sum((numbers[i] - mean) ** 2 for i in range(n))
    rho = numerator / denominator
    Z0 = rho * np.sqrt(n / (1 - rho ** 2))
    Z_alpha = np.sqrt(2) * erfinv(1 - alpha)
    if abs(Z0) < Z_alpha:
        return "Accepted"
    else:
        return "Rejected"

def gap_test(numbers, alpha, low, high):
    gaps = []
    gap = 0
    in_range = False
    for number in numbers:
        if low <= number <= high:
            if in_range:
                gaps.append(gap)
                gap = 0
            in_range = True
        else:
            if in_range:
                gap += 1
    k = len(gaps)
    mean_gap = np.mean(gaps)
    chi_square_stat = (k - mean_gap) ** 2 / mean_gap
    p_value = 1 - chi2.cdf(chi_square_stat, df=k-1)
    if p_value > alpha:
        return "Accepted"
    else:
        return "Rejected"

def poker_test(numbers, alpha):
    n = len(numbers)
    counts = {comb: 0 for comb in itertools.combinations_with_replacement('0123456789', 5)}
    for number in numbers:
        digits = str(number).replace('.', '')[:5]
        counts[tuple(sorted(digits))] += 1
    expected = n / len(counts)
    chi_square_stat = sum((count - expected) ** 2 / expected for count in counts.values())
    p_value = 1 - chi2.cdf(chi_square_stat, df=len(counts) - 1)
    if p_value > alpha:
        return "Accepted"
    else:
        return "Rejected"

if __name__ == "__main__":
    filename = input("Enter the filename containing random numbers: ")
    numbers = load_random_numbers(filename)

    while True:
        print("\nSelect a test to perform:")
        print("1) Autocorrelation Test")
        print("2) Gap Test")
        print("3) Poker Test")
        print("4) Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            lag = int(input("Enter the lag (k): "))
            alpha = float(input("Enter the significance level (alpha): "))
            result = autocorrelation_test(numbers, lag, alpha)
            print(f"Autocorrelation Test result: {result}")
        elif choice == 2:
            low = float(input("Enter the lower bound: "))
            high = float(input("Enter the upper bound: "))
            alpha = float(input("Enter the significance level (alpha): "))
            result = gap_test(numbers, alpha, low, high)
            print(f"Gap Test result: {result}")
        elif choice == 3:
            alpha = float(input("Enter the significance level (alpha): "))
            result = poker_test(numbers, alpha)
            print(f"Poker Test result: {result}")
        elif choice == 4:
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select a valid option.")
