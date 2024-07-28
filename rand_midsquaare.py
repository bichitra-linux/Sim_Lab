def mid_square(seed, num_digits):
    # Determine the number of digits in the seed
    seed_length = len(str(seed))
    
    if seed_length != num_digits:
        raise ValueError("Seed must have the specified number of digits")
    
    # List to store generated random numbers
    random_numbers = []

    # Loop to generate random numbers
    for _ in range(10):  # generate 10 random numbers
        # Square the seed
        squared = seed ** 2

        # Convert squared number to string to extract the middle digits
        squared_str = str(squared).zfill(num_digits * 2)  # Ensure the string has the proper length

        # Calculate the starting index for middle digits extraction
        mid_start = (len(squared_str) - num_digits) // 2
        mid_end = mid_start + num_digits

        # Extract the middle digits and form the new seed
        seed = int(squared_str[mid_start:mid_end])

        # Append the new seed to the list of random numbers
        random_numbers.append(seed)

    return random_numbers

# Example usage
seed = 2504  # Seed value with 4 digits
num_digits = 4  # Number of digits to consider in each step
random_numbers = mid_square(seed, num_digits)
print(random_numbers)
