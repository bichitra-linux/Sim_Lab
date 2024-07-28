import random
from datetime import datetime

def generate_random_numbers(quantity):
    # Generate filename with current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"random_{current_time}.txt"
    
    with open(filename, 'w') as file:
        for _ in range(quantity):
            random_number = round(random.uniform(0, 1), 5)
            file.write(f"{random_number}\n")
    
    return filename

if __name__ == "__main__":
    # Specify the quantity of random numbers
    quantity = 100  # Replace with desired quantity

    filename = generate_random_numbers(quantity)
    print(f"{quantity} random numbers have been written to {filename}")
