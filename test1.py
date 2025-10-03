def multiply_loop(baseNumber, numberOfLoops):
    # Create a sum variable
    sum = 0

    # Create a for loop to add baseNumber to sum
    for _ in range(numberOfLoops):
        sum += baseNumber
    return f"The result of multiplying {baseNumber} by {numberOfLoops} is: {sum}"

print(multiply_loop(5, 4))