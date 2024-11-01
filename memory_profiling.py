from memory_profiler import profile

@profile
def memory_hungry_function():
    # Create a large list
    big_list = [i * i for i in range(100000)]
    # Do some operations
    filtered_list = [x for x in big_list if x % 2 == 0]
    return sum(filtered_list)

if __name__ == "__main__":
    memory_hungry_function()