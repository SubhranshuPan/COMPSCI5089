import numpy as np

x = np.array([0, 1, 2])
y = np.array([1, 1, 3])
a = 2

def print_matrix(name, matrix):
    print(name, "=", matrix)


print_matrix("a", a)
print_matrix("x", x)
print_matrix("y", y)