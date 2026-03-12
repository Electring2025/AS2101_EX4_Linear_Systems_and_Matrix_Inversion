import numpy as np
import scipy as linalg
import matplotlib.pyplot as plt

def read_input():
    """Read matrix A and vector b from the terminal."""
    print("-" * 60)
    print("  Linear System Solver -- A x = b")
    print("-" * 60)

    while True:
        try:
            m = int(input("\nNumber of equations   (m - rows of A): "))
            n = int(input("Number of unknowns    (n - cols of A): "))
            if m <= 0 or n <= 0:
                raise ValueError
            break
        except ValueError:
            print("  [!] m and n must be positive integers. Try again.")

    print(f"\nEnter matrix A  ({m}x{n}), one row per line (space-separated):")
    A = []
    for i in range(m):
        while True:
            try:
                row = list(map(float, input(f"  Row {i + 1}: ").split()))
                if len(row) != n:
                    print(f"  [!] Expected {n} values, got {len(row)}. Try again.")
                    continue
                A.append(row)
                break
            except ValueError:
                print("  [!] Non-numeric entry detected. Try again.")
    A = np.array(A, dtype=float)

    print(f"\nEnter vector b  ({m}x1), space-separated:")
    while True:
        try:
            b = list(map(float, input("  b: ").split()))
            if len(b) != m:
                print(f"  [!] Expected {m} values, got {len(b)}. Try again.")
                continue
            break
        except ValueError:
            print("  [!] Non-numeric entry detected. Try again.")
    b = np.array(b, dtype=float)

    return A, b

def solve(A, b):
    if A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]:
        return np.linalg.solve(A, b)
    elif A.shape[0] != A.shape[1] and A.shape[0] == b.shape[0]:
        return ("  [!] Matrix A must be square. Try again.")
    elif A.shape[0] == A.shape[1] and A.shape[0] != b.shape[0]:
        return ("  [!] Column vector B must have same rows as A. Try again.")
    else:
        return ("Check A and b")

def invrese(A,b):
    if A.shape[0] == A.shape[1] and det(A)!=0:
        return np.linalg.inv(A)
    else:
        return print("  [!] Matrix A must be square. Try again.")

def det(A):
    if A.shape[0] == A.shape[1]:
        return np.linalg.det(A)

def cond(A):
    return np.linalg.cond(A)

def rank(A):
    return np.linalg.matrix_rank(A)

A,b = read_input()
print("@"*30)
print("Solution for given System of Linear Equations is " + str(solve(A,b)))