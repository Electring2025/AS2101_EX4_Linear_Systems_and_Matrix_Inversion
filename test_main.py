"""Quick automated test -- feeds a 3x3 system into main.analyse() and writes LaTeX."""
import numpy as np
from main import analyse, build_latex

# Test 1: 3x3 non-singular system
print("=" * 60)
print("TEST 1: 3x3 Non-singular system")
print("=" * 60)
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

res = analyse(A, b)
for k, v in res.items():
    if isinstance(v, np.ndarray) and v.ndim == 2:
        print(f"  {k}:\n{v}")
    else:
        print(f"  {k}: {v}")

tex = build_latex(res)
with open("output_test1.tex", "w", encoding="utf-8") as f:
    f.write(tex)
print("\n  -> output_test1.tex written\n")

# Test 2: 2x3 underdetermined consistent system
print("=" * 60)
print("TEST 2: 2x3 Underdetermined system")
print("=" * 60)
A2 = np.array([[1, 2, 3],
               [4, 5, 6]], dtype=float)
b2 = np.array([1, 2], dtype=float)

res2 = analyse(A2, b2)
for k, v in res2.items():
    if isinstance(v, np.ndarray) and v.ndim == 2:
        print(f"  {k}:\n{v}")
    else:
        print(f"  {k}: {v}")

tex2 = build_latex(res2)
with open("output_test2.tex", "w", encoding="utf-8") as f:
    f.write(tex2)
print("\n  -> output_test2.tex written\n")

# Test 3: 3x2 overdetermined inconsistent system
print("=" * 60)
print("TEST 3: 3x2 Overdetermined (inconsistent)")
print("=" * 60)
A3 = np.array([[1, 1],
               [1, 2],
               [1, 3]], dtype=float)
b3 = np.array([1, 3, 2], dtype=float)

res3 = analyse(A3, b3)
for k, v in res3.items():
    if isinstance(v, np.ndarray) and v.ndim == 2:
        print(f"  {k}:\n{v}")
    else:
        print(f"  {k}: {v}")

tex3 = build_latex(res3)
with open("output_test3.tex", "w", encoding="utf-8") as f:
    f.write(tex3)
print("\n  -> output_test3.tex written\n")

# Test 4: 3x3 singular system
print("=" * 60)
print("TEST 4: 3x3 Singular system")
print("=" * 60)
A4 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]], dtype=float)
b4 = np.array([1, 2, 3], dtype=float)

res4 = analyse(A4, b4)
for k, v in res4.items():
    if isinstance(v, np.ndarray) and v.ndim == 2:
        print(f"  {k}:\n{v}")
    else:
        print(f"  {k}: {v}")

tex4 = build_latex(res4)
with open("output_test4.tex", "w", encoding="utf-8") as f:
    f.write(tex4)
print("\n  -> output_test4.tex written\n")

print("ALL TESTS PASSED")
