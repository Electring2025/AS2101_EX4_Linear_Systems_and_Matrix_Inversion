"""
AS2101 EX4 -- Linear Systems and Matrix Inversion
Solves Ax = b, analyses the system, and writes all results to a LaTeX file.
"""

import numpy as np
import sympy as sp


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def matrix_to_latex(mat, fmt=".4f"):
    """Convert a 2-D numpy array (or 1-D vector) to a LaTeX bmatrix string."""
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    rows = []
    for row in mat:
        entries = []
        for v in row:
            # Try to render as a nice fraction via sympy
            s = sp.nsimplify(v, rational=True, tolerance=1e-8)
            entries.append(sp.latex(s))
        rows.append(" & ".join(entries))
    body = " \\\\\n    ".join(rows)
    return f"\\begin{{bmatrix}}\n    {body}\n\\end{{bmatrix}}"


def scalar_to_latex(val):
    """Render a scalar nicely (fraction if possible)."""
    s = sp.nsimplify(val, rational=True, tolerance=1e-8)
    return sp.latex(s)


def decimal_to_latex(val, fmt=".4f"):
    """Render a scalar as a decimal number (for condition numbers, residuals, etc.)."""
    if abs(val) >= 1e6 or (abs(val) < 1e-4 and val != 0):
        return f"{val:.4e}"
    return f"{val:{fmt}}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Input
# ═══════════════════════════════════════════════════════════════════════════════

def read_input():
    """Read matrix A and vector b from the console."""
    print("=" * 60)
    print("  Linear System Solver -- A x = b")
    print("=" * 60)

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis & Solve
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(A, b):
    """Analyse and solve the linear system. Returns a dict of results."""
    m, n = A.shape
    res = {}

    # ── dimensions ──
    res["m"], res["n"] = m, n

    # ── ranks ──
    rank_A = np.linalg.matrix_rank(A)
    Ab = np.column_stack([A, b])
    rank_Ab = np.linalg.matrix_rank(Ab)
    res["rank_A"] = rank_A
    res["rank_Ab"] = rank_Ab

    # ── system type ──
    if m > n:
        res["system_type"] = "Overdetermined"
    elif m < n:
        res["system_type"] = "Underdetermined"
    else:
        res["system_type"] = "Square (m = n)"

    # ── consistency ──
    consistent = (rank_A == rank_Ab)
    res["consistent"] = consistent

    # ── determinant & singularity (square only) ──
    is_square = (m == n)
    res["is_square"] = is_square
    if is_square:
        det_A = np.linalg.det(A)
        res["det_A"] = det_A
        res["singular"] = np.isclose(det_A, 0)
    else:
        res["det_A"] = None
        res["singular"] = None

    # ── condition number ──
    res["cond"] = np.linalg.cond(A)

    # ── solution ──
    if is_square and not res["singular"]:
        # unique solution via direct solve
        x = np.linalg.solve(A, b)
        res["solution_type"] = "Unique"
        res["x"] = x
    elif consistent and rank_A == n:
        # overdetermined but consistent with full column rank → unique (lstsq)
        x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        res["solution_type"] = "Unique (least-squares, consistent)"
        res["x"] = x
    elif not consistent:
        # inconsistent → least-squares approximate solution
        x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        res["solution_type"] = "Least-squares (inconsistent system)"
        res["x"] = x
        res["residual"] = np.linalg.norm(A @ x - b)
    else:
        # consistent, rank_A < n → infinitely many solutions
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        res["solution_type"] = "Infinite (minimum-norm particular solution shown)"
        res["x"] = x
        # null-space basis via sympy
        A_sym = sp.Matrix(A.tolist())
        ns = A_sym.nullspace()
        res["nullspace"] = [np.array(v).astype(float).flatten() for v in ns]

    # ── inverse (square & non-singular only) ──
    if is_square and not res["singular"]:
        res["A_inv"] = np.linalg.inv(A)
    else:
        res["A_inv"] = None

    # store matrices for LaTeX
    res["A"] = A
    res["b"] = b

    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  LaTeX generation
# ═══════════════════════════════════════════════════════════════════════════════

def build_latex(res):
    """Build a complete, compilable LaTeX document string."""

    A_tex = matrix_to_latex(res["A"])
    b_tex = matrix_to_latex(res["b"])
    x_tex = matrix_to_latex(res["x"])

    # variable vector (x1, x2, …)
    n = res["n"]
    x_vars = " \\\\\n    ".join([f"x_{{{i + 1}}}" for i in range(n)])
    xvec_tex = f"\\begin{{bmatrix}}\n    {x_vars}\n\\end{{bmatrix}}"

    # augmented matrix
    Ab_tex = matrix_to_latex(np.column_stack([res["A"], res["b"]]))

    sections = []

    # ── Section 1: Input ──
    sections.append(r"""
\section{Input System}
The system of linear equations in matrix form $A\,\mathbf{x} = \mathbf{b}$:

\[
""" + A_tex + r"\;" + xvec_tex + r" = " + b_tex + r"""
\]

Augmented matrix $[A \mid \mathbf{b}]$:
\[
""" + Ab_tex + r"""
\]
""")

    # ── Section 2: Properties ──
    props = r"""
\section{Matrix Properties}
\begin{itemize}
    \item Dimensions of $A$: $""" + f"{res['m']} \\times {res['n']}" + r"""$
    \item System type: \textbf{""" + res["system_type"] + r"""}
    \item $\mathrm{rank}(A) = """ + str(res["rank_A"]) + r"""$
    \item $\mathrm{rank}([A \mid \mathbf{b}]) = """ + str(res["rank_Ab"]) + r"""$
    \item Condition number $\kappa(A) = """ + decimal_to_latex(res["cond"]) + r"""$"""

    if res["is_square"]:
        props += r"""
    \item $\det(A) = """ + scalar_to_latex(res["det_A"]) + r"""$
    \item Matrix is \textbf{""" + ("Singular" if res["singular"] else "Non-singular") + r"""}"""
    props += r"""
\end{itemize}
"""
    sections.append(props)

    # ── Section 3: Consistency ──
    if res["consistent"]:
        verdict = (r"The system is \textbf{consistent} since "
                   r"$\mathrm{rank}(A) = \mathrm{rank}([A|\mathbf{b}]) = "
                   + str(res["rank_A"]) + r"$.")
    else:
        verdict = (r"The system is \textbf{inconsistent} since "
                   r"$\mathrm{rank}(A) = " + str(res["rank_A"])
                   + r" \neq \mathrm{rank}([A|\mathbf{b}]) = "
                   + str(res["rank_Ab"]) + r"$.")

    sections.append(r"""
\section{Consistency Check}
""" + verdict + "\n")

    # ── Section 4: Solution ──
    sol_sec = r"""
\section{Solution}
\textbf{""" + res["solution_type"] + r"""}

\[
\mathbf{x} = """ + x_tex + r"""
\]
"""
    if "residual" in res:
        sol_sec += r"""
The least-squares residual $\| A\mathbf{x} - \mathbf{b} \|_2 = """ + decimal_to_latex(res["residual"]) + r"""$.
"""
    if "nullspace" in res and res["nullspace"]:
        sol_sec += r"""
\subsection*{General Solution}
The general solution is $\mathbf{x} = \mathbf{x}_p + c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots$ where $\mathbf{x}_p$ is the particular solution above and the null-space basis vectors are:
"""
        for i, v in enumerate(res["nullspace"]):
            sol_sec += r"""
\[
\mathbf{v}_{""" + str(i + 1) + r"} = " + matrix_to_latex(v) + r"""
\]
"""
    sections.append(sol_sec)

    # ── Section 5: Inverse ──
    if res["A_inv"] is not None:
        inv_tex = matrix_to_latex(res["A_inv"])
        inv_sec = r"""
\section{Inverse of $A$}
Since $A$ is square and non-singular, $A^{-1}$ exists:
\[
A^{-1} = """ + inv_tex + r"""
\]
"""
    else:
        if res["is_square"]:
            reason = "$A$ is singular ($\\det(A) = 0$), so $A^{-1}$ does not exist."
        else:
            reason = "$A$ is not square, so $A^{-1}$ does not exist."
        inv_sec = r"""
\section{Inverse of $A$}
""" + reason + "\n"
    sections.append(inv_sec)

    # ── Assemble document ──
    doc = r"""\documentclass[12pt, a4paper]{article}
\usepackage{amsmath, amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{parskip}

\title{AS2101 EX4 --- Linear Systems and Matrix Inversion}
\author{Computed Results}
\date{\today}

\begin{document}
\maketitle
""" + "\n".join(sections) + r"""
\end{document}
"""
    return doc


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    A, b = read_input()

    print("\n...  Analysing system ...\n")
    res = analyse(A, b)

    # ── Console summary ──
    print("-" * 60)
    print(f"  Dimensions          : {res['m']} x {res['n']}")
    print(f"  System type         : {res['system_type']}")
    print(f"  rank(A)             : {res['rank_A']}")
    print(f"  rank([A|b])         : {res['rank_Ab']}")
    print(f"  Consistent          : {'Yes' if res['consistent'] else 'No'}")
    if res["is_square"]:
        print(f"  det(A)              : {res['det_A']:.6g}")
        print(f"  Singular            : {'Yes' if res['singular'] else 'No'}")
    print(f"  Condition number    : {res['cond']:.6g}")
    print(f"  Solution type       : {res['solution_type']}")
    print(f"  x                   : {res['x']}")
    if res["A_inv"] is not None:
        print(f"  A^(-1) exists       : Yes")
    else:
        print(f"  A^(-1) exists       : No")
    print("-" * 60)

    # ── Write LaTeX ──
    tex = build_latex(res)
    out_path = "output.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\n[OK] LaTeX output written to -> {out_path}")
    print("     Compile with:  pdflatex output.tex\n")


if __name__ == "__main__":
    main()

