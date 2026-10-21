# Lab 1 Prep Pack (assessed lab, 9–10am)

Covers: lab-rule fixes, Lecture 1 code drills, Lecture 2 walkthrough, tonight's schedule.

---

## Part 0: Fix these before the lab

| Issue | Where | Fix |
|---|---|---|
| **`import numpy as np` inside an answer cell** | Lab 0 Task 1 cell. The email says no import statements in answers, because they can break the grader. | `np` is already imported in the setup cell, so just write `a = np.array([2, -1, 4])`. |
| Hidden tests use **different inputs** | Lab 0 Task 3 (`inner`) | Write general functions of their arguments. Never hard-code numbers from the visible test. |
| Shape asserts | Lab 0 tests such as `a.shape == (3,)` | A "vector" is 1D `(n,)`, not `(n,1)`. Read the assert before you decide the shape. |
| Procedure | Email and Lab 0 notes | Only edit `# YOUR CODE HERE` cells, and replace **both** lines, including `raise NotImplementedError()`. Don't add, delete or copy cells. Then Restart & Run All, then submit to Moodle **from your own slot**. |

---

## Part 1: Lecture 1 code you could be asked to write (about 60 min)

The Lab 0 tasks were all Lecture 1: vectors, `@`, norm, `M @ x`, mean vector. So Lab 1 almost certainly extends these. Cover the right-hand column and retype each one from memory in a **scratch notebook**, not the lab file.

| Task | Idiomatic answer |
|---|---|
| Weighted sum / lerp | `(1 - a) * x + a * y` |
| L1, L2, L∞, L0 | `np.sum(np.abs(x))`, `np.sqrt(np.sum(x**2))`, `np.max(np.abs(x))`, `np.count_nonzero(x)` |
| Same, via NumPy | `np.linalg.norm(x, ord)` with `ord` = 1, 2, `np.inf`, 0 |
| Distance between x and y | `np.linalg.norm(x - y)` |
| Unit vector | `x / np.linalg.norm(x)` |
| Normalise every **row** | `X / np.linalg.norm(X, axis=1, keepdims=True)` |
| Dot product | `x @ y` or `np.dot(x, y)` |
| Cosine similarity | `(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))` |
| Mean vector | `X.mean(axis=0)` |
| Centre the data | `X - X.mean(axis=0)` |
| Covariance (N−1) | `Xc = X - X.mean(axis=0); Xc.T @ Xc / (X.shape[0] - 1)`, which equals `np.cov(X, rowvar=False)` |
| Matrix–vector, matrix–matrix | `A @ x`, `A @ B` (**never `*`**) |
| Outer product | `np.outer(x, y)` or `x.reshape(-1,1) @ y.reshape(1,-1)` |
| Identity / diagonal / zeros | `np.eye(n)`, `np.diag(v)`, `np.zeros((m, n))` |
| Lower triangular | `np.tri(n)` (upper is `.T`) |
| Clockwise rotation by θ | `np.array([[c, s], [-s, c]])`, with `c = np.cos(th)`, `s = np.sin(th)` |
| "Do A, then B" | `B @ A` |
| Float equality | `np.allclose(a, b)`, never `==` |

**Possible kNN-style task** (the lecture covers kNN conceptually, so it could appear):

```python
d = np.linalg.norm(X - q, axis=1)           # distance from query q to every row
idx = np.argsort(d)[:k]                     # indices of k nearest
pred = np.bincount(labels[idx]).argmax()    # majority label (labels = non-negative ints)
```

### Traps that cost marks

1. `axis=0` gives one value per **column** (the mean vector). `axis=1` gives one per **row**.
2. `*` is elementwise. `@` is the matrix product. `A**2` is also elementwise.
3. Shape check `(p×q)(q×r)`: the middle numbers must match.
4. `np.var` divides by N. `np.cov` divides by N−1.
5. `np.linalg.norm(X, axis=1)` returns `(N,)`, so use `keepdims=True` before dividing a matrix by it.
6. `np.cov(X)` needs `rowvar=False` when variables are columns.
7. L0 counts non-zero entries (it is not a true norm).
8. `np.tri(n)` is *lower* triangular (the lecture's variable names in cell 140 are swapped).

---

## Part 2: Lecture 2 walkthrough (about 45 min)

**Is it in tomorrow's lab? My read, not confirmed.** The notebook header says "Lecture Week 3", Lecture 1 says "Week 1", and Lab 0 said it would draw on Lecture 1 plus self-study. Lecture 2 even says whitening and PCA "will be covered in this week's lab", which suggests they belong to a later lab. So weight Lecture 1 heavily, and treat these cards as insurance. Check the Moodle lab description if you can.

Each card gives the 5-year-old idea, then the math and code.

### 1. Graphs as matrices

- **Idea:** a map of roads becomes a grid of numbers. A 1 at row i, column j means "there is a road from i to j".
- **Code:**

  ```python
  out_deg = A.sum(axis=1)      # row sums = edges leaving
  in_deg  = A.sum(axis=0)      # column sums = edges entering
  np.allclose(A, A.T)          # symmetric means undirected
  A + A.T                      # make undirected
  np.diag(A)                   # non-zero means self-loops
  A / A.sum(axis=1, keepdims=True)   # rows sum to 1 ("conserving")
  ```

- **Trap:** the comments in lecture cell 14 have rows and columns swapped. `axis=0` sums *down* the columns (in-flow).
- **Flow:** `x_next = x @ A` (row vector on the left). After t days: `x @ np.linalg.matrix_power(A, t)`.
- **Row sum meaning:** >1 is a source, <1 is a sink, =1 conserves.

### 2. Matrix powers

- **Idea:** do the same machine again and again.
- `np.linalg.matrix_power(A, n)` for square A. The lecture's `powm` loops `B = A @ B` starting from `np.eye`.
- **Trap:** `A**n` is elementwise.

### 3. Eigenvectors and eigenvalues

- **Idea:** most arrows get turned *and* stretched by a matrix. Eigenvectors are the special arrows that are only stretched. The eigenvalue is the stretch factor.
- **Math:** `A v = λ v`. Square matrices only.
- **Hand example** (from cell 58): `A = [[1, .5], [.5, 1]]` has λ = 1.5 with v ∝ [1, 1], and λ = 0.5 with v ∝ [1, −1].
- **Code:**

  ```python
  evals, evecs = np.linalg.eig(A)     # eigenvectors are the COLUMNS: evecs[:, i]
  evals, evecs = np.linalg.eigh(A)    # for symmetric A: stabler, orthogonal, ascending order
  order = np.argsort(-np.abs(evals))  # sort by size, largest first
  evals, evecs = evals[order], evecs[:, order]
  A @ evecs[:, 0]                     # should equal evals[0] * evecs[:, 0]
  ```

- **Power iteration** (leading eigenvector):

  ```python
  for _ in range(n):
      x = A @ x
      x = x / np.linalg.norm(x, np.inf)
  ```

  The eigenvalue is `(A @ x) / x`, taken elementwise, and it is the same in every entry.

### 4. Eigendecomposition and low-rank approximation

- **Idea:** a matrix is a recipe of "directions" and "stretches". Keep the big stretches, drop the tiny ones, and you keep most of the matrix.
- **Math:** for symmetric A, `A = Q Λ Qᵀ`.
- **Code:**

  ```python
  A_rec = Q @ np.diag(evals) @ Q.T
  evals_trunc = np.where(np.abs(evals) < 0.2, 0, evals)   # drop small ones
  ```

- The **eigenspectrum** is the sorted |λ|. One big value and many small ones means the matrix is easy to approximate.

### 5. PCA

- **Idea:** find the direction in which the data cloud is longest.
- **Math:** principal components are the eigenvectors of the covariance matrix. Component length is √λ.
- **Code:**

  ```python
  C = np.cov(X, rowvar=False)
  evals, evecs = np.linalg.eigh(C)       # ascending, so take the LAST columns for the biggest
  Z = (X - X.mean(axis=0)) @ evecs[:, ::-1][:, :k]   # project onto top k
  ```

  (Beyond the lecture: fraction of variance explained is `evals / evals.sum()`.)

### 6. Trace, determinant, definiteness

| Quantity | Meaning | Code | Link to eigenvalues |
|---|---|---|---|
| Trace | sum of diagonal | `np.trace(A)` | sum of λ |
| Determinant | volume scaling | `np.linalg.det(A)` | product of λ |

- Check with the example above: trace = 2 = 1.5 + 0.5, and det = 0.75 = 1.5 × 0.5.
- det = 0 means **singular**, so there is no inverse.
- **Positive definite** means all λ > 0. **Positive semi-definite** means all λ ≥ 0. Covariance matrices are PSD.

### 7. Inverse and linear systems

- **Idea:** the inverse is the "undo button".
- **Math:** `A⁻¹A = I`, `(AB)⁻¹ = B⁻¹A⁻¹` (the order flips, like the transpose).
- **Code:**

  ```python
  np.linalg.inv(A)
  x = np.linalg.inv(A) @ y        # solves A x = y
  ```

  In real work `np.linalg.solve(A, y)` is preferred. That is beyond the lecture, which only warns that inversion is numerically unstable.
- **Easy inverses:** orthogonal gives `A.T`, diagonal gives `np.diag(1 / np.diag(D))`.
- Square matrices only, and det ≠ 0.

### 8. SVD

- **Idea:** *any* matrix is rotate, then stretch, then rotate.
- **Math:** `A = U Σ Vᵀ`. U and V are orthogonal, Σ is diagonal with non-negative singular values in descending order.
- **Code traps:**

  ```python
  u, s, vt = np.linalg.svd(A)   # s is a 1D VECTOR; vt is ALREADY transposed
  A_rec = u @ np.diag(s) @ vt   # square case
  ```

  For a non-square A, Σ must be padded with zeros to shape `(m, n)`.
- **Uses:**
  - Inverse: `V Σ⁻¹ Uᵀ`, i.e. `vt.T @ np.diag(1/s) @ u.T`.
  - Pseudo-inverse: `np.linalg.pinv(A)`. For non-square A, `x = pinv(A) @ y` minimises `‖Ax − y‖₂`.
  - Matrix power for symmetric A: `u @ np.diag(s**p) @ vt`.
- **Lecture error:** cell 129 says eigenvectors of `AᵀA` give U. It's the other way round: **`AAᵀ` gives U, `AᵀA` gives V**. The lecture's own code in cell 130 does it correctly.

### 9. Rank and condition number

| Quantity | Meaning | Code |
|---|---|---|
| Rank | number of non-zero singular values | `np.linalg.matrix_rank(A)` |
| Condition number | σ_max / σ_min | `np.linalg.cond(A)` |

- A big condition number means ill-conditioned, so inversion is unreliable.
- Rank less than n means singular.

### 10. Whitening (only if it appears)

```python
Xc = X - X.mean(axis=0)
u, s, vt = np.linalg.svd(np.cov(Xc, rowvar=False))
W  = vt.T @ np.diag(1 / np.sqrt(s)) @ u.T
Xw = Xc @ W            # now zero mean, identity covariance
```

Check: `np.allclose(np.cov(Xw, rowvar=False), np.eye(d))`. The lecture stresses that the inverse square root is *not* an elementwise operation on Σ.

---

## Part 3: Tonight's schedule

| Time | What |
|---|---|
| 60 min | Part 1 table: retype every function from memory, then compare |
| 45 min | Part 2: run cells 42, 58, 122–123, 160 of the Lecture 2 notebook and predict the output before each run |
| 15 min | Part 0 checklist, and fix the import in Lab 0 for habit |
| Then stop | Sleep beats another hour of reading |

**In the lab itself:**

- Run the setup cell first.
- Read each assert before you write code.
- Run the visible test after every task.
- Restart & Run All before you submit.
