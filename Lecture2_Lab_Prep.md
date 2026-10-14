# Lecture 2 Lab Prep: Computational Linear Algebra

Source: `idss_lecture_02_complinearalgebra_v20242025a.ipynb` (read in full).
The slide PDF could not be rendered on this machine (no poppler), but it is the same lecture.
Every code snippet below was run and checked with assertions before going in this file.

---

## 0. How to use this (60–90 min)

1. Read **Section 1** (what's marked examinable) and **Section 2** (how questions get asked), 10 min.
2. Go through **Section 3** top to bottom. For each task: cover the answer, write it in a scratch notebook, then compare. 50 min.
3. Do the **Section 5** rapid-fire Q&A out loud. 10 min.
4. Skim **Section 6** (traps) just before you sleep and again just before the lab.

If you are short on time, the priority order is: **3.1 → 3.2 → 3.4 → 3.5 → 3.7 → 3.8 → 3.9**.

---

## 1. What the lecture itself marks as examinable

The notebook wraps these formulas in an `examinable` environment (the ♠ marker). Know these cold:

| Formula | Plain English |
|---|---|
| x_n = A x_(n−1) / ‖A x_(n−1)‖∞ | power iteration: multiply, then rescale |
| Tr(A) = Σ aᵢᵢ = Σ λᵢ | trace = sum of diagonal = sum of eigenvalues |
| det(A) = Π λᵢ | determinant = product of eigenvalues |
| A = U Σ Vᵀ | SVD: rotate, scale, rotate |
| A⁻¹ = V Σ⁻¹ Uᵀ | inverse from the SVD |
| Xʷ = (X − μ) Σ^(−1/2) | whitening |

The summary slide's learning outcomes are your topic list:
graphs as matrices, flows, eigen, power iteration, trace/det, PSD, SVD, linear systems, inverse, pseudo-inverse, whitening, low-rank approximation.

---

## 2. How lab questions will probably look

Same template as Lab 0 (inferred from Lab 0; I have not seen Lab 1):

- A markdown cell with the task and the **variable or function name** you must create.
- A `# YOUR CODE HERE` cell. You replace the code *and* the `raise NotImplementedError()`.
- A visible test cell (`tick.marks`, `check_hash`, shape asserts) then hidden tests with different inputs.

Question shapes to expect:

| Shape | Example | What to do |
|---|---|---|
| Compute a value | "Compute the in-degree of every vertex of `adj` into `in_deg`" | one NumPy line, check the shape |
| Write a function | "Write `is_psd(A)` returning True or False" | general code, no hard-coded numbers |
| Fill a gap in a pipeline | "centre, take covariance, find top eigenvector, project" | chain the lecture's steps in order |
| Short string answer | `answer = "singular"` | like Lab 0 Task 8, see Section 5 |
| Interpret | "which vertex is a sink?" | read row sums |

Rules that still apply: no `import` in answer cells, 1D vectors are `(n,)`, use `@` not `*`, Restart & Run All before submitting.

---

## 3. Task bank (concept, code, trap)

Use `np` as already imported. All of these were checked.

### 3.1 Graphs as matrices

**ELI5:** a map of one-way roads written as a grid. A 1 in row *i*, column *j* means "road from *i* to *j*".

```python
out_deg = A.sum(axis=1)          # edges LEAVING each vertex: row sums
in_deg  = A.sum(axis=0)          # edges ENTERING each vertex: column sums
undirected = np.allclose(A, A.T) # symmetric means undirected
A_sym = A + A.T                  # force undirected (every arrow becomes two-way)
loops = np.count_nonzero(np.diag(A))   # non-zero diagonal means self-transitions
```

Checked on the lecture graph: out-degrees `[3,2,2,3,0,0,0,1]`, in-degrees `[1,2,1,3,1,1,1,1]`.

**Trap:** lecture cell 14 labels `axis=0` as "along rows". Ignore the comment. `axis=0` sums *down* each column, so it gives in-flow.

**Sources, sinks, conserving** (weighted graph):

- row sum > 1: **source** (makes stuff)
- row sum < 1: **sink** (consumes stuff)
- row sum = 1: **conserves** (only re-routes)

```python
A_cons = A / A.sum(axis=1, keepdims=True)   # every row now sums to 1
```

**Trap:** a row of all zeros (like E, F, G above) gives 0/0 = `nan`. The lecture says "so long as each vertex has some flow out".

### 3.2 Flow and matrix powers

**ELI5:** "where are the packages tomorrow?" is one multiplication. "In a week?" is the same machine run seven times.

```python
x1 = x0 @ A                              # row vector on the LEFT: tomorrow
x7 = x0 @ np.linalg.matrix_power(A, 7)   # a week from now
```

Hand-rolled version from the lecture (know it, it may be asked):

```python
def powm(A, n):
    B = np.eye(A.shape[0])     # start with identity
    for _ in range(n):
        B = A @ B
    return B
```

**Trap:** `A**7` is *elementwise*. Matrix power needs `np.linalg.matrix_power`. Only square matrices have powers.

**Check:** if A is conserving and x0 sums to 1, then x0 @ Aᵗ still sums to 1.

**Steady state:** apply a conserving matrix many times and the distribution stops changing. That fixed vector is an eigenvector (eigenvalue 1). "All roads lead to Rome."

### 3.3 Eigenvectors and eigenvalues

**ELI5:** most arrows get turned *and* stretched by a matrix. The special arrows that only get stretched, not turned, are eigenvectors. The stretch factor is the eigenvalue.

**Math:** A v = λ v, for square A only.

```python
evals, evecs = np.linalg.eig(A)       # eigenvectors are the COLUMNS: evecs[:, i]
i   = np.argmax(np.abs(evals))        # biggest by absolute value
lam = evals[i]
v   = evecs[:, i]
np.allclose(A @ v, lam * v)           # the defining check -> True
```

Sort largest first (lecture pattern, cells 52/58):

```python
order = np.argsort(-np.abs(evals))
evals = evals[order]
evecs = evecs[:, order]
```

- Unit-length columns come back (`np.linalg.norm(evecs, axis=0)` is all 1s).
- Symmetric matrix → use `np.linalg.eigh` (more stable, orthogonal, but **ascending** order).
- Eigenvalues are unique, eigenvectors are not (sign flips are fine).

**Hand example (lecture cell 58):** A = [[1, 0.5], [0.5, 1]] has λ = 1.5 with v ∝ [1, 1] and λ = 0.5 with v ∝ [1, −1].

**Power iteration** (examinable):

```python
def power_iterate(A, x, n=500):
    for _ in range(n):
        x = A @ x
        x = x / np.linalg.norm(x, np.inf)
    return x / np.linalg.norm(x, 2)

lam = ((A @ x) / x)[0]          # eigenvalue: elementwise ratio, same in every entry
```

Why normalise: without it x explodes or collapses to 0. Result may differ from `eig` by a **sign**.

### 3.4 Eigendecomposition and low-rank approximation

**ELI5:** a matrix is a recipe of "directions" and "how much to stretch each". Keep the big stretches, drop the tiny ones, and you keep most of the matrix.

```python
evals, Q = np.linalg.eigh(S)                 # S symmetric
S_back = Q @ np.diag(evals) @ Q.T            # Σ = Q Λ Q^T  -> equals S

# keep only the k biggest |eigenvalues|
o = np.argsort(-np.abs(evals))[:k]
S_k = Q[:, o] @ np.diag(evals[o]) @ Q[:, o].T      # rank k

# lecture's version: zero the small ones
approx_evals = np.where(np.abs(evals) < 0.2, 0, evals)
```

- **Eigenspectrum** = |λ| sorted largest first.
- One big eigenvalue and many small: easy to approximate (one direction does most of the work).
- All similar: hard to approximate; the transform stretches every direction about equally.
- A zero eigenvalue means a dimension is squashed flat, so the matrix is singular.

### 3.5 Mean, covariance, PCA

**ELI5:** a cloud of dots. PCA finds the direction in which the cloud is longest.

```python
mu = X.mean(axis=0)                      # mean vector, shape (d,)
C  = np.cov(X, rowvar=False)             # d x d, divides by N-1
evals, evecs = np.linalg.eigh(C)         # ASCENDING
order = np.argsort(-evals)               # largest first
W  = evecs[:, order[:k]]                 # d x k : top-k principal components
Z  = (X - mu) @ W                        # N x k : projected data
explained = evals[order] / evals.sum()   # fraction of variance per component
```

Facts (checked numerically):

- Variance of the data along PC1 = the largest eigenvalue.
- `evals.sum()` = `np.trace(C)` = sum of the per-column variances.
- Principal component *length* is √λ (the lecture's ellipse picture).
- Principal components = eigenvectors of the covariance matrix.

**Traps:**

1. `np.cov(X)` treats rows as variables. You need `rowvar=False` (or `X.T`).
2. `eigh` is ascending, so the **last** column is the biggest.
3. The lecture projects *uncentred* data (`x @ evecs[:, 0]`, cell 78). The principled version subtracts the mean first. In the lab, follow the task text.
4. `x @ evecs[:, 0]` gives shape `(N,)`. `x @ evecs[:, :1]` gives `(N, 1)`. Match the assert.

### 3.6 Trace, determinant, definiteness

| Quantity | Meaning | Code | Link to eigenvalues |
|---|---|---|---|
| Trace | sum of the diagonal | `np.trace(A)` | **sum** of λ |
| Determinant | how much volume is scaled | `np.linalg.det(A)` | **product** of λ |

Example: A = [[1, 0.5], [0.5, 1]]: trace 2 = 1.5 + 0.5, det 0.75 = 1.5 × 0.5.

- det = 0 means at least one λ = 0, so **singular**, no inverse, information lost.

| Eigenvalues | Name |
|---|---|
| all > 0 | positive definite |
| all ≥ 0 | positive semi-definite (PSD) |
| all < 0 | negative definite |
| all ≤ 0 | negative semi-definite |

```python
def is_psd(A, tol=1e-10):
    return bool(np.all(np.linalg.eigvalsh(A) >= -tol))   # for symmetric A
```

- Covariance matrices are always symmetric and PSD.
- Why `tol`: floating point can give −1e-17 for a true zero.
- Positive definite: xᵀAx > 0 for all non-zero x (A never rotates x by more than 90°).
- Quick check: [[1,2],[2,1]] has eigenvalues 3 and −1, so it is *not* PSD.

### 3.7 Inverse and linear systems

**ELI5:** the inverse is the "undo" button.

- A⁻¹(Ax) = x, A⁻¹A = I, (A⁻¹)⁻¹ = A.
- **(AB)⁻¹ = B⁻¹A⁻¹** (order flips, same as the transpose rule).

```python
Ainv = np.linalg.inv(A)
x = np.linalg.inv(A) @ y          # solves A x = y   (lecture method)
x = np.linalg.solve(A, y)         # same answer, preferred in real work
np.allclose(A @ x, y)             # verify
```

Easy inverses:

| Matrix | Inverse | Cost |
|---|---|---|
| orthogonal (rotation) | `A.T` | O(1) |
| diagonal | `np.diag(1 / np.diag(D))` | O(n) |
| general n×n | `np.linalg.inv` | O(n³) |

**Traps:**

- Inverse needs a **square** matrix with det ≠ 0. Singular input raises `LinAlgError`.
- The inverse of a *sparse* matrix is generally dense.
- Inverting directly is numerically fragile. Iterative / optimisation methods only solve for one specific y, so they are more stable.
- An orthogonal matrix has det ±1 and eigenvalues of magnitude 1.
- A linear system reads as "each output is a weighted sum of the inputs": row *i* of A holds the weights for y_i.

### 3.8 SVD

**ELI5:** *any* matrix, even a non-square one, is "rotate, stretch, rotate".

**Math:** A = U Σ Vᵀ, A is m×n, U is m×m orthogonal, Σ is m×n diagonal (singular values, ≥ 0, descending), V is n×n orthogonal.

```python
u, s, vt = np.linalg.svd(A)
# s is a 1D VECTOR of singular values, not a matrix
# vt is ALREADY transposed (it is V^T)

# square case
A_back = u @ np.diag(s) @ vt

# non-square: build Sigma with the right shape
Sig = np.zeros(A.shape)
k = min(A.shape)
Sig[:k, :k] = np.diag(s)
A_back = u @ Sig @ vt
```

Shapes for a 2×3 matrix: `u` is (2,2), `s` is (2,), `vt` is (3,3).

Uses (all checked):

```python
A_inv  = vt.T @ np.diag(1 / s) @ u.T                 # inverse, square only (examinable)
A_pinv = np.linalg.pinv(A)                           # pseudo-inverse, any shape
x      = np.linalg.pinv(A) @ y                       # least-squares solution of A x = y
A_p    = u @ np.diag(s ** p) @ vt                    # A^p, for symmetric PSD A
```

- Pseudo-inverse: if an exact solution exists you get it. If not, you get the x minimising ‖Ax − y‖₂ (the closest in L2).
- `A**0.5` is **not** the matrix square root. Take the square root of the singular values, not of the entries.
- Relation to eigen: eigenvectors of **AAᵀ** give U, eigenvectors of **AᵀA** give V, and σᵢ = √|λᵢ|.
- For symmetric PSD A, singular values = eigenvalues.

**Caveats:**

- Lecture cell 129 has U and V swapped. The code in cell 130 is right.
- `u @ diag(s**p) @ vt` is only right for **symmetric PSD** matrices. For an indefinite symmetric matrix like [[1,2],[2,1]], p = 2 gives the wrong sign. (Beyond the lecture, I checked it numerically.) For an inverse, p = −1 is fine.

### 3.9 Rank and condition number

| Quantity | Meaning | Code |
|---|---|---|
| Rank | number of non-zero singular values; dimensions kept | `np.linalg.matrix_rank(A)` |
| Condition number | σ_max / σ_min; how fragile inversion is | `np.linalg.cond(A)` |

```python
s = np.linalg.svd(A, compute_uv=False)
rank = int(np.sum(s > 1e-10))
cond = s.max() / s.min()
```

- Full rank means invertible (det ≠ 0). Rank < n means singular / deficient rank.
- Small cond means well-conditioned. Large cond means ill-conditioned, nearly singular, inversion gives garbage.
- Identity has cond = 1.
- Lecture cell 160: [[1,2,3],[2,4,6],[0,−1,−5]] has rank 2 (row 2 is 2 × row 1) and one singular value ≈ 0.
- A 4×4 matrix of rank 2 maps R⁴ into a 2D plane.

### 3.10 Whitening

**ELI5:** squash a stretched, tilted cloud into a round ball centred on zero.

```python
Xc = X - X.mean(axis=0)                      # 1. centre
u, s, vt = np.linalg.svd(np.cov(Xc, rowvar=False))
W  = vt.T @ np.diag(1 / np.sqrt(s)) @ u.T    # 2. inverse square root of Σ
Xw = Xc @ W                                  # 3. apply
```

Check (verified): `np.allclose(Xw.mean(axis=0), 0)` and `np.allclose(np.cov(Xw, rowvar=False), np.eye(d))`.

**Trap:** Σ^(−1/2) is **not** `1/np.sqrt(Sigma)` elementwise. The `s` here comes from the SVD of the *covariance matrix*, not of X.

---

## 4. Mini exercises in lab style

Try these without looking at Section 3. Answers are below.

1. `adj` is a (6,6) adjacency matrix. Create `top_hub`: the **index** of the vertex with the largest out-degree.
2. Write `steady(A, n=1000)` that starts from a uniform row vector and returns the distribution after `n` steps of `x @ A`.
3. Given symmetric `S`, create `lam_max`: the eigenvalue with the largest absolute value.
4. Given data `X` (N×d), create `Z`: X projected onto its **top 2** principal components, shape (N, 2).
5. Write `is_invertible(A)` returning True/False using the **rank**.
6. Given `A` (3×2) and `y` (3,), create `x_hat` minimising ‖Ax − y‖₂.
7. Given square `A`, create `cn`: its condition number using only `np.linalg.svd`.
8. Write `inv_sqrt(S)` for symmetric PSD `S` using the SVD.

<details><summary>Answers</summary>

```python
top_hub = np.argmax(adj.sum(axis=1))

def steady(A, n=1000):
    x = np.ones(A.shape[0]) / A.shape[0]
    for _ in range(n):
        x = x @ A
    return x

ev = np.linalg.eigvalsh(S)
lam_max = ev[np.argmax(np.abs(ev))]

evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))
W = evecs[:, np.argsort(-evals)[:2]]
Z = (X - X.mean(axis=0)) @ W            # (N, 2)

def is_invertible(A):
    return A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]

x_hat = np.linalg.pinv(A) @ y

s = np.linalg.svd(A, compute_uv=False)
cn = s.max() / s.min()

def inv_sqrt(S):
    u, s, vt = np.linalg.svd(S)
    return u @ np.diag(1 / np.sqrt(s)) @ vt
```

</details>

---

## 5. Rapid-fire concept Q&A (for string / multiple-choice answers)

| Question | Answer |
|---|---|
| Which matrices have eigenvalues? | square only |
| Which matrices have an SVD? | **any** matrix |
| What does an eigenvector do under A? | only scaled, not rotated |
| det(A) = 0 means? | singular, not invertible, a dimension collapses |
| Sum of eigenvalues? | trace |
| Product of eigenvalues? | determinant |
| Covariance matrix is? | square, symmetric, positive semi-definite |
| Principal components are? | eigenvectors of the covariance matrix |
| Which principal component matters most? | the one with the largest eigenvalue |
| Inverse of an orthogonal matrix? | its transpose |
| Inverse of a diagonal matrix? | reciprocal of each diagonal entry |
| (AB)⁻¹ = ? | B⁻¹A⁻¹ |
| Why not invert to solve Ax = y? | numerically unstable; iterative methods are better |
| What does the pseudo-inverse give you? | the least-squares (L2-closest) solution |
| Rank of a matrix? | number of non-zero singular values |
| Large condition number means? | ill-conditioned, nearly singular |
| Whitening achieves? | zero mean, identity covariance |
| Why is `eigh` preferred for symmetric matrices? | more stable, orthogonal eigenvectors |
| Row sums > 1 / < 1 / = 1 in a flow graph? | source / sink / conserving |
| Power iteration finds? | the leading (largest |λ|) eigenvector |
| Matrix power and what shape must A be? | repeated application, square |
| Time to invert a general n×n matrix? | O(n³) |
| SVD as geometry? | rotate, scale, rotate |

---

## 6. Traps that cost marks

1. `**` on a matrix is elementwise. Use `np.linalg.matrix_power`, `@`, or `powm`.
2. `eig` returns eigenvectors as **columns**: `evecs[:, i]`, never `evecs[i]`.
3. `eigh` returns **ascending** order, `svd` returns **descending**. Re-sort `eig` yourself.
4. `np.linalg.svd` returns `vt`, already transposed, and `s` as a 1D vector.
5. For non-square SVD, Σ must be padded to shape (m, n).
6. `np.cov` needs `rowvar=False` for data with variables in columns.
7. Eigenvectors and power iteration results can differ by a **sign**. Compare `np.abs(...)` or check `A @ v ≈ λ v`.
8. `axis=0` collapses rows (one value per column). `axis=1` gives one value per row.
9. Floats: use `np.allclose`, never `==`. Add a tolerance to rank/PSD tests.
10. Row-vector flow is `x @ A`. Column-vector flow is `A @ x`. The lecture uses row vectors for graphs.
11. Whitening needs the matrix inverse square root, not elementwise `sqrt`.
12. Check the shape asserts: `(n,)` versus `(n,1)` versus `(1,n)`.

---

## 7. Night-before checklist

- [ ] Section 3 tasks 3.1 to 3.5 retyped from memory
- [ ] The six examinable formulas in Section 1 written out without looking
- [ ] Section 4 exercises done
- [ ] Section 6 read once
- [ ] Lab 0 Task 1 cell: remove `import numpy as np`
- [ ] Sleep
