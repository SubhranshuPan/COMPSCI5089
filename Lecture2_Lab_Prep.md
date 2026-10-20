# Lecture 2 Lab Prep: Computational Linear Algebra

Source: `idss_lecture_02_complinearalgebra_v20242025a.ipynb` (read in full).
The slide PDF could not be rendered on this machine (no poppler), but it is the same lecture.
Every code snippet below was run and checked before going in this file.

---

## 0. How to use this (60–90 min)

The cards in Section 3 now follow one pattern, so you always know where to look:

> **Story** (a picture to hold in your head) → **One line of maths** → **The code, word by word** (what every word does and *why it is there*) → **Recall hook** (one sentence to remember the code) → **How they might ask it**

Order:

1. Read **Section 2A** once (the NumPy words you keep seeing). 10 min.
2. Go through **Section 3**. For each card: read the story, read the word-by-word table, then cover the code and retype it in a scratch notebook. 50 min.
3. Do the **Section 4** exercises. Then **Section 5** out loud.
4. Skim **Section 6** (traps) tonight and again just before the lab.

If you are short on time, the priority order is: **3.1 → 3.2 → 3.4 → 3.5 → 3.7 → 3.8 → 3.9**.

The one trick for remembering code: **don't memorise lines, memorise the story, and the line follows from it.**
(For example: "undo the recipe in reverse order" *is* `vt.T @ np.diag(1/s) @ u.T`.)

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

## 2A. The words you keep seeing (read this once and the cards get easy)

### The big picture in four lines

- A **vector** is a list of numbers, like an arrow or a shopping list.
- A **matrix** is a **machine** that takes an arrow in and gives a new arrow out. `A @ x` means "put `x` through machine `A`".
- A **square** matrix is a machine whose input and output have the same size. Only square machines can be repeated (powers), have eigenvectors, or be undone (inverse).
- **Shape is everything.** Most lost marks are a shape that is off by one axis.

### Shapes: what the lab calls a "vector"

| You write | Shape | Plain English |
|---|---|---|
| `np.array([1, 2, 3])` | `(3,)` | a flat list. **This is the lab's "vector".** |
| `x.reshape(-1, 1)` | `(3, 1)` | a column (3 rows, 1 column) |
| `x.reshape(1, -1)` | `(1, 3)` | a row (1 row, 3 columns) |
| `A` with 4 rows, 3 columns | `(4, 3)` | rows first, then columns |

### The `axis` rule: the axis is the direction that gets squashed flat

Take `A = [[0,1,1],[0,0,1],[1,0,0]]`.

| Code | What happens | Result | Shape |
|---|---|---|---|
| `A.sum(axis=0)` | squash **downwards**, one total per **column** | `[1, 1, 2]` | `(3,)` |
| `A.sum(axis=1)` | squash **sideways**, one total per **row** | `[2, 1, 1]` | `(3,)` |

### Word list

| Word | What it does | Why you see it |
|---|---|---|
| `np.array([...])` | builds an array from a Python list | creating test data |
| `A.T` | transpose: flip across the diagonal, rows become columns | `Qᵀ`, `Vᵀ`, `Aᵀ` |
| `A.shape` | `(rows, columns)` | sizing things, building zero sheets |
| `A @ B` | **real** matrix product | all the maths |
| `A * B` | multiplies entry by entry, **not** the matrix product | almost always a bug |
| `A ** 2` | squares each entry, **not** `A @ A` | almost always a bug |
| `np.eye(n)` | the "do-nothing machine": 1s on the diagonal, 0 elsewhere | starting point for repeated products |
| `np.diag(v)` | given a **flat list**, builds a diagonal **matrix** | `Λ`, `Σ` |
| `np.diag(M)` | given a **matrix**, pulls out its diagonal as a flat list | self-loops, reciprocals |
| `.sum(axis=…)` | adds entries along an axis | degrees, row totals |
| `keepdims=True` | keeps the squashed axis as size 1, so shape `(n,1)` instead of `(n,)` | lets you divide a matrix **row by row** |
| `np.abs`, `np.sqrt` | applied to every entry separately | sorting by size, `1/√σ` |
| `np.argmax(v)` | the **position** of the biggest entry | "which eigenvalue is largest?" |
| `np.argsort(v)` | the positions that would sort `v` from small to big | re-ordering eigenvalues |
| `np.argsort(-v)` | same, but the minus flips it, so **biggest first** | "largest eigenvalue first" |
| `evecs[:, i]` | `:` means all rows, `i` picks one column, so the i-th eigenvector | eigenvectors live in columns |
| `evecs[:, order]` | re-orders the columns to match `order` | keeping vectors paired with their values |
| `evecs[:, :k]` | the first `k` columns | top-k components |
| `x[::-1]` | reversed copy | flip ascending to descending |
| `np.allclose(a, b)` | "equal up to rounding" | `0.1 + 0.2 == 0.3` is `False`, but `allclose` is `True` |
| `np.where(cond, a, b)` | per entry: if `cond` is true take `a`, else `b` | zeroing small eigenvalues |
| `np.zeros(shape)` | a blank sheet of zeros | padding Σ |
| `np.count_nonzero(v)` | how many entries are not zero | counting self-loops |
| `np.linalg.xxx` | the linear-algebra toolbox | `eig`, `eigh`, `svd`, `inv`, `solve`, `pinv`, `det`, `norm`, `matrix_power`, `matrix_rank`, `cond` |

### Three things worth seeing once

**1. `argsort(-abs(...))` on an example.** `evals = [0.5, -3, 1.5]`.
`np.abs` gives `[0.5, 3, 1.5]`, the minus gives `[-0.5, -3, -1.5]`, and `argsort` of that is `[1, 2, 0]`.
Read it as: "put entry 1 first (it is the biggest), then entry 2, then entry 0."
Then `evals[order]` and `evecs[:, order]` re-order the values and the vectors in the **same** way.

**2. `x @ P` versus `P @ x`.** With `x = [1, 0, 0]` and `P = [[.5,.5,0],[0,.5,.5],[.5,0,.5]]`:

- `x @ P` gives `[0.5, 0.5, 0]`, which is **row 0** of `P`.
- `P @ x` gives `[0.5, 0, 0.5]`, which is **column 0** of `P`.

So with the vector on the left, a depot's row is "where it sends its stock". This is why the graph cards use `x @ A`.

**3. `keepdims`, and the silent bug without it.** Row sums of `W = [[1,3],[1,1]]` are `[4, 2]`.

- `W / W.sum(axis=1, keepdims=True)` gives `[[.25,.75],[.5,.5]]`. Every row sums to 1. Correct.
- `W / W.sum(axis=1)` gives `[[.25,1.5],[.25,.5]]`. Rows now sum to 1.75 and 0.75. **No error, wrong answer**, because the shape `(2,)` divides across the columns instead of down the rows.

---

## 3. Task bank (story, code word by word, how it's asked)

Use `np` as already imported. All code here was checked.

---

### 3.1 Graphs as matrices

**Story.** A town's one-way road map, written as a grid.
**Row = where you start. Column = where you can go.** A `1` means "there is a road".

- Count the 1s **along a row**: roads **leaving** that place (out-degree).
- Count the 1s **down a column**: roads **arriving** at that place (in-degree).

Memory hook: **Rows go out, Columns Come in.**

**The code, word by word**

```python
out_deg = A.sum(axis=1)
in_deg  = A.sum(axis=0)
undirected = np.allclose(A, A.T)
A_sym = A + A.T
loops = np.count_nonzero(np.diag(A))
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `A.sum(axis=1)` | add up each **row** | one total per row = roads leaving |
| `A.sum(axis=0)` | add up each **column** | one total per column = roads arriving |
| `A.T` | the map with every road's direction reversed | to test "is every road two-way?" |
| `np.allclose(A, A.T)` | "does the reversed map equal the original?" | yes means every road has a return road, i.e. **symmetric = undirected**. `allclose`, not `==`, because of decimals. |
| `A + A.T` | original plus reversed | gives every one-way road a twin, so the graph becomes undirected |
| `np.diag(A)` | entries `A[i,i]` | a road from place *i* to itself, i.e. a self-loop |
| `np.count_nonzero(...)` | how many of those are not 0 | counts the self-loops |

Checked on the lecture graph: out-degrees `[3,2,2,3,0,0,0,1]`, in-degrees `[1,2,1,3,1,1,1,1]`.

**Weighted graphs: source, sink, conserving.** A row says how much each place ships onward.

- row sums to **more than 1**: it makes extra stuff, a **source**
- row sums to **less than 1**: it keeps some, a **sink**
- row sums to **exactly 1**: it only passes things on, **conserving**

```python
A_cons = A / A.sum(axis=1, keepdims=True)
```

Read it as: "divide each row by **its own** total, so every row now sums to 1".
`keepdims=True` is what makes this work row by row (see the silent bug in Section 2A).

**Recall hook:** *Rows out, columns in; normalise a row by its own total, `keepdims=True`.*

**How they might ask it:** "Compute `in_deg` of `adj`." / "Write `is_undirected(A)`." / "Which vertex is a sink?" (find the row sum below 1).

**Watch out:** lecture cell 14's comments swap rows and columns. Ignore the comments. An all-zero row (a dead end) gives `0/0 = nan` when normalised.

---

### 3.2 Flow and matrix powers

**Story.** `x` is a list of how much stock is at each depot. `A` is the delivery rule: **row *i* is depot *i*'s delivery list** (what share goes to each other depot).
One multiplication `x @ A` is "one night of deliveries". Seven nights is the same button pressed seven times.

Why `x` goes on the **left**: each depot splits its stock along its own row. That is `x @ A` (see the `x @ P` example in Section 2A).

**The code, word by word**

```python
x1 = x0 @ A
x7 = x0 @ np.linalg.matrix_power(A, 7)

def powm(A, n):
    B = np.eye(A.shape[0])
    for _ in range(n):
        B = A @ B
    return B
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `x0 @ A` | stock tomorrow | row vector on the left, as above |
| `np.linalg.matrix_power(A, 7)` | `A @ A @ … @ A`, seven times | "press the button 7 times" as one matrix |
| `np.eye(A.shape[0])` | the do-nothing machine, same size as `A` | like starting a multiplication at `1`: nothing has happened yet |
| `A.shape[0]` | number of rows, which equals `n` for a square matrix | to build an identity of the right size |
| `for _ in range(n)` | repeat `n` times. `_` means "I don't need the counter" | press the button `n` times |
| `B = A @ B` | stack one more `A` on top | `B` is the running product |

**Steady state ("all roads lead to Rome").** Keep pressing the button on a conserving matrix and the stock settles into a fixed pattern. That pattern is an eigenvector with eigenvalue 1. The total stays 1 on the way (mass is conserved).

**Recall hook:** *Row vector on the left; press the button `n` times; `**` would square each entry instead.*

**How they might ask it:** "Write `powm(A, n)`." / "After 3 days, what share is at each depot?" / "Check mass is conserved" (the sum is still 1).

**Watch out:** `A**7` is not a matrix power, it raises each entry to 7. Only square matrices have powers.

---

### 3.3 Eigenvectors and eigenvalues

**Story.** Press a machine onto a rubber sheet. Most arrows get swung around *and* stretched.
A few special arrows **stay on their own line**: they only get longer or shorter. Those are the **eigenvectors** ("eigen" = "own"). How much longer or shorter is the **eigenvalue**.

- `λ = 1.5` means "gets 1.5 times longer"
- `λ = 0.5` means "gets half as long"
- negative `λ` means "flips to point the other way"

**One line of maths:** `A v = λ v`. Machine applied to `v` gives the same `v`, just scaled.

**Check it by hand.** `A = [[1, 0.5],[0.5, 1]]`:

- `v = [1, 1]`: `A v = [1·1+0.5·1, 0.5·1+1·1] = [1.5, 1.5] = 1.5 · v`, so `λ = 1.5`
- `v = [1, −1]`: `A v = [1−0.5, 0.5−1] = [0.5, −0.5] = 0.5 · v`, so `λ = 0.5`

**The code, word by word**

```python
evals, evecs = np.linalg.eig(A)
i   = np.argmax(np.abs(evals))
lam = evals[i]
v   = evecs[:, i]
np.allclose(A @ v, lam * v)
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `evals, evecs = ...` | the function returns **two** things, unpacked into two names | one list of stretches, one matrix of arrows |
| `evals` | flat list of the `λ`s, shape `(n,)` | the stretch amounts |
| `evecs` | an `n×n` matrix whose **columns** are the unit eigenvectors | so the i-th eigenvector is `evecs[:, i]`, never `evecs[i]` |
| `np.abs(evals)` | sizes ignoring sign | `−3` stretches more than `1.5`, so "largest" means largest **absolute** value |
| `np.argmax(...)` | the **position** of the largest | we need the position to pick the matching column |
| `evals[i]` and `evecs[:, i]` | the matching pair | value `i` belongs to column `i` |
| `np.allclose(A @ v, lam * v)` | the defining test `A v = λ v` | proves it really is an eigenpair |

**Sorting biggest first (lecture pattern)**

```python
order = np.argsort(-np.abs(evals))
evals = evals[order]
evecs = evecs[:, order]
```

Why three lines: the order of `eig`'s output is arbitrary. `argsort(-abs)` gives positions biggest-first (worked example in Section 2A). You must re-order **both** the values and the columns of the vectors, or the pairs break.

**`eig` or `eigh`?**

- `eigh` is for **symmetric** matrices only (covariance matrices, `A + A.T`). It is faster, more stable, gives orthogonal eigenvectors, and **always returns real numbers**. But it returns **smallest first**.
- `eig` works for any square matrix.
- On this machine (NumPy 2.5.3), `eig` returned complex-typed numbers (`1.5+0.j`) even for a symmetric matrix, while `eigh` gave plain floats. If you ever see `+0.j`, it is harmless, but `eigh` avoids it. I don't know which NumPy the lab uses.

**Power iteration (examinable)**

**Story:** photocopy a photocopy. Feed the result back in again and again, and the strongest direction takes over. Rescale each round so the paper size doesn't blow up.

```python
def power_iterate(A, x, n=500):
    for _ in range(n):
        x = A @ x
        x = x / np.linalg.norm(x, np.inf)
    return x / np.linalg.norm(x, 2)

lam = ((A @ x) / x)[0]
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `x = A @ x` | push the arrow through the machine | the repeated step |
| `np.linalg.norm(x, np.inf)` | the biggest absolute entry (the L∞ norm from Lecture 1) | the size to scale by |
| `x / that` | rescale so the biggest entry is 1 | stops explosion or collapse to 0 |
| `n=500` | enough rounds to settle | convergence |
| last line, `norm(x, 2)` | make it unit length | so it matches `eig`'s unit vectors |
| `((A @ x) / x)[0]` | `A x = λ x`, so dividing entry by entry gives `λ` in every slot; take slot `[0]` | read off the eigenvalue |

The result may differ from `eig` by a **sign**. That is fine.

**Recall hook:** *`evecs[:, i]` is column i; largest means `argmax(abs)`; power iteration = multiply, then divide by the biggest entry, repeat.*

**How they might ask it:** "Return the eigenvalue of largest magnitude of `A`." / "Complete `power_iterate`." / "Verify that `v` is an eigenvector of `A`."

**Watch out:** eigenvalues are unique, eigenvectors are not (sign flips). Compare with `np.abs(...)` or with the `A v = λ v` check, not with `==`.

---

### 3.4 Eigendecomposition and low-rank approximation

**Story.** `Σ = Q Λ Qᵀ` read right to left, for a symmetric matrix:

1. `Qᵀ`: look at the world from the matrix's favourite angle
2. `Λ`: stretch each axis on its own, no mixing
3. `Q`: turn back to normal

Low-rank approximation is like a band: keep the **loudest** instruments (big eigenvalues) and drop the near-silent ones. The tune is almost the same, but you store far less.

**The code, word by word**

```python
evals, Q = np.linalg.eigh(S)
S_back = Q @ np.diag(evals) @ Q.T

o = np.argsort(-np.abs(evals))[:k]
S_k = Q[:, o] @ np.diag(evals[o]) @ Q[:, o].T

approx_evals = np.where(np.abs(evals) < 0.2, 0, evals)
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `Q` | matrix of eigenvectors, one per column | the "angles" |
| `np.diag(evals)` | turns the flat list into a diagonal matrix `Λ` | `@` needs a matrix, not a list |
| `Q @ Λ @ Q.T` | the three steps above | rebuilds `S`; `np.allclose(S_back, S)` should be `True` |
| `np.argsort(-np.abs(evals))[:k]` | positions of the `k` biggest `|λ|` | choosing which instruments to keep |
| `Q[:, o]` | only the chosen columns | keep those directions |
| `evals[o]` | only the chosen eigenvalues | keep their matching stretches |
| `np.where(np.abs(evals) < 0.2, 0, evals)` | "wherever `|λ|` is under 0.2 put 0, otherwise keep `λ`" | the lecture's way of silencing small ones |

The rebuilt `S_k` has **rank k**.

**Reading the eigenspectrum** (the sorted `|λ|`):

- one big value, many small: easy to approximate
- all similar: hard to approximate; the machine stretches every direction about equally
- a zero eigenvalue: a direction is squashed flat, so the matrix is singular

**Recall hook:** *`Q @ np.diag(evals) @ Q.T`: turn, stretch, turn back. Keep the biggest columns to compress.*

**How they might ask it:** "Reconstruct `S` from `evals` and `Q`." / "Write `truncate(S, k)`." / "What is the rank of the approximation?" (`k`).

**Watch out:** the middle must be `np.diag(...)` of a list, not the raw list.

---

### 3.5 Mean, covariance, PCA

**Story.** A cloud of dots shaped like a tilted rugby ball. PCA finds the **long axis** (principal component 1), then the next axis at right angles.
**Projecting** means casting a **shadow** of every dot onto that axis: a 2D cloud becomes a 1D line of shadows that keeps as much spread as possible.

Recipe, in order:

1. **Centre**: slide the cloud so its middle is at the origin.
2. **Covariance**: a table of how the columns lean together. The diagonal is each column's own variance. Off-diagonal says "when one goes up, does the other?".
3. **Eigen of the covariance**: eigenvectors are the axes of the ball, eigenvalues say how much spread is along each.
4. **Pick the top k** axes.
5. **Project** (shadows).

**The code, word by word**

```python
mu = X.mean(axis=0)
C  = np.cov(X, rowvar=False)
evals, evecs = np.linalg.eigh(C)
order = np.argsort(-evals)
W  = evecs[:, order[:k]]
Z  = (X - mu) @ W
explained = evals[order] / evals.sum()
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `X` | shape `(N, d)`: N dots (rows), d features (columns) | the usual data layout |
| `X.mean(axis=0)` | squash down the rows, so one mean **per column**, shape `(d,)` | the mean vector, the centre of the cloud |
| `np.cov(X, rowvar=False)` | covariance table, shape `(d, d)` | `rowvar=False` says "columns are the variables"; the default assumes the opposite |
| `np.linalg.eigh(C)` | eigen of a symmetric matrix | covariance is always symmetric, so `eigh` |
| `np.argsort(-evals)` | positions biggest first | `eigh` returns **smallest first**, so we flip it |
| `order[:k]` | the first `k` of those positions | the top-k |
| `evecs[:, order[:k]]` | the top-k columns, shape `(d, k)` | the axes to project on |
| `X - mu` | subtract the mean from every row | step 1, centring (broadcasting: a `(d,)` list is subtracted from each row) |
| `(X - mu) @ W` | `(N,d) @ (d,k)` gives `(N,k)` | shadows of every dot: `k` numbers per dot |
| `evals[order] / evals.sum()` | each axis's share of the total spread | "fraction of variance explained" (beyond the lecture) |

**Shape check to do in your head:** the inner numbers must match, `(N, d) @ (d, k)`. Output `(N, k)`.

Facts (checked numerically):

- variance of the data along PC1 equals the largest eigenvalue
- `evals.sum()` equals `np.trace(C)` equals the sum of the per-column variances
- a component's length is `√λ` (the ellipse picture in the lecture)
- principal components are the eigenvectors of the covariance matrix

**Recall hook:** *centre, `np.cov(..., rowvar=False)`, `eigh`, flip with `argsort(-evals)`, shadow with `@`.*

**How they might ask it:** "Compute the covariance matrix of `X`." / "Project `X` onto its top 2 components, shape `(N, 2)`." / "What fraction of variance does PC1 explain?"

**Watch out:** the lecture projects *uncentred* data (`x @ evecs[:, 0]`, cell 78). The proper version subtracts the mean first. Follow the task text. `x @ evecs[:, 0]` gives shape `(N,)`, while `x @ evecs[:, :1]` gives `(N, 1)`, so match the assert.

---

### 3.6 Trace, determinant, definiteness

**Story.** Push a unit square through the machine.

- **Trace**: add the diagonal. Also equals the **sum of the stretches** (eigenvalues).
- **Determinant**: by how much the **area** (volume) is scaled. `0.75` means "the shape shrinks to 75%". It also equals the **product of the stretches**.
- **det = 0**: one stretch is 0, so a direction is **squashed flat like a pancake**. You can't un-flatten a pancake, so no inverse.
- **Positive semi-definite (PSD)**: *no negative stretches*. The machine never flips an axis inside out.

| Quantity | Code | Link to eigenvalues |
|---|---|---|
| Trace | `np.trace(A)` | **sum** of `λ` |
| Determinant | `np.linalg.det(A)` | **product** of `λ` |

Example with `A = [[1, 0.5],[0.5, 1]]`: trace `2 = 1.5 + 0.5`, det `0.75 = 1.5 × 0.5`.

| Eigenvalues | Name |
|---|---|
| all > 0 | positive definite |
| all ≥ 0 | positive semi-definite (PSD) |
| all < 0 | negative definite |
| all ≤ 0 | negative semi-definite |

**The code, word by word**

```python
def is_psd(A, tol=1e-10):
    return bool(np.all(np.linalg.eigvalsh(A) >= -tol))
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `np.linalg.eigvalsh(A)` | just the eigenvalues, symmetric matrices only, real numbers | all we need; `h` means symmetric |
| `>= -tol` | "greater or equal to a tiny negative number" | a true 0 can come out as `-1e-17` because of rounding; `tol` forgives that |
| `np.all(...)` | True only if **every** entry passes | PSD needs all of them |
| `bool(...)` | turns NumPy's `True` into a plain Python `True` | a hidden test may check `is True`, which a NumPy boolean can fail |

Covariance matrices are always symmetric and PSD. Quick check: `[[1,2],[2,1]]` has eigenvalues 3 and −1, so it is **not** PSD.
Positive definite also means `xᵀAx > 0` for every non-zero `x`.

**Recall hook:** *trace = sum of eigenvalues, det = product, det 0 = pancake, PSD = no negative stretches.*

**How they might ask it:** "Write `is_psd(A)`." / "Compute trace and det, and say if `A` is invertible." / `answer = "singular"`.

---

### 3.7 Inverse and linear systems

**Story.** The inverse is the **undo button**: `A⁻¹(Ax) = x`.

**Socks and shoes.** `(AB)⁻¹ = B⁻¹A⁻¹`.
`AB` means do `B` first, then `A` (socks on, then shoes). To undo, take the shoes off first, then the socks: `B⁻¹ A⁻¹`. **The order flips**, just like the transpose rule.

**Solving `A x = y`** means: "which input `x` did the machine turn into this `y`?" Just undo the machine: `x = A⁻¹ y`.

**The code, word by word**

```python
x = np.linalg.inv(A) @ y
x = np.linalg.solve(A, y)
np.allclose(A @ x, y)
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `np.linalg.inv(A)` | builds the full undo button | the lecture's method |
| `... @ y` | presses it on `y` | gives `x` |
| `np.linalg.solve(A, y)` | finds `x` directly without building the undo button | same answer, cheaper and more accurate; used in real work |
| `np.allclose(A @ x, y)` | put `x` back through the machine, do we get `y`? | the verification |

Shapes: `y` of shape `(3,)` gives `x` of shape `(3,)`. `y` of shape `(3,1)` gives `(3,1)`.

**Easy inverses (special machines)**

| Machine | Undo | Code | Cost |
|---|---|---|---|
| orthogonal (a pure rotation) | rotate backwards, which is the transpose | `A.T` | O(1) |
| diagonal (pure stretching) | stretch by the reciprocal | `np.diag(1 / np.diag(D))` | O(n) |
| general n×n | no shortcut | `np.linalg.inv` | O(n³) |

Read `np.diag(1 / np.diag(D))` from the inside out: pull the diagonal out as a list, take `1/` of each entry, rebuild a diagonal matrix.

**Recall hook:** *undo = inverse; shoes off before socks; `solve` beats `inv @`; rotation undo = transpose; stretch undo = reciprocal.*

**How they might ask it:** "Solve `A x = y` for `x`." / "Write `inv_diag(D)`." / "Which of these matrices has no inverse?" (`det = 0`).

**Watch out:** the inverse needs a **square** matrix with `det ≠ 0`; otherwise you get `LinAlgError`. The inverse of a sparse matrix is generally dense. Inverting directly is numerically fragile.

---

### 3.8 SVD

**Story.** *Any* machine, even a non-square one, can be explained as three simple moves:

1. `Vᵀ`: **turn your head**
2. `Σ`: **stretch** along the axes (the stretch amounts are the **singular values**, never negative, biggest first)
3. `U`: **turn again**

Why this matters: each move is trivially undone. A rotation's undo is the transpose, and a stretch's undo is `1/s`. So inverses, powers and square roots all become easy.

**The code, word by word**

```python
u, s, vt = np.linalg.svd(A)
A_back = u @ np.diag(s) @ vt

Sig = np.zeros(A.shape)
k = min(A.shape)
Sig[:k, :k] = np.diag(s)
A_back = u @ Sig @ vt
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `u` | the matrix `U` (the last turn) | |
| `s` | the singular values as a **flat list**, not a matrix | so wrap it with `np.diag(s)` before `@` |
| `vt` | `Vᵀ`, **already flipped** (`t` for "transposed") | don't flip it again; to get `V`, write `vt.T` |
| `np.diag(s)` | list to diagonal matrix | `Σ` |
| `np.zeros(A.shape)` | a blank `m×n` sheet | non-square `Σ` must be `m×n` |
| `k = min(A.shape)` | the number of singular values | the size of the little square |
| `Sig[:k, :k] = np.diag(s)` | paste the little diagonal into the top-left corner of the blank sheet | pads with zeros to the right shape |

Shapes for a 2×3 matrix: `u` is `(2,2)`, `s` is `(2,)`, `vt` is `(3,3)`.

**Inverse: undo the recipe in reverse order**

```python
A_inv  = vt.T @ np.diag(1 / s) @ u.T
A_pinv = np.linalg.pinv(A)
x      = np.linalg.pinv(A) @ y
A_p    = u @ np.diag(s ** p) @ vt
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `u.T` | undo the last turn | the recipe was `Vᵀ` then `Σ` then `U`, so undo starts with `U` |
| `np.diag(1 / s)` | undo the stretch | `1/s` entry by entry |
| `vt.T` | undo the first turn | goes last |
| the order, right to left | `u.T`, then `1/s`, then `vt.T` | **reverse the recipe, reverse the order** (matrix products read right to left) |
| `np.linalg.pinv(A)` | **pseudo-inverse**: a best-effort undo that also works for non-square `A` | for when there is no exact undo |
| `pinv(A) @ y` | the `x` that makes `A x` as close as possible to `y` | least squares: like drawing the best line through scattered points |
| `s ** p` | raise each stretch to the power `p` | `p = 0.5` gives a square root, `p = -1` an inverse |

- If an exact solution exists, `pinv` finds it. If not, it finds the `x` minimising `‖Ax − y‖₂`.
- `A**0.5` is **not** the matrix square root. You take the square root of the singular values, not of the entries.
- Relation to eigen: eigenvectors of **AAᵀ** give `U`, eigenvectors of **AᵀA** give `V`, and `σᵢ = √|λᵢ|`.
- For symmetric PSD `A`, the singular values equal the eigenvalues.

**Caveats**

- Lecture cell 129 has `U` and `V` swapped. The code in cell 130 is right.
- `u @ diag(s**p) @ vt` is only right for **symmetric PSD** matrices. For an indefinite symmetric matrix like `[[1,2],[2,1]]`, `p = 2` gives the wrong sign (beyond the lecture, checked numerically). `p = −1` (the inverse) is fine.

**Recall hook:** *`u, s, vt = svd(A)`; `s` is a flat list, `vt` is already flipped; inverse = `vt.T @ diag(1/s) @ u.T`.*

**How they might ask it:** "Reconstruct `A` from its SVD." / "Compute `A⁻¹` using the SVD." / "Solve `A x = y` for a non-square `A`." / "Return the singular values of `A`."

---

### 3.9 Rank and condition number

**Story.**

- **Rank** = "how many directions survive". A machine of rank 2 acting on 3D space squashes everything onto a flat sheet. Rank less than `n` means singular.
- **Condition number** = the **wobbly table**: longest stretch divided by shortest. A huge ratio is a table with one short leg: a tiny nudge on the input gives a huge change on the output, so inverting is unreliable.

| Quantity | Meaning | Code |
|---|---|---|
| Rank | number of non-zero singular values | `np.linalg.matrix_rank(A)` |
| Condition number | σ_max / σ_min | `np.linalg.cond(A)` |

**The code, word by word**

```python
s = np.linalg.svd(A, compute_uv=False)
rank = int(np.sum(s > 1e-10))
cond = s.max() / s.min()
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `compute_uv=False` | "only give me `s`, skip `u` and `vt`" | all we need here, so the result is a flat list |
| `s > 1e-10` | a list of True/False: "is this stretch really non-zero?" | a true 0 may come out as `1e-17`, so we test against a tiny number, not `== 0` |
| `np.sum(...)` | counts the Trues (True counts as 1) | that count is the rank |
| `int(...)` | plain Python integer | tidy output |
| `s.max() / s.min()` | biggest stretch over smallest | the condition number |

- Full rank means invertible (`det ≠ 0`). Identity has `cond = 1`, the best possible.
- Small `cond`: well-conditioned. Large `cond`: ill-conditioned, nearly singular.
- Lecture cell 160: `[[1,2,3],[2,4,6],[0,-1,-5]]` has rank 2 because row 2 is twice row 1. One singular value is about 0.
- A 4×4 matrix of rank 2 maps `R⁴` into a flat 2D plane.

**Recall hook:** *rank = count the non-zero singular values; cond = biggest over smallest.*

**How they might ask it:** "Compute the rank of `A`." / "Is `A` well-conditioned?" / "Write `cond(A)` using `svd`."

---

### 3.10 Whitening

**Story.** An egg-shaped, off-centre cloud. Two steps turn it into a round ball:

1. **Slide** the cloud so its centre sits at the origin.
2. **Squash** the long direction and **stretch** the short one until the egg becomes a ball of radius 1.

The result has **zero mean** and **identity covariance**.

**The code, word by word**

```python
Xc = X - X.mean(axis=0)
u, s, vt = np.linalg.svd(np.cov(Xc, rowvar=False))
W  = vt.T @ np.diag(1 / np.sqrt(s)) @ u.T
Xw = Xc @ W
```

| Piece | Meaning | Why it's there |
|---|---|---|
| `X - X.mean(axis=0)` | subtract each column's mean | step 1: slide to the origin |
| `np.cov(Xc, rowvar=False)` | covariance table of the centred data, `(d,d)` | it describes the egg's shape |
| `np.linalg.svd(...)` | split that table into `u, s, vt` | the SVD makes inverse square roots easy |
| `s` | here these are the **variances** along the axes | `√s` is the standard deviation |
| `1 / np.sqrt(s)` | 1 divided by the standard deviation | "squash by the spread" |
| `vt.T @ np.diag(...) @ u.T` | the inverse-square-root machine `Σ^(−1/2)` | the same "reverse the recipe" pattern as the inverse, with `1/√s` instead of `1/s` |
| `Xc @ W` | push every row through `W` | step 2: the egg becomes a ball |

Verify: `np.allclose(Xw.mean(axis=0), 0)` and `np.allclose(np.cov(Xw, rowvar=False), np.eye(d))`.

**Recall hook:** *centre, SVD of the covariance, `vt.T @ diag(1/sqrt(s)) @ u.T`, multiply.*

**How they might ask it:** "Whiten `X` into `X_white`." / "Check the whitened data has identity covariance."

**Watch out:** `Σ^(−1/2)` is **not** `1/np.sqrt(Sigma)` applied entry by entry. The `s` comes from the SVD of the *covariance matrix*, not from the SVD of `X`.

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
