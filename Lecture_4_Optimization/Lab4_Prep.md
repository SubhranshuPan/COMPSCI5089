# Lab 4 Prep Pack: Optimisation I (assessed lab, Thursday)

How to use this: work top to bottom. Every block follows **5-year-old version → maths → code → one check question**. Answers to the check questions are at the end (Part 9). All code in this file was run and tested before going in.

**What I have and haven't seen**
- I read the whole Lecture 4 notebook plus its helper files (`utils/history.py`, `utils/optimisers.py`, `evenspace.py`). The PDF is an export of that same notebook.
- I have **not** seen the Lab 4 notebook itself. The "likely tasks" list below is a prediction from the lecture and from how Lab 2 was built. It is not a leak. Treat the drills as practice for the *skills*, not the questions.

---

## Part 0: Plan for today (about 5.5 hours of work)

| Time | Block | Goal |
|---|---|---|
| 25 min | 1–2 | Parameters, objective, writing a loss function |
| 20 min | 3 | Constraints and penalty functions |
| 15 min | 4 | Convex / local minima / continuity (theory only) |
| 25 min | 5 | Linear least squares (the one *direct* method) |
| 20 min | 6 | `scipy.optimize.minimize` |
| 25 min | 7 | Grid search and random search |
| 20 min | 8 | Hill climbing |
| 30 min | 9 | Simulated annealing |
| 25 min | 10 | Genetic algorithm |
| 20 min | 11 | Memory/ants, convergence, what goes wrong |
| 60 min | Part 7 | Drills: close this file, retype each answer from memory |
| 20 min | Part 8 | Dry run: restart-and-run-all discipline, checklist |

Take a 10-minute break after every two blocks. Tonight: sleep. Algorithms you can *retype from the skeleton in Part 5* beat algorithms you "sort of remember".

---

## Part 1: What the Lab 2 reference tells you about the format

I read your completed Lab 2. Lab 4 will almost certainly use the same machinery.

| What Lab 2 did | What it means for Lab 4 |
|---|---|
| Task 0: `student_id` (8 chars) and signature, 0 marks | Do it first. Run the test cell. |
| "provided data, do not change" cells | Never edit them. Your code reads variables from them. |
| Answer cell = replace **both** `# YOUR CODE HERE` and `raise NotImplementedError()` | Leave neither behind, or it still raises. |
| Tests assert **shapes** (`proj.shape == (6,)`, `cov.shape == (2, 2)`) | Return exactly the shape asked: scalar vs 1D vs 2D. |
| Tests check a **defining property**, "true for any correct answer" (e.g. `M @ v == λ v`) | For Lab 4, expect property tests: returned loss equals `L(returned_theta)`, loss is lower than the start, result is near the known minimum, trace never goes up. Many different correct implementations pass. |
| `np.allclose(..., atol=1e-6)`, and "equal or equal after flipping sign" | Compare floats with tolerance, never `==`. |
| `bool(pc1_matches_svd) is True` | If asked for a True/False, store a real `bool`. |
| Half the marks were **hidden** (10 visible, 10 hidden) | A passing visible test is not enough. Write general code that works for other inputs. |
| Hints named the classic mistake (column vs row) | Read every hint. It is telling you where marks are lost. |
| A "provided" last line (e.g. `reconstruction_error = ...`) | Keep your variable names exactly as the provided line expects. |
| Rules from Lab 1: no `import` in answer cells, don't add/delete/copy cells | The setup cell already imports `numpy as np`. If it follows the lecture it will also import `scipy.optimize` and `matplotlib`, but **read the setup cell to confirm** (I haven't seen it). Just use what's there. |

**Randomness warning.** Search algorithms use random numbers, so a test can't check an exact answer. If a test *is* seeded and compares exact numbers, the **order and number of your random calls** matters, so follow the lecture's structure exactly (sample, then evaluate, then compare). Never call `np.random.seed` inside your own function.

---

## Part 2: Predicted task types

Ranked by how likely I think they are, each with the block that covers it.

| # | Likely task | Skill | Block |
|---|---|---|---|
| 1 | Write an objective function (line-fit sum of squares, distance to targets, etc.) | return a **scalar**, depends on θ | 2 |
| 2 | Turn a maximise problem into a minimise one | negate | 1 |
| 3 | Add a penalty for a constraint | `np.where` / `np.maximum` | 3 |
| 4 | Solve least squares directly | `pinv`, `lstsq`, `solve` | 5 |
| 5 | Call `scipy.optimize.minimize`, read `.x` and `.fun` | usage | 6 |
| 6 | Implement grid search / random search | loop + best tracking | 7 |
| 7 | Implement hill climbing | neighbour + accept if better | 8 |
| 8 | Implement simulated annealing: acceptance probability and temperature schedule | formula → code | 9 |
| 9 | Genetic algorithm pieces: selection, crossover, mutation | `argsort`, `np.where` | 10 |
| 10 | Short concept answers (convex? which method? why did it get stuck?) | theory | 4, 11, Part 6 |

---

## Part 3: Core ideas, in order

### Block 1: What optimisation is (10 min)

**5-year-old version.** You are tuning a radio. The knobs are what you can change. "How bad does it sound?" is one number. You keep turning knobs until that number is as small as possible.

**Maths.**
- **Parameters** θ: the things you may adjust (a vector, e.g. `[angle, strength]`). The set of all possible θ is the *parameter space*.
- **Objective function** L(θ): takes θ, returns **one number** = how bad. Also called loss, cost, fitness, energy.
- **Goal:** `θ* = argmin L(θ)`. "argmin" means "the θ that gives the smallest L", not the smallest L itself.
- **Constraints:** limits on θ (knob only goes 0–10). The allowed region is the **feasible set**.
- **Maximise?** Minimise `-L` instead. The lecture's stone throw returns `-np.abs(pos[0])` so a longer throw gives a *lower* number.
- **Common shape:** `L(θ) = ‖f(x; θ) − y‖`, the distance between what your model outputs and the reference. Only θ changes. The data x and y stay fixed.

**Code habit.** A loss function takes a **1D NumPy array** θ and returns a **single float**.

```python
def quad_loss(theta):
    return (theta[0] - 3)**2 + (theta[1] + 1)**2     # minimum at [3, -1]

def neg_quad(theta):          # "maximise quad" becomes "minimise the negative"
    return -quad_loss(theta)
```

**Check 1:** A function returns a NumPy array of shape `(1,)` instead of a float. Why can that be a problem for an optimiser or a test?

### Block 2: Writing objective functions (15 min)

**5-year-old version.** Guess the answer, measure how wrong you are, add up the wrongness. Smaller total = better guess.

**Pattern A: line fit** `y = m·x + c`, θ = `[m, c]`, loss = sum of squared errors:

`L(θ) = Σᵢ (m·xᵢ + c − yᵢ)²`

```python
def line_loss(theta, x, y):
    m, c = theta                      # unpack
    return np.sum((m * x + c - y) ** 2)
```
Shapes: `x`, `y` are `(N,)`, so `m*x + c - y` is `(N,)` and `np.sum` collapses it to a scalar.

**Pattern B: geometric median.** The point closest (in total distance) to a set of points. θ = a 2D position.

`L(θ) = Σᵢ ‖θ − xᵢ‖₂`

```python
def median_loss(theta, targets):               # targets: (N, 2), theta: (2,)
    return np.sum(np.linalg.norm(targets - theta, axis=1))
```
`targets - theta` broadcasts `(N,2) − (2,)` → `(N,2)`. `norm(..., axis=1)` gives one distance per **row** → `(N,)`. `np.sum` → scalar. (Same `axis` rule as Lab 1: axis=1 means "one answer per row".)

> **Lecture bug, don't copy it.** The lecture's `loss(x)` for this example is
> `np.sum(np.linalg.norm(x[:,None] - targets.T, axis=1, ord=np.inf))`.
> I ran it: it builds a `(2, 50)` array, takes the norm along axis 1, giving shape `(2,)`, so it returns the sum over the two *coordinates* of the largest deviation. It also uses `ord=np.inf` while the text says L2. On my test data it returned 4.75 where the real sum of distances is 63.5. The optimiser still "works" because it just minimises whatever it is given. Use the version above.

**Pattern C: evenly spaced points.** Pack many 2D points into one long θ, reshape inside the loss:

```python
def spacing_loss(theta, shape=(64, 2), alpha=0.8):
    p = theta.reshape(shape)                                   # back to (64, 2)
    d = np.linalg.norm(p[:, :, None] - p[:, :, None].T, axis=1) # (64, 64) pairwise distances
    return np.sum((alpha - d) ** 2)
```
Idea to remember: **a whole configuration of points is one point in a big vector space** (64 points × 2 = a 128-dimensional θ). The optimiser needs a 1D array, which is why you `ravel()` going in and `reshape` coming out.

**Check 2:** In `median_loss`, what happens if you write `axis=0` instead of `axis=1`?

### Block 3: Constraints and penalties (20 min)

**5-year-old version.** Two ways to stop a child going in the road: build a fence (hard constraint), or say "you lose a sweet for every step you take towards the road" (penalty).

**Maths.**
- Equality constraint `c(θ) = 0`: θ must sit **on a surface** (e.g. `‖θ‖₂ − 1 = 0` → unit sphere).
- Inequality constraint `c(θ) ≤ 0`: θ must sit **inside a region** (e.g. `‖θ‖∞ − 10 ≤ 0` → box from −10 to 10).
- **Box constraint:** each parameter has its own min and max. The easiest and most widely supported kind.
- **Hard (constrained optimiser):** guarantees the answer is legal; fewer algorithms support it.
- **Soft (penalty):** `L'(θ) = L(θ) + λ(θ)`, where `λ(θ)` is 0 when legal and grows the more you break the rule. Any optimiser works; but it can still return an illegal answer if the penalty is too weak.
- Turning a hard constraint into a penalty is called **relaxation**.

**Code: the lecture's penalty** (start punishing at 90% of the limit):

```python
def penalty(strength, limit=1.0, k=500.0):
    cutoff = limit * 0.90
    return np.where(strength > cutoff, k * (strength - cutoff) ** 2, 0.0)

def throw_penalised(theta):
    return throw(theta) + penalty(theta[1])      # original loss + penalty
```
Tested values with `limit=1`: strength 0.5 → 0, 0.95 → 1.25, 1.0 → 5, 1.5 → 180. Notice the penalty grows **quadratically**, so big violations hurt far more than small ones.

**Box penalty for every parameter** (general, vectorised):

```python
def box_penalty(theta, lo=0.0, hi=1.0, k=100.0):
    over  = np.maximum(0, theta - hi)      # how far above the top
    under = np.maximum(0, lo - theta)      # how far below the bottom
    return k * np.sum(over**2 + under**2)
```
`np.maximum(0, arr)` is elementwise, which is why it works on a whole vector. Don't use `max(0, arr)` (plain Python `max` on an array raises an error).

**Hard-constraint version with scipy** (bounds go straight into the optimiser, see Block 6):
`scipy.optimize.minimize(throw, [65.0, 0.1], method="tnc", bounds=[[0, 90], [0, max_strength]])`

**Check 3:** Give one advantage and one disadvantage of a penalty compared with a hard constraint.

### Block 4: Properties of objective functions (15 min, theory)

**5-year-old version.** A bowl has one bottom. A bumpy field has many dips; one is the deepest, and a ball can get stuck in a shallow one.

| Word | Meaning | Why you care |
|---|---|---|
| **Local minimum** | Every nearby move makes L bigger | Local search can get trapped here |
| **Global minimum** | The lowest value anywhere | The real goal |
| **Convex** | One minimum only (a bowl) | Finding *any* minimum = finding *the* minimum. You can stop. |
| **Non-convex** | Several dips | Local search may stop in the wrong one; the result depends on the starting point |
| **Convex optimisation** | Convex L **and** convex constraints | Very efficient special solvers (linear programming, quadratic programming) exist |
| **Continuous** | Tiny change in θ → tiny change in L | Lets "look near here" (locality) work |
| **Discontinuous** | L can jump | Local search isn't guaranteed to work; much harder |
| **Discrete** optimisation | θ takes separate values (e.g. orderings) | Usually harder than continuous |

How the lecture knows least squares is convex: it's a quadratic (no powers above 2), and a quadratic has at most one minimum.

**Check 4:** Your loss has three valleys. You run hill climbing from two different starting points and get two different answers. Why?

### Block 5: Linear least squares, the direct method (25 min)

**5-year-old version.** Draw the straight line that is, overall, closest to all the dots. There's a formula that gives the answer in one go, no searching.

**Maths.** Minimise `L(x) = ‖A x − y‖₂²`. Setting the slope to zero gives the **normal equations**:

`(AᵀA) x = Aᵀ y`  →  `x* = (AᵀA)⁻¹ Aᵀ y = A⁺ y`   (A⁺ is the **pseudo-inverse**)

For a line `y = m·x + c`, build `A` with one column per parameter: a column of x's and a column of 1's (the 1's multiply c).

```python
A = np.stack([x, np.ones_like(x)]).T     # shape (N, 2); columns [x, 1]  → theta = [m, c]
```

Four equivalent ways (I ran them and they agree to numerical precision):

```python
theta = np.linalg.pinv(A) @ y                       # pseudo-inverse (what the lecture does)
theta = np.linalg.solve(A.T @ A, A.T @ y)           # normal equations directly
theta = np.linalg.lstsq(A, y, rcond=None)[0]        # purpose-built; [0] picks the solution
theta = np.polyfit(x, y, 1)                         # returns [m, c] for degree 1
```

> **Order trap.** The lecture builds `A = np.stack([line_x**0, line_x**1]).T`, so columns are `[1, x]` and the answer comes out as `[c, m]` (offset first). That's why its plot code uses `regression[1]` for the gradient. But its `loss(theta)` uses `theta = [m, c]`. Two conventions in one notebook. **Always check which parameter is first before comparing against `[m, c]`.**

Quadratic or higher: `np.vander(x, 3)` gives columns `[x², x, 1]`, so the answer is `[a, b, c]` for `a·x² + b·x + c`.

```python
coef = np.linalg.lstsq(np.vander(x, 3), y, rcond=None)[0]
```

Shape check: `A` is `(N, p)`, `y` is `(N,)`, the answer is `(p,)`.

**Check 5:** For 20 data points and the model `y = m·x + c`, what shape is `A` and what shape is the result?

### Block 6: `scipy.optimize.minimize` (20 min)

**5-year-old version.** A ready-made robot that turns the knobs for you. You give it the "how bad" function and a starting guess.

```python
res = scipy.optimize.minimize(L, x0, method="Nelder-Mead")
res.x        # best θ found (1D array)
res.fun      # L at that θ  (a float)
res.success  # did it say it converged?
res.nit      # iterations used
```

- `x0`: starting guess. Use floats: `np.array([0.0, 0.0])` or `[65.0, 0.1]`. The result can depend on it for non-convex problems.
- Extra fixed data goes in `args`: `minimize(f, x0, args=(a, b))` calls `f(theta, a, b)`. Or use a `lambda`: `lambda t: median_loss(t, pts)`.
- `bounds=[(lo, hi), (lo, hi)]` is a hard box constraint. Needs a method that supports it: `"L-BFGS-B"`, `"TNC"`.
- **Methods:** `Nelder-Mead` needs only L values (a *zeroth-order* method). `BFGS` uses slopes (*first-order*; it estimates them if you don't supply them, and that's next week's topic).
- Maximising? Minimise `-L`, then report `-res.fun` (the lecture prints `np.abs(res.fun)`).
- Many variables? Flatten: `minimize(loss, theta.ravel())`, then `res.x.reshape(theta.shape)`.

Tested: line fit with Nelder-Mead gave `[2.0168, 0.9885]`, identical to the least-squares answer. With `bounds=[(0,1.5),(-1,3)]` the slope gets pinned at the 1.5 limit.

**Check 6:** `res = minimize(neg_quad, [0.0, 0.0], method="Nelder-Mead")` for the objective `neg_quad(θ) = −quad_loss(θ)`. Is that a good idea? (Hint: look at what `neg_quad` is.)

---

## Part 4: Iterative search algorithms (the heart of the lab)

**Everything here is zeroth-order**: the algorithm only ever *asks for L(θ)*, never the slope. Use these when you know nothing else about the problem.

### The one skeleton to memorise

Every algorithm below is this loop. Only the two marked lines change.

```python
def search(L, ..., iters):
    best_theta = <start guess>
    best_loss  = L(best_theta)
    trace      = [best_loss]                  # best-so-far after each step
    for i in range(iters):
        proposal = <PROPOSE>                  # <-- differs per algorithm
        l = L(proposal)
        if <ACCEPT>:                          # <-- differs per algorithm
            best_theta, best_loss = proposal, l
        trace.append(best_loss)
    return best_theta, best_loss, np.array(trace)
```

| Algorithm | PROPOSE | ACCEPT |
|---|---|---|
| Random search | `sample_fn()` | `l < best_loss` |
| Hill climbing | `neighbour_fn(best_theta)` | `l < best_loss` |
| Simulated annealing | `neighbour_fn(state)` | `l < loss` **or** random coin flip with probability `exp(-(l-loss)) * T(i)` |

**The four "metaheuristics" the lecture lists** (each is one idea you can bolt on):

| Idea | Plain meaning | Needs | Used by |
|---|---|---|---|
| **Locality** | Look *near* where you are | continuity | hill climbing, annealing |
| **Temperature** | Early on, allow some bad moves | local minima exist | simulated annealing |
| **Population** | Keep many candidates at once | none | genetic algorithm, ants |
| **Memory** | Remember good/bad places | none | ant colony |

### Block 7: Grid search and random search (25 min)

**Grid search, 5-year-old version.** Check every square on a chessboard, one by one.

Divide each parameter's range into `divs` equal points; try every combination; keep the best. Cost: **`divs ** d` evaluations** for `d` parameters. 8 points per dimension in 100 dimensions is `8**100 ≈ 2×10⁹⁰` evaluations. That's the *curse of dimensionality* ("exponential in the number of dimensions").

```python
def grid_search(L, ranges, divs):
    axes  = [np.linspace(lo, hi, divs) for lo, hi in ranges]
    grids = np.meshgrid(*axes, indexing="ij")
    pts   = np.stack([g.ravel() for g in grids], axis=1)     # (divs**d, d): every combination
    best_theta, best_loss = None, np.inf
    for theta in pts:                                        # theta is a 1D array
        l = L(theta)
        if l < best_loss:
            best_theta, best_loss = theta, l
    return best_theta, best_loss
```
The lecture uses `itertools.product(*axes)` instead. If `itertools` isn't already imported in the lab's setup cell you **cannot import it in your answer**, which is why I give the `meshgrid` version. Note `itertools.product` yields **tuples**, and `tuple * 2.0` raises `TypeError`, so wrap with `np.array(theta)` before calling `L`.

Tested: 15×15 grid on the line problem → `[2.14, 0.71]`, loss 3.2. It's the right neighbourhood but only as precise as the grid spacing.

Grid search cons (lecture): hugely inefficient, must choose bounds in advance, biased towards the early "corners", depends on number of divisions, can step right over a narrow minimum. Pros: works on any continuous space, needs no knowledge of L, trivial to code.

**Random search, 5-year-old version.** Throw darts blindly, remember the best one.

```python
def random_search(L, sample_fn, iters):
    best_theta, best_loss = None, np.inf
    trace = []
    for _ in range(iters):
        theta = sample_fn()                  # e.g. lambda: np.random.normal(0, 5, size=(2,))
        l = L(theta)
        if l < best_loss:
            best_theta, best_loss = theta, l
        trace.append(best_loss)
    return best_theta, best_loss, np.array(trace)
```
- `sample_fn` is a function with **no arguments** that returns one random θ. You *call* it: `sample_fn()`.
- Pros: can't get trapped in a local minimum (no structure used), trivial, almost always beats grid search. Cons: extremely inefficient, no guarantee it improves in any given step.
- Joke algorithm in the lecture: **bogosort** (shuffle until sorted) is random search over orderings, `O(n!)`.

**Hyperparameters** = settings of the *optimiser* (grid: ranges and divs; hill climbing: step size) as opposed to the *parameters* θ of the objective. A perfect optimiser would have none.

**Check 7:** 5 parameters, 10 grid points each. How many calls to `L`?

### Block 8: Hill climbing (20 min)

**5-year-old version.** Blindfolded on a hill, you take a small random step. If it's lower, keep it; if not, step back. Repeat.

Same as random search but proposals come from a small random nudge around the current best:

```python
def hill_climbing(L, guess_fn, neighbour_fn, iters):
    best_theta = guess_fn()
    best_loss  = L(best_theta)
    trace = [best_loss]
    for _ in range(iters):
        proposal = neighbour_fn(best_theta)        # nudge the CURRENT best
        l = L(proposal)
        if l < best_loss:
            best_theta, best_loss = proposal, l
        trace.append(best_loss)
    return best_theta, best_loss, np.array(trace)

guess     = lambda: np.random.normal(0, 5, size=(2,))
neighbour = lambda t: t + np.random.normal(0, 0.25, t.shape)      # small Gaussian nudge, same shape as t
```
Tested: 800 iterations on the line problem → `[2.018, 0.993]`, loss 1.159, and the trace never went up.

- **Step size** (here 0.25) is the hyperparameter. Too small → crawls (slow progress). Too large → bounces around (noisy). The lecture shows 0.01 … 2.0 side by side.
- **Cons:** gets stuck in local minima, struggles on plateaus (flat regions), stuck behind ridges, needs continuity.
- **Fixes:** *multiple restarts* (run again from new random starts, keep the best); *adaptive step size* (grow the step if no improvement for a while).

**Aliasing trap (silent bug).** If you ever modify θ in place after saving it, your "best" changes too:
```python
best = t;        t += 1    # best is now ALSO changed  (same array)
best = t.copy(); t += 1    # safe
```
`t + noise` creates a new array, which is why the nudge is safe. `t += noise` is not.

**Check 8:** Why does hill climbing call `neighbour_fn(best_theta)` and random search call `sample_fn()` with no input?

### Block 9: Simulated annealing (30 min)

**5-year-old version.** Same blindfolded walk, but at the start you're "wobbly and hot": you sometimes accept a step *up*, which lets you hop out of a small dip. As you "cool down", you stop doing silly uphill steps.

**Maths (the lecture's version).** Current loss `l`, proposed loss `l'`, iteration `i`:

- if `l' < l`: accept with probability **1** (always go downhill)
- otherwise: accept with probability `P = exp(-(l' − l)) · T(i)`
- temperature schedule `T(i) = exp(-i / r)`, `r` = cooling rate (bigger `r` = cools slower)

Reading it: `l' − l` is "how much worse". The worse the move, the smaller `exp(-…)`; the later in the run, the smaller `T`. So **bad moves are rare when they're very bad or when it's late.**

> **Formula caveats (read these).**
> 1. The lecture's markdown writes `e^{-(l-l')}`, which would be *bigger than 1* for an uphill move. The code is correct: `np.exp(-(proposal_loss - loss)) * temperature_fn(i)`. Use the code.
> 2. The textbook version divides inside the exponent, `exp(-(l'−l)/T)`. This lecture **multiplies** by T instead. If the lab gives a formula, follow the lab. If not, follow the lecture.

Numbers I computed (loss 3 → 4 with T = 0.5): `P = e^-1 · 0.5 = 0.184`. Loss 3 → 2: `P = 1`. Equal loss (3 → 3): `e^0 · 0.5 = 0.5`.

```python
def temperature(i):
    return np.exp(-i / 100.0)          # i=0 → 1.0 ; i=100 → 0.37 ; i=300 → 0.05

def accept_prob(loss, new_loss, T):
    if new_loss < loss:
        return 1.0
    return np.exp(-(new_loss - loss)) * T

def simulated_anneal(L, guess_fn, neighbour_fn, temperature_fn, iters):
    state = guess_fn()
    loss  = L(state)
    best_theta, best_loss = state, loss
    trace = [best_loss]
    for i in range(iters):
        proposal = neighbour_fn(state)                 # nudge the CURRENT STATE, not the best
        new_loss = L(proposal)
        if np.random.uniform(0, 1) < accept_prob(loss, new_loss, temperature_fn(i)):
            state, loss = proposal, new_loss           # move (even if uphill)
        if loss < best_loss:
            best_theta, best_loss = state, loss        # remember best EVER seen
        trace.append(best_loss)
    return best_theta, best_loss, np.array(trace)
```

Two things differ from hill climbing, and a marker looks for both:
1. The walker (`state`) can move uphill, so you need **separate variables** for "where I am" (`state`) and "best ever" (`best_theta`).
2. The coin flip: `np.random.uniform(0,1) < p`. With `p = 1` it always passes (uniform is < 1), so the downhill case needs no special branch.

Tested: 1000 iterations on the line problem → `[2.017, 0.987]`, loss 1.1587. The line problem is convex, so annealing gains nothing over hill climbing there (the lecture says exactly this). It pays off on **non-convex** problems like the evenly-spaced points.

- **Pros:** much less likely to be trapped than hill climbing, simple, empirically strong. **Cons:** needs a good temperature schedule and neighbourhood (more hyperparameters), no convergence guarantee, slow if uphill steps aren't needed.
- Origin: designed at IBM for chip-layout routing. Name from annealing metal (cool slowly → better structure).

**Check 9:** `loss = 5`, `new_loss = 6`, `T = 1`. What is the acceptance probability? What if `T = 0`?

### Block 10: Genetic algorithm (25 min)

**5-year-old version.** Keep a zoo of candidate answers. The best quarter get to have babies (mixing their features, with small random mutations); everyone else is replaced by babies.

Vocabulary: **population** (many θ's), **genotype** (the θ vector), **selection** (keep the best), **crossover** (mix two parents), **mutation** (small random change).

```python
def genetic_search(L, pop, guess_fn, mutation_fn, iters, keep=0.25):
    population = np.array([guess_fn() for _ in range(pop)])   # (pop, d)
    d    = population.shape[1]
    top  = int(pop * keep)                                    # how many survive
    trace = []
    for _ in range(iters):
        losses = np.array([L(p) for p in population])
        order  = np.argsort(losses)                           # best (lowest) first
        losses, population = losses[order], population[order] # sort BOTH the same way
        trace.append(losses[0])                               # best this generation
        for j in range(top, pop):                             # replace everyone outside the top group
            mum  = np.random.randint(0, top)                  # random survivor index
            dad  = np.random.randint(0, top)
            mask = np.random.randint(0, 2, d)                 # random 0/1 per dimension
            child = np.where(mask == 0, population[mum], population[dad])   # crossover
            population[j] = mutation_fn(child)                              # mutation
    return population[0], losses[0], np.array(trace)
```
Key lines:
- `np.argsort(losses)` returns the *indices* that would sort the array. Use them to reorder `losses` **and** `population` together. Forgetting to reorder `population` is the classic bug.
- `np.where(mask == 0, mum_vec, dad_vec)`: per dimension, take from mum where mask is 0, else from dad. That is **crossover**.
- `mutation_fn` can be the same as the hill-climbing `neighbour` (add small Gaussian noise).
- `np.random.randint(0, top)` excludes `top`, so the index is always a valid survivor.

Cost: **`pop × iters`** evaluations of L (one per member per generation), which is a lot.

Tested: pop 20, 200 iterations on the line problem → loss 1.1588, and the best-so-far trace never rose.

- **Pros:** easy, works on discrete and continuous, tolerant of local minima, very flexible. **Cons:** lots of hyperparameters (population size, keep fraction, mutation size, crossover rule…), no convergence guarantee, many evaluations.
- Crossover works when θ splits into meaningful parts; on a tangled vector it just averages into mush.

**Check 10:** With `pop=40`, `keep=0.25`, how many individuals survive each generation, and how many new children are made?

### Block 11: Memory, convergence, and what goes wrong (20 min, theory)

**Ant colony optimisation (memory + population).** Ants wander, and when one finds food it leaves a **pheromone** trail on the way back; other ants follow strong trails; trails **evaporate** over time so the colony doesn't lock onto one spot. In optimisation: a population of candidates plus a data structure remembering good paths. Idea name: **stigmergy** (indirect coordination through traces left in the environment). Good for route/path-finding. Cons: complex, even more hyperparameters, no convergence guarantee. Algorithms without memory are **memoryless** and may re-check the same places.

**Convergence.** An algorithm *converges* when it settles on a solution. In convex problems that is the global minimum; in non-convex ones it might be a local minimum it cannot leave. Heuristics (random search, annealing, GA) have **no guarantee** of converging. Diagnose with a **plot of loss against iteration**: you want it to fall steeply and then flatten.

**What the loss plot tells you**

| Symptom | Cause | Fix |
|---|---|---|
| Falls very slowly | Steps too small | Bigger step / neighbourhood |
| Noisy, bounces, or blows up | Steps too big, or an "abyss" (L keeps decreasing forever) | Smaller step; add constraints |
| Flat for a long time | Plateau, or stuck in a local minimum | Momentum/memory; random restarts; annealing |
| Stalls where slopes cancel | Saddle point (down one way, up another) | Next week's gradient methods |
| Hopeless on cliffs/jumps | Steep or discontinuous L | Stochastic methods smooth it out |

**"Use the right algorithm" (the lecture's decision order)**
1. It's **least squares** → solve directly (`pinv` / `lstsq`).
2. It's **convex** → use a convex solver (radically more efficient).
3. You can get **derivatives** (or automatic differentiation) → use a **first-order** method (next week).
4. Otherwise → a general **zeroth-order** method: simulated annealing or a genetic algorithm.

**Check 11:** You see the loss drop for 50 iterations then sit perfectly flat for 4,000. Name two possible causes and one fix.

---

## Part 5: One-page comparison (memorise this table)

| Method | Idea | Gets stuck in local minima? | Needs continuity? | Main hyperparameters | Evaluations of L |
|---|---|---|---|---|---|
| Least squares | Solve normal equations | n/a (convex) | n/a | none | 1 solve |
| Grid search | Try every grid point | No, but may miss | No | ranges, `divs` | `divs**d` |
| Random search | Guess, keep best | No | No | sampler, `iters` | `iters` |
| Hill climbing | Nudge, keep if better | **Yes** | **Yes** | step size, `iters` | `iters` |
| Simulated annealing | Hill climb + allow uphill early | Less likely | Mostly | step size, `T(i)`, `iters` | `iters` |
| Genetic algorithm | Population + crossover + mutation | Somewhat robust | No | `pop`, `keep`, mutation, `iters` | `pop × iters` |
| Ant colony | Population + pheromone memory | Somewhat | No | many | varies |

---

## Part 6: Concept answers in one line each (for short-answer questions)

- **Optimisation is search**, using structure in the problem to search efficiently.
- **Why one algorithm for everything?** Formulate the problem as parameters + objective (+ constraints) and a generic algorithm applies. "We specify the problem, not the solution."
- **Why is L usually "expensive"?** Long computation, a real-world experiment, danger, or paid data, so good optimisers use few evaluations.
- **Soft vs hard constraint:** soft = penalty added to L (any optimiser, may be violated); hard = built into the optimiser (guaranteed, fewer optimisers).
- **Why is grid search bad in high dimensions?** Cost `divs**d`, exponential in the number of parameters.
- **Why does random search never get "trapped"?** It uses no local structure. The price is that it also exploits none.
- **Why does hill climbing depend on the start point?** Non-convex L has several basins; it stays in whichever it starts in.
- **What does temperature do?** Controls how often an uphill move is accepted: high early (explore), low late (exploit).
- **Exploration vs exploitation:** wander widely to find good regions vs dig into a known good region. Annealing and ants both trade these off over time.
- **Hyperparameter vs parameter:** parameters are what the optimiser tunes in L; hyperparameters tune the optimiser itself.

---

## Part 7: Drills (do these cold, then check against Part 10)

These are parallel exercises, not the real lab questions. Write each in a scratch notebook, not the lab file. Assume `import numpy as np` is already done.

1. **Loss.** Write `quad_loss(theta)` for `L = (θ₀ − 3)² + (θ₁ + 1)²`. Then write the version to *maximise* it.
2. **MAE loss.** `mae(theta, x, y)` for the line `y = θ₀·x + θ₁` using the **mean absolute error**.
3. **Box penalty.** `box_penalty(theta, lo, hi, k)` that is 0 when every parameter is inside `[lo, hi]` and `k × (sum of squared violations)` otherwise. Vectorised, no Python loop.
4. **Quadratic fit.** Given arrays `x`, `y`, find `[a, b, c]` for `y ≈ a·x² + b·x + c` in one step.
5. **Random search** for 2 parameters sampled uniformly from `[-5, 5]`. Return `(best_theta, best_loss)`. Run it on `quad_loss`.
6. **Hill climbing** that also returns how many proposals were accepted.
7. **Schedules.** Write three `T(i)`: exponential decay with `r = 100`, geometric `0.95**i`, and linear from 1 to 0 over 1000 iterations (never below 0).
8. **Cost counting.** A grid search has 5 parameters and 10 divisions each. Evaluations? What if you add one more parameter?
9. **Acceptance.** `accept_prob(loss, new_loss, T)` using the lecture rule. Evaluate it for `(3, 2, 0.5)`, `(3, 4, 0.5)`, `(3, 3, 0.5)`.
10. **Crossover.** `crossover(mum, dad)` that picks each element from either parent at random (use `np.where`).
11. **Wire it up.** Use `scipy.optimize.minimize` on `quad_loss` from `[0.0, 0.0]` with box bounds `[(0, 2), (-0.5, 5)]`. What `res.x` do you expect, and why?
12. **Debug.** This hill climber has a bug. Find it.
    ```python
    best = guess_fn()
    for _ in range(iters):
        best += np.random.normal(0, 0.1, best.shape)
        if L(best) < best_loss:
            best_loss = L(best)
    ```

---

## Part 8: Lab-day checklist

Before you start
- [ ] Open **your slot's** notebook only. Run the setup cell first.
- [ ] Task 0: 8-character `student_id` and your typed signature. Run its test cell.

For every task
- [ ] Read the whole question *and the hint* (hints name the trap).
- [ ] Note the required **shape/type** of each output variable.
- [ ] Replace **both** `# YOUR CODE HERE` and `raise NotImplementedError()`. No `import` lines.
- [ ] Run the visible test. Then ask: *would this still work on different inputs?* (hidden tests). No hard-coded numbers from the visible example.
- [ ] Don't touch "provided" cells or test cells.

Common ways to lose marks (specific to this lab)
- [ ] Loss returns an array or tuple instead of a float.
- [ ] Used `*` where `@` was needed, or `axis=0` where `axis=1` was needed (norm per row).
- [ ] `max(0, arr)` instead of `np.maximum(0, arr)`.
- [ ] Forgot to negate when maximising, or forgot to flip the sign back when reporting.
- [ ] Reordered `losses` but not `population` after `argsort`.
- [ ] Saved `best = theta` then mutated `theta` in place (use `.copy()` or build new arrays).
- [ ] Used a tuple (from `itertools.product`) as θ without `np.array(...)`.
- [ ] Least-squares column order `[1, x]` vs `[x, 1]` giving `[c, m]` vs `[m, c]`.
- [ ] In annealing, compared against `best_loss` instead of the current `loss` for the acceptance rule; or used a single variable for both state and best.
- [ ] Called `sample_fn` instead of `sample_fn()` (forgot the brackets).
- [ ] Compared floats with `==`. Use `np.isclose` / `np.allclose`.

At the end
- [ ] **Kernel → Restart Kernel and Run All Cells.** No red errors anywhere.
- [ ] Check the marks summary at the bottom.
- [ ] Save, then upload the `.ipynb` to Moodle.

`History` helper (from `utils/history.py`), in case the lab provides it and you need to read it:
`o = History()` → `o.track(theta, loss)` after every evaluation (it keeps the best so far in `o.best_theta` / `o.best`) → `return o.finalise()`, which gives an object with `.theta`, `.loss`, `.loss_trace`, `.all_theta`, `.all_loss`, `.best_losses`. If the lab says "return a History", use that. Otherwise my plain-tuple versions are fine, but **match whatever return type the question asks for.**

---

## Part 9: Check-question answers

1. Tests and optimisers expect a scalar. A `(1,)` array may fail an `isinstance`/shape assert, or break `<` comparisons and `np.isclose` assumptions. Return `float(...)` or use a scalar-producing reduction like `np.sum`.
2. `axis=0` takes the norm down each *column* of the `(N, 2)` array, giving 2 numbers (one per coordinate) instead of N distances. The loss would then be wrong, and it wouldn't be the geometric median.
3. Advantage: any optimiser works / handles genuinely soft limits. Disadvantage: the answer can still violate the constraint (no guarantee), and the penalty can be hard to design. (Hard constraint: guaranteed legal but fewer optimisers.)
4. Hill climbing only looks locally and the function is non-convex, so each start rolls into the nearest valley. The result depends on initial conditions.
5. `A` is `(20, 2)`; result is `(2,)`.
6. No. `neg_quad` has no minimum: it decreases forever as θ grows (the "abyss"). Nelder-Mead will run away or hit its iteration limit. To maximise `quad_loss`, you'd need constraints; to *minimise* `quad_loss` just pass `quad_loss`. (The negate trick is for a function that has a peak, like the stone-throw distance.)
7. `10**5 = 100,000`.
8. Random search can sample anywhere, so it needs no starting point. Hill climbing is *local*: it needs a current best to nudge around.
9. `exp(-(6-5)) · 1 = 0.368`. With `T = 0`, probability 0, so it never accepts an uphill move (pure hill climbing).
10. 10 survive (`int(40 × 0.25)`), 30 children replace the rest.
11. Causes: stuck in a local minimum; on a plateau; step size too small to escape; (annealing: temperature already cooled to ~0). Fix: random restarts, annealing/higher temperature, larger step, or population/memory.

---

## Part 10: Drill solutions (tested)

```python
# 1
def quad_loss(theta):
    return (theta[0] - 3)**2 + (theta[1] + 1)**2
def neg_quad(theta):
    return -quad_loss(theta)

# 2
def mae(theta, x, y):
    return np.mean(np.abs(theta[0] * x + theta[1] - y))

# 3
def box_penalty(theta, lo, hi, k):
    over  = np.maximum(0, theta - hi)
    under = np.maximum(0, lo - theta)
    return k * np.sum(over**2 + under**2)
# box_penalty(np.array([1.5, -0.5]), 0, 1, 100) -> 50.0 ; inside the box -> 0.0

# 4
coef = np.linalg.lstsq(np.vander(x, 3), y, rcond=None)[0]       # [a, b, c]
# same as np.linalg.pinv(np.vander(x, 3)) @ y

# 5
def random_search(L, sample_fn, iters):
    best_theta, best_loss = None, np.inf
    for _ in range(iters):
        t = sample_fn()
        l = L(t)
        if l < best_loss:
            best_theta, best_loss = t, l
    return best_theta, best_loss
# random_search(quad_loss, lambda: np.random.uniform(-5, 5, 2), 2000)  -> near [3, -1]

# 6
def hill_count(L, guess_fn, neighbour_fn, iters):
    best = guess_fn(); best_loss = L(best); accepted = 0
    for _ in range(iters):
        p = neighbour_fn(best); pl = L(p)
        if pl < best_loss:
            best, best_loss, accepted = p, pl, accepted + 1
    return best, best_loss, accepted

# 7
T_exp = lambda i, r=100.0: np.exp(-i / r)
T_geo = lambda i, a=0.95: a ** i
T_lin = lambda i, n=1000: max(0.0, 1 - i / n)       # plain max is fine here: i is a scalar

# 8   10**5 = 100000 ; adding one parameter -> 10**6 (x10 each time)

# 9
def accept_prob(loss, new_loss, T):
    if new_loss < loss:
        return 1.0
    return np.exp(-(new_loss - loss)) * T
# (3,2,0.5) -> 1.0 ; (3,4,0.5) -> 0.184 ; (3,3,0.5) -> 0.5

# 10
def crossover(mum, dad):
    mask = np.random.randint(0, 2, len(mum))
    return np.where(mask == 0, mum, dad)

# 11  minimum of quad_loss is [3, -1], which is outside the box.
# The best legal point is the nearest corner/edge: x = 2 (upper bound), y = -0.5 (lower bound).
# res = scipy.optimize.minimize(quad_loss, [0.0, 0.0], bounds=[(0, 2), (-0.5, 5)], method="L-BFGS-B")
# res.x -> [2., -0.5]

# 12  Bug: the code modifies best in place BEFORE checking whether it is better, and never
#     reverts a worse step; "best" and the proposal are the same array. Also best_loss is
#     never initialised. Correct version:
best = guess_fn(); best_loss = L(best)
for _ in range(iters):
    proposal = best + np.random.normal(0, 0.1, best.shape)     # new array
    l = L(proposal)
    if l < best_loss:
        best, best_loss = proposal, l
```

---

## Last-minute card (read this over breakfast)

- Loss: 1D array in, **one float** out. Maximise → negate.
- Penalty = `L + λ`, where λ is 0 when legal. `np.maximum`, `np.where`, squared violation.
- Least squares: `pinv(A) @ y` or `lstsq`. Check column order.
- `minimize(L, x0, method=..., bounds=...)`; read `.x` and `.fun`.
- Every search = *start → propose → score → accept?* Only propose and accept change.
- Hill climbing nudges the **best**. Annealing nudges the **state** and accepts uphill with `exp(-(l'-l)) * T(i)`, `T(i) = exp(-i/r)`.
- GA: `argsort` (reorder both arrays) → keep top fraction → `np.where(mask==0, mum, dad)` → mutate.
- Grid search costs `divs**d`. GA costs `pop × iters`.
- No imports, replace both lines, Restart & Run All, submit.
