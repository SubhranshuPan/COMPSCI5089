# Lab Prep Pack: Optimisation II (gradient descent), assessed tomorrow 9-10 am

How to use this: go top to bottom. Every block is **plain-language version → maths → code → one check question**. Check-question answers and drill solutions are in Part 8. Every code snippet in this file was run before going in; the numbers quoted are real outputs.

**What I have and haven't seen**
- I read the whole Lecture 5 notebook and your completed `lab3.ipynb`.
- I have **not** seen tomorrow's lab notebook. The task list in Part 2 is a prediction from the lecture and from how Lab 3 was built. Practise the *skills*; don't expect these exact questions.
- `autograd` is **not installed** on this machine, and Lab 3 says "do not import additional packages". So I expect you to write gradients **by hand**. That is why Block 5 is the most important block in this file.

---

## Part 0: Plan (about 4 hours, then sleep)

| Time | Block | Goal |
|---|---|---|
| 10 min | 1 | Why gradients beat guessing |
| 15 min | 2 | Derivative, gradient, which way is downhill |
| 20 min | 3 | The gradient descent loop |
| 15 min | 4 | Step size (the thing that goes wrong) |
| **30 min** | **5** | **Writing a gradient by hand (most likely marks)** |
| 25 min | 6 | Line fit end to end on Lab 3's data |
| 15 min | 7 | Finite differences (the "check my gradient" tool) |
| 15 min | 8 | Autograd, Jacobian, Hessian: vocabulary only |
| 20 min | 9 | Momentum and random restart |
| 20 min | 10 | Stochastic relaxation and SGD |
| 15 min | 11 | Continuity, Lipschitz, second-order: theory only |
| 60 min | Part 6 | Drills: close this file, retype from memory |
| 15 min | Part 7 | Exam-day checklist |

Break 10 min after every two blocks. If you're short on time, do **3, 4, 5, 6, 7** and skim the rest: that is where the code marks are.

---

## Part 1: What Lab 3 tells you about the format

Everything below comes from your completed `lab3.ipynb`. Tomorrow's lab will almost certainly use the same machinery.

| What Lab 3 did | What it means tomorrow |
|---|---|
| Task 0: `student_id` (8 chars) + signature, 0 marks | Do it first. Run the test cell. |
| 20 marks: **14 visible, 6 hidden** | A green tick is not the end. Write general code, not code that only fits the shown data. |
| Answer cell: replace **both** `# YOUR CODE HERE` and `raise NotImplementedError()` | Leave neither behind. |
| No imports in answer cells | `np` is already imported. Use it. |
| Function signatures are fixed by the plotting cells: `model(x, theta)`, `loss(y_true, y_pred)`, `guess()`, `neighbour(theta)` | **Keep names, argument order and return values exactly.** The provided plot cell calls your function with that signature. |
| `theta = [intercept, slope]`: `linear_model` was `theta[1]*x + theta[0]` | **Trap.** The lecture uses `[m, c]` (slope first). Always read the provided model before writing a gradient: `grad[0]` must match `theta[0]`. |
| Functions return `(theta, best_loss, history)`; `history` is a list of `(theta, loss)` tuples | Return exactly this shape if asked. |
| Tests like `history[i][1] >= history[i+1][1]` (loss never goes up) | Plain gradient descent with a sensible step satisfies this. With a step that's too big it won't. |
| Threshold tests: `best_loss < 5.0`, then `< 8.0`, `< 10.0` | "Increasingly strict" tests. The first is easy, later ones need a good step size or enough iterations. |
| `np.random.seed(42)` is set **outside** your function | Never call `np.random.seed` inside your own function. |
| "I RECOMMEND NOT TO WASTE TOO LONG ON THIS" hint | Time-box: 60 minutes total. If stuck for 8 minutes, move on and come back. |
| Final step: Restart Kernel and Run All, check the marks summary, submit | Do this 10 minutes before the end, not at the last second. |

---

## Part 2: Predicted task types

Ranked by how likely I think they are.

| # | Likely task | Skill | Block |
|---|---|---|---|
| 1 | Write the **gradient** of the MSE loss for the line (and maybe a polynomial) | Chain rule, hand-derived, `np.mean` | 5 |
| 2 | Write the **gradient descent loop** returning `(theta, loss, history)` | `theta = theta - lr*grad` | 3 |
| 3 | Pick a **step size / number of steps** so a loss threshold passes | Step-size behaviour | 4, 6 |
| 4 | Compute a gradient **numerically** (finite differences) | Central difference loop | 7 |
| 5 | Add **momentum** | One extra line with velocity `v` | 9 |
| 6 | **SGD / minibatch** version | Shuffle, slice, gradient on the slice | 10 |
| 7 | **Random restart** around gradient descent | Loop + keep the best | 9 |
| 8 | Compare with Lab 3's random search / hill climbing | Reading plots | 6 |
| 9 | Written/short answers: Jacobian shape, Hessian, Lipschitz, why second order fails | Theory | 8, 11 |

---

## Block 1: Why gradients? (10 min)

**Plain version.** You're blindfolded on a hill and want the lowest point. Random search = teleport to random spots and ask "am I lower?". Gradient descent = feel the slope under your feet and take a step downhill. Feeling the slope is much cheaper than teleporting a million times.

**Maths.** The lecture sorts algorithms by what they need:

| Order | Needs | Examples |
|---|---|---|
| zeroth | only `L(θ)` | random search, hill climbing, annealing, grid search |
| **first** | `L(θ)` and **`∇L(θ)`** | **gradient descent**, SGD, momentum |
| second | also the Hessian `∇²L(θ)` | Newton methods |

**Why not always use search?** Slow, no guarantee of progress, many hyperparameters. Deep networks have millions of parameters, so search is hopeless.

**Check Q1.** Is hill climbing zeroth-order or first-order? Why?

---

## Block 2: Derivative and gradient (15 min)

**Plain version.** The derivative is the slope of the ground at your feet. The gradient is the same thing when you can walk in many directions: one slope number per direction. **The gradient points uphill.** So to go downhill you walk the *opposite* way: that's the minus sign.

**Maths.**
- One parameter: `L(θ) = θ²` → `L'(θ) = 2θ`. At θ = 2 the slope is 4 (uphill to the right), so go left.
- Many parameters: `∇L(θ) = [∂L/∂θ₀, ∂L/∂θ₁, …]`. One entry per parameter, same length as θ.
- `∂L/∂θ₁` ("partial derivative") = how much `L` changes if you nudge **only** θ₁, holding the others fixed.

**Code.** The gradient is just a function from a vector to a vector of the same length:

```python
def loss(theta):   # vector -> scalar
    return (theta[0] - 3)**2 + 2*(theta[1] + 1)**2

def grad(theta):   # vector -> vector (same length as theta)
    return np.array([2*(theta[0] - 3), 4*(theta[1] + 1)])
```

At `[0, 0]` this gives `[-6, 4]`. I checked it against finite differences (Block 7): same answer.

**Check Q2.** `L(θ) = θ²`, you're at θ = 2. What is the gradient, and do you move left or right?

---

## Block 3: The gradient descent loop (20 min)

**Plain version.** Repeat: feel the slope, take a small step downhill. Stop after a fixed number of steps (or when the slope is almost flat).

**Maths.**

```
θ_next = θ − δ · ∇L(θ)       δ ("delta", lr) = step size
```

**Code.** Learn this skeleton by heart. It's the same shape as your Lab 3 `hill_climbing`:

```python
def gradient_descent(loss, grad, theta0, lr, steps):
    theta = np.array(theta0, dtype=float)          # float + copy
    history = [(theta.copy(), loss(theta))]
    for t in range(steps):
        theta = theta - lr * grad(theta)           # the ONLY new idea
        history.append((theta.copy(), loss(theta)))
    return theta, loss(theta), history
```

Three details that lose marks:
1. **`dtype=float`.** The lecture's version does `theta += ...` on `np.array(theta0)`. If you pass an int list like `[-10]` it crashes with `UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64')`. I reproduced this.
2. **`.copy()` when saving to history.** If you write `theta -= ...` (in place) and `history.append(theta)`, every history entry is the *same array*. I tested it: `all(a is hist[0] for a in hist)` is `True`. Use `theta = theta - ...` (a new array each time) **and** `.copy()`.
3. **Minus sign.** `theta + lr*grad` climbs the hill. If your loss grows, check this first.

On the bowl `(θ₀-3)² + 2(θ₁+1)²` from `[0,0]` with `lr=0.1` the first steps are loss 11 → 6.48 → 3.95 → 2.45, and after 100 steps it's at `[3, -1]` (the true minimum).

**Check Q3.** `L(θ) = θ²`, start θ = −10, δ = 0.25. What is θ after one step?

---

## Block 4: Step size (15 min)

**Plain version.** Walking down a valley: tiny steps take forever; giant steps jump across the valley and land higher on the other side.

**Maths (the lecture's example, `L = θ²`).** One step gives `θ − δ·2θ = (1 − 2δ)·θ`. So each step multiplies θ by `(1 − 2δ)`. That factor decides everything:

| δ | factor `1-2δ` | what happens (I ran these from θ = −10) |
|---|---|---|
| 0.01 | 0.98 | converges, slowly: **605 steps** |
| 0.25 | 0.5 | converges: **18 steps** |
| 0.5 | 0 | lands exactly on 0 in **1 step** |
| 0.98 | −0.96 | oscillates (flips sign each step) but still shrinks: 300 steps |
| 1.0 | −1 | **bounces ±10 forever** (never stops) |
| 1.5 | −2 | diverges, the numbers explode |

Rule: it works when `|1 − 2δ| < 1`, i.e. `0 < δ < 1` here. A steeper function has a smaller allowed δ. The lecture says this is tied to the **Lipschitz constant**: steeper (bigger K) means smaller safe step.

**Same thing in 2D.** Bowl above has curvature 2 in θ₀ and 4 in θ₁. The steep direction sets the limit: δ < 2/4 = 0.5. I ran it: `lr=0.4` converges, `lr=0.5` bounces forever on θ₁, `lr=0.6` explodes (loss ~1e29).

**Practical rules for the lab:**
- Loss is `nan`/`inf`/exploding → **step too big**. Divide `lr` by 10.
- Loss falls but not enough in your allowed steps → step too small, or too few steps. Multiply `lr` by 3, or add steps.
- If the test says "within T steps", you can't just add steps. Tune `lr`.

**Check Q4.** With `L = θ²` and δ = 1.0, what does θ do?

---

## Block 5: Writing a gradient by hand (30 min, most important)

**Plain version.** The loss is "average of squared mistakes". To nudge a parameter, ask: "if I change this parameter a bit, how much does the *average squared mistake* change?" Squaring a mistake makes it change twice as fast as the mistake itself (the "2"), and the parameter only affects the mistake through its own multiplier (the "× x" for a slope, "× 1" for an offset).

**Maths: the recipe (chain rule).** For `L = mean( (pred − y)² )`:

```
error  e_i = pred_i − y_i
∂L/∂θⱼ = mean( 2 · e_i · ∂pred_i/∂θⱼ )
```

So you need two things for each parameter: the error vector, and "how does the prediction move when this parameter moves" (∂pred/∂θⱼ).

**Worked example (a parallel model, not the line).** Model `pred = c + a·x²`, with `theta = [c, a]`.
- ∂pred/∂c = 1 → `∂L/∂c = mean(2·e)`
- ∂pred/∂a = x² → `∂L/∂a = mean(2·e·x²)`

```python
def quad_model(x, theta):
    return theta[0] + theta[1] * x**2

def quad_grad(theta, x, y):
    err = quad_model(x, theta) - y          # prediction minus truth
    return np.array([2 * np.mean(err),
                     2 * np.mean(err * x**2)])
```

I ran this on made-up data and compared with finite differences. Both gave `[-4.9236, -9.7559]`. Then I ran gradient descent from `[0, 0]` with `lr=0.05` for 2000 steps. It recovered `[1.487, 0.710]` against a true `[1.5, 0.7]`.

**The pattern to memorise**
1. `err = model(x, theta) - y` (vector of length N)
2. One entry per parameter: `2 * np.mean(err * (∂pred/∂that parameter))`
3. Return `np.array([...])` **in the same order as theta**.

**Order matters.** If your model is `theta[1]*x + theta[0]`, then `grad[0]` is the offset's derivative and `grad[1]` is the slope's. Mixing them gives a gradient that points somewhere wrong (it won't crash, it'll just not converge).

**Sum versus mean.** With `np.sum` instead of `np.mean` the gradient is exactly N times bigger (I measured a ratio of 60 on Lab 3's 60 points). Then your safe step size is N times smaller. Lab 3's `mse_loss` uses `np.mean`, so use `np.mean` in the gradient too.

**Your job (Drill 2):** apply the same recipe to the line model, then check it with Block 7's tool. If your numbers agree to ~6 decimals, it's right.

**Check Q5.** Model `pred = w·x` (no offset), `L = mean((w·x − y)²)`. What is `∂L/∂w`?

---

## Block 6: Line fit end to end on Lab 3's data (25 min)

**Plain version.** Same story as Lab 3, but instead of guessing lines at random, you slide downhill on the loss surface.

**The setup (Lab 3's conventions):** `linear_model(x, theta) = theta[1]*x + theta[0]`; `mse_loss(y_true, y_pred) = np.mean((y_true - y_pred)**2)`; 60 data points with x between 0 and 10.

**The ground truth** (what the best straight line is, from `np.polyfit`): offset **7.4774**, slope **−1.1676**, MSE **4.1024**. Your gradient descent should end here.

**What I measured** (start `[0, 0]`, correct gradient):

| setting | result |
|---|---|
| `lr=0.02`, 500 steps | `[7.390, -1.155]`, loss 4.1041 |
| `lr=0.02`, 1000 steps | `[7.476, -1.1675]`, loss 4.1024 |
| `lr=0.02`, 3000 steps | `[7.4774, -1.16765]`, loss 4.1024 |
| `lr=0.001`, 5000 steps | `[6.66, -1.05]`, loss 4.25 (too small, not there yet) |
| `lr=0.01` or `0.02` | converges |
| `lr ≥ 0.03` | **diverges** (the loss goes to `inf`) |

The largest safe step is about **0.028** for this data. The reason is the Hessian: its largest eigenvalue is 71.4, and the safe limit is `2 / 71.4`. The smallest eigenvalue is 0.44, which is why the slow direction needs thousands of steps. This is the lecture's "downhill is not always the shortest route": one direction is steep, one is shallow, and a single `lr` is a compromise.

**Polynomial trap (Lab 3 had a polynomial task).** For a degree-3 polynomial on the raw x (0 to 10), x³ reaches 1000, so the gradient is huge in the x³ direction. The safe step is only about **7e-6** (I measured: `1e-5` diverges, `5e-6` doesn't). Even after 50,000 steps at `5e-6` the loss is 9.2 against the true best 0.54. Practical advice:
- If a polynomial gradient descent blows up, drop `lr` to `1e-6`-ish.
- Dividing x by 10 first (`x/10`) lets you use `lr=0.1` and gets to loss 1.15 in 50k steps, but only do that if the task lets you change inputs.
- If the lab gives you `guess` and `poly_neighbour` for polynomials, hill climbing may just be the intended tool. Read the task text.

**Generalised polynomial gradient** (works for the line too):

```python
def poly_grad(theta, x, y):
    err = poly_model(x, theta) - y
    return np.array([2 * np.mean(err * x**i) for i in range(len(theta))])
```

(`∂pred/∂θᵢ = xⁱ`.) I checked it against finite differences on Lab 3's data: all four entries match.

**Check Q6.** Your loss prints `nan` after 8 steps. First thing to change?

---

## Block 7: Finite differences (15 min): your "is my gradient right?" tool

**Plain version.** To know the slope, nudge the parameter a tiny bit each way and see how the loss changed. No calculus, just evaluate the loss twice.

**Maths (central difference):**

```
∂L/∂θᵢ ≈ ( L(θ + h·eᵢ) − L(θ − h·eᵢ) ) / (2h)       eᵢ = unit vector with a 1 in slot i
```

**Code.**

```python
def numerical_gradient(f, theta, h=1e-5):
    theta = np.array(theta, dtype=float)
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        e = np.zeros(len(theta)); e[i] = h
        g[i] = (f(theta + e) - f(theta - e)) / (2*h)
    return g
```

Use it to check any hand-derived gradient: `np.allclose(your_grad(theta), numerical_gradient(loss, theta), atol=1e-4)`.

**Why we don't use it for training (lecture points):**
1. **Cost.** Two loss evaluations per parameter, per step. A million parameters = 2 million loss evaluations for one gradient.
2. **Floating-point problems.** Adds tiny `h` to a big `x`, subtracts two almost-equal numbers, divides by tiny `2h`. I measured the error for `sin` at x = 1: `h=0.1` error 9e-4; `h=1e-5` error 1e-11 (good); `h=1e-10` error 6e-8; `h=1e-14` error 4e-3 (rounding noise wins). Pick `h` around 1e-5 to 1e-6.

**Check Q7.** A model has 1,000 parameters. How many loss evaluations for one central-difference gradient?

---

## Block 8: Autograd, Jacobian, Hessian: vocabulary only (15 min)

**Plain version.** Autograd is a program that works out the gradient for you by tracking every operation (the "computational graph") and applying the chain rule backwards. Deep-learning libraries (PyTorch, JAX, TensorFlow) do this. In the lecture it's `grad(loss)`. You can't use it in the lab (not installed, no extra imports), so your Block 5 skill replaces it.

**Shapes (likely short-answer questions):**

| Object | For | Shape |
|---|---|---|
| derivative | scalar → scalar | single number |
| **gradient** `∇L` | `L: ℝⁿ → ℝ` (our loss) | length-`n` vector |
| **Jacobian** | `f: ℝⁿ → ℝᵐ` | `m × n` matrix |
| **Hessian** `∇²L` | `L: ℝⁿ → ℝ` | `n × n` matrix |

The gradient is just the Jacobian of a scalar function, with a single row. The Hessian is the Jacobian of the gradient.

**Check Q8.** `f: ℝ³ → ℝ²`. Shape of its Jacobian?

---

## Block 9: Momentum and random restart (20 min)

**Plain version.** *Momentum:* a ball with weight keeps rolling in the direction it was going, so tiny bumps don't stop it and it moves faster along long valleys. *Random restart:* if you get stuck in a small dip, drop a new ball somewhere else and keep the best result.

**Maths (momentum).**

```
v     = α·v + δ·∇L(θ)
θ_new = θ − v
```

`α` between 0 and 1. `α = 0` is plain gradient descent. Closer to 1 = more memory.

**Code.**

```python
def momentum_descent(loss, grad, theta0, lr, alpha, steps):
    theta = np.array(theta0, dtype=float)
    v = np.zeros_like(theta)                       # velocity starts at rest
    history = [(theta.copy(), loss(theta))]
    for t in range(steps):
        v = alpha * v + lr * grad(theta)
        theta = theta - v
        history.append((theta.copy(), loss(theta)))
    return theta, loss(theta), history

def gd_restarts(loss, grad, sample, lr, steps, restarts):
    best_theta, best_loss = None, np.inf
    for r in range(restarts):
        theta, l, _ = gradient_descent(loss, grad, sample(), lr, steps)
        if l < best_loss:
            best_theta, best_loss = theta, l
    return best_theta, best_loss
```

(`sample` is a function with no arguments that returns a random starting point, like Lab 3's `guess`.)

**What I measured on Lab 3's line:** with `lr=0.01`, plain GD still hasn't reached loss 4.11 after 500 steps (it's at 4.252). With `alpha=0.9` it passes 4.11 at **step 50**, and after 300 steps it's at `[7.4774, -1.16764]`, loss 4.1024. Momentum is not magic; it lets you move faster along the shallow direction.

**Restart demo:** `f(θ) = θ⁴ − 8θ² + 2θ` has two dips. Starting at 3 gradient descent ends in the shallow one (θ=1.93, f=−12.07); starting at −3 or 0 ends in the deep one (θ=−2.06, f=−20.06). Restarts from random starting points keep the best of these.

**Lecture quirk:** the lecture starts `vel = dL(theta)`, mine starts at zero. Both are fine.

**Check Q9.** What does `alpha = 0` give you?

---

## Block 10: Stochastic relaxation and SGD (20 min)

**Plain version.** *SGD:* to know which way is downhill for a huge dataset, you don't need to look at every point. Look at a random handful, step based on that. The direction is a bit noisy but much cheaper, and one pass through lots of data is plenty. *Stochastic relaxation:* a cliff (like "eaten or not eaten") has no usable slope. Average over many slightly-different versions of the cliff and you get a smooth ramp that has one.

**Maths.** If `L(θ) = Σᵢ Lᵢ(θ)` (a sum over data points), then `∇L = Σᵢ ∇Lᵢ`. So the gradient on a random subset (a **minibatch**) is a cheap estimate of the full gradient. One full pass over the data = one **epoch**. The noise also helps jump out of shallow minima.

**Code.** The gradient function now takes the data subset:

```python
def sgd(grad, theta0, x, y, lr, batch_size, epochs):
    theta = np.array(theta0, dtype=float)
    n = len(x)
    for epoch in range(epochs):
        order = np.random.permutation(n)                     # shuffle once per epoch
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start:start + batch_size]            # one minibatch
            theta = theta - lr * grad(theta, x[idx], y[idx])
    return theta
```

It uses the same `line_grad(theta, x, y)` as Drill 2, now called on a slice. Because the gradient uses `np.mean`, it is already the average over the batch (the lecture divides a *sum* gradient by `batch_size` for the same effect).

- Updates per epoch = `n // batch_size`. For the lecture's 100,000 points with `batch_size=100` that's **1000 updates per epoch**.
- Lecture result: **one epoch** over 100,000 noisy points recovered the line. I reran it: it ended at `[1.990, 1.005]` against a true `[2, 1]` (offset, slope).
- On Lab 3's 60 points, `lr=0.01, batch_size=10, 300 epochs` ended at `[7.447, -1.249]`: close, but noisy, not exactly the optimum. Don't expect SGD to hit exactly 4.1024.
- Smaller batch means noisier steps. `batch_size=1` is the noisiest.

**Check Q10.** 100,000 points, `batch_size=100`: how many parameter updates in one epoch?

---

## Block 11: Continuity, Lipschitz, second order: theory only (15 min)

**Continuity ladder (smoother = easier).** discontinuous < `C⁰` (no jumps) < `C¹` (slope has no jumps) < `C²`. First-order methods need at least `C¹` and a defined gradient.

**Lipschitz continuity.** The function has a maximum steepness `K`: `|f(x) − f(y)| ≤ K·|x − y|`. Small `K` = gentle, big `K` = steep. The safe step size shrinks as `K` grows. (Picture: a cone slid along the curve that never touches it.)

**Critical points** (gradient = 0), classified by the **Hessian's eigenvalues**:

| eigenvalues | point |
|---|---|
| all positive (positive definite) | minimum |
| all negative | maximum |
| mixed signs | saddle |
| some zero | plateau / ridge |

**Second-order methods** use the Hessian to jump straight to the bottom of a local bowl. They're much faster and escape saddles, but need an `n × n` matrix. For `n = 1,000,000` that's `10¹² × 8 bytes = 8 terabytes`. That is the "curse of dimensionality yet again". L-BFGS approximates it with less memory.

**Other limits to remember (one line each):**
- Gradient descent finds *a* local minimum, not necessarily *the* global one (random restart, momentum, SGD noise help).
- Needs a differentiable objective (stochastic relaxation relaxes that).
- Gradient descent's step is steepest *locally*, not the shortest route.
- Only guaranteed to find the global minimum if the function is convex and the step is right.

**Check Q11.** All eigenvalues of the Hessian are positive. Minimum, maximum, or saddle?

---

## Part 5: Quirks in the lecture notebook (so they don't trip you)

I found these by reading and running the lecture code. None are things you must fix; just be aware.

1. **Typo in the analytic example.** The markdown says `f'(x) = 4x³ − 20x − 100` and `f'' = 12x² − 20`. The code, and the correct maths for `x⁴ − 40x² − 100x`, are `4x³ − 80x − 100` and `12x² − 80`. At x = 5: `f' = 0`, `f'' = 220 > 0`, so a minimum.
2. **Docstring versus code.** The lecture `gradient_descent` docstring says it stops "when change in loss is less than tol". The code actually stops when `sum(|gradient|)` is below `tol`.
3. **No iteration limit in the lecture version.** With δ = 1.0 on `θ²` it loops forever. Your lab version has `steps`.
4. **`theta_0` as ints crashes** the lecture's in-place `theta += ...` (see Block 3).
5. **Parameter order differs between lecture and lab.** Lecture line fit: `theta = [m, c]` (slope first). Lab 3: `theta = [intercept, slope]`.
6. **The lecture SGD** tracks the *full-data* loss after every minibatch step. That's only for plotting; it defeats the speed benefit.
7. **`autograd` / `scipy` Newton-CG cells** need `autograd`, which isn't installed here. I didn't run those cells; I tested the numpy equivalents instead.

---

## Part 6: Drills (60 min). Close this file, type from memory

Use `x`, `y` from Lab 3's data cell, plus `linear_model`, `mse_loss` from Lab 3. Solutions in Part 8, but try first.

1. **Bowl.** Without looking: write `loss` and `grad` for `L(θ) = (θ₀−3)² + 2(θ₁+1)²`. Run 100 steps from `[0, 0]` with `lr = 0.1`. Expect `[3, -1]`.
2. **Line gradient.** Write `line_grad(theta, x, y)` for Lab 3's `linear_model` (`theta = [intercept, slope]`) with the mean-squared-error loss. **Check** against `numerical_gradient` at `[7.0, -1.0]`. The numbers should agree to about 6 decimals.
3. **Gradient descent function.** Write `gradient_descent(loss, grad, theta0, lr, steps)` returning `(theta, best_loss, history)` where `history` is a list of `(theta, loss)`. Test that loss never goes up with `lr=0.02`.
4. **Fit the line.** Use 2 and 3 on Lab 3's data from `[0, 0]`. Reach loss < 4.11 and compare with the true `[7.4774, -1.1676]`. How many steps do you need at `lr=0.02`?
5. **Find the cliff.** Try `lr = 0.01, 0.02, 0.03, 0.05`. Which diverge? Predict the borderline before you run it.
6. **Momentum.** Add `alpha`. At `lr=0.01` how many steps until loss < 4.11, with `alpha=0` vs `0.9`?
7. **Numerical gradient.** Write `numerical_gradient(f, theta, h)` without looking.
8. **Minibatch.** Write `sgd(grad, theta0, x, y, lr, batch_size, epochs)`. Fit the 60 points with `batch_size=10`.
9. **Polynomial.** Write `poly_grad(theta, x, y)` for the degree-3 model. Check with Drill 7's tool. Then find an `lr` that doesn't blow up.
10. **Restarts.** Write `gd_restarts` and use it on `f(θ) = θ⁴ − 8θ² + 2θ`. Which of the two dips does each start fall into?

---

## Part 7: Exam-day checklist

**Before you code**
- [ ] Run the imports cell. Set `student_id` (8 characters) and the signature. Run the Task 0 tests.
- [ ] For each task, read the provided helper cells (model, loss, plotting) **before** writing: note the argument order and the order of `theta`.
- [ ] Don't edit cells without `# YOUR CODE HERE`. No `import`.

**While you code**
- [ ] Gradient: same length and order as `theta`; uses `np.mean`; error is `prediction − y`.
- [ ] Update: `theta = theta - lr * grad(theta)` (minus!).
- [ ] `theta = np.array(theta0, dtype=float)`; save `theta.copy()` to history.
- [ ] Return exactly what's asked: `(theta, loss, history)`.
- [ ] Never `np.random.seed(...)` inside your function.
- [ ] Verify the gradient with `numerical_gradient` before you debug anything else.

**When it goes wrong**

| Symptom | Likely cause |
|---|---|
| Loss goes up or to `nan`/`inf` | Step too big (divide by 10), or plus sign instead of minus |
| Loss goes down very slowly | Step too small; try 3× bigger |
| Converges but to the wrong line | Gradient entries in the wrong order, or wrong parameter order |
| `UFuncTypeError` | Integer starting `theta`; use `dtype=float` |
| History all identical | Missing `.copy()` with an in-place update |
| `ValueError: shapes` | Gradient returned a list of arrays; use `np.array([...])` of scalars |
| Hidden-test worry | Does it work for a different `theta0`, `lr`, and number of steps? |

**Last 10 minutes:** Kernel → Restart and Run All, no red cells, read the marks summary, submit to Moodle **before time is up**.

---

## Part 8: Answer key

**Check questions**
1. **Zeroth-order.** It only evaluates `L(θ)` at random nearby points; it never uses a derivative.
2. Gradient `2θ = 4`. It's positive (uphill is to the right), so you move **left**.
3. Gradient at −10 is `−20`. `θ = −10 − 0.25·(−20) = −10 + 5 = −5`.
4. `θ → −θ` each step: it bounces between −10 and +10 forever.
5. `∂L/∂w = 2 · mean( (w·x − y) · x )`.
6. Divide `lr` by 10 (step too big).
7. **2,000** (two per parameter).
8. **2 × 3** (outputs × inputs).
9. Plain gradient descent.
10. `100,000 / 100 =` **1000**.
11. **Minimum.**

**Drills**

```python
# 1. bowl
loss = lambda t: (t[0]-3)**2 + 2*(t[1]+1)**2
grad = lambda t: np.array([2*(t[0]-3), 4*(t[1]+1)])

# 2. line gradient (theta = [intercept, slope])
def line_grad(theta, x, y):
    err = (theta[1]*x + theta[0]) - y
    return np.array([2*np.mean(err),         # d/d intercept: pred changes by 1
                     2*np.mean(err * x)])    # d/d slope:     pred changes by x

# 3. gradient descent: see Block 3

# 4/5. L = lambda t: mse_loss(y, linear_model(x, t)); G = lambda t: line_grad(t, x, y)
#      gradient_descent(L, G, [0, 0], 0.02, 1000)
#      -> about 1000 steps get to loss 4.1024; lr=0.02 converges (factor safe), 0.03+ diverges.
#      Borderline is 2/71.4 = 0.028.

# 6. momentum: see Block 9. alpha=0: still 4.25 after 500 steps; alpha=0.9: under 4.11 at step 50.

# 7. numerical_gradient: see Block 7.
# 8. sgd: see Block 10.
# 9. poly_grad: see Block 6. lr around 5e-6 or smaller for raw x in [0,10].
# 10. gd_restarts: see Block 9. Start 3 -> shallow dip (f=-12.07); starts -3 or 0 -> deep dip (f=-20.06).
```

**Reference numbers to sanity-check your own output (Lab 3 data, mean-squared-error loss)**
- Best line: offset 7.4774, slope −1.1676, loss 4.1024.
- Best cubic: loss 0.543.
- Gradient at `[7, -1]`: `[0.788, 6.748]`.
- Safe step for the line: below about 0.028. For the raw cubic: below about 7e-6.
