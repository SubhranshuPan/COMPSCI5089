# COMPSCI5089 / IDSS: Lab 5 Probability Assessment Cheat Sheet
**Focus Topics:** Conditional Probabilities • Entropy • Expectation • Bayes' Rule  
**Target Duration:** 45–60 minute high-yield review before entering the lab.

---

## 0. Lab Survival & Execution Protocol (How Labs Work)

Based on Labs 0–4 in `Lab_solutions/`:
1. **Lab Machine Exclusive**: Must be executed and submitted from lab machines during the timetabled lab.
2. **Task 0 Setup**: Always fill in `student_id` (8 digits) and `student_typewritten_signature` first.
3. **Autograder Mechanism (`utils.tick`)**:
   - Each task has visible tests (`with tick.marks(n): assert ...`) that give instant feedback.
   - Passing visible tests is necessary but not sufficient; hidden tests evaluate edge cases (e.g., zeroes, uniform inputs, empty arrays, extreme probabilities).
4. **Vectorization & Data Types**:
   - Use NumPy arrays (`np.ndarray`) and avoid raw Python loops wherever possible.
   - Ensure functions return the exact requested type (e.g., `float` vs `np.float64` vs 1D array).
5. **Golden Pre-Submission Habit**:
   - In Jupyter: **Kernel $\rightarrow$ Restart & Run All**.
   - Verify every cell runs with code `0` and no assertion errors or `NaN`s appear.

---

## 1. Quick Reference: The 4 Core Lab 5 Pillars

| Concept | Mathematical Formula | Key Intuition | Primary NumPy Implementation |
| :--- | :--- | :--- | :--- |
| **Expectation $\mathbb{E}[X]$** | $\sum_x x \, P(X=x)$ | Probability-weighted average outcome | `np.sum(values * pmf)` |
| **Expectation $\mathbb{E}[g(X)]$** | $\sum_x g(x) \, P(X=x)$ | Average score under non-linear transform $g$ | `np.sum(g(values) * pmf)` |
| **Variance $\text{Var}(X)$** | $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | Spread / 2nd central moment | `np.sum(pmf * values**2) - (np.sum(pmf * values))**2` |
| **Marginal $P(X)$** | $\sum_y P(X, Y)$ | Summing out unwanted variable $Y$ | `np.sum(joint, axis=1)` (row marginal) |
| **Conditional $P(Y \mid X)$** | $\frac{P(X, Y)}{P(X)}$ | Probability of $Y$ given known condition $X$ | `joint / P_X[:, None]` (row-normalized) |
| **Bayes' Rule** | $\frac{P(D \mid H) P(H)}{\sum_k P(D \mid H_k) P(H_k)}$ | Inverting conditional probability using evidence | `(p_D_H * p_H) / np.sum(p_D_H_all * p_H_all)` |
| **Shannon Entropy $H(X)$** | $-\sum_x P(x) \log_2 P(x)$ | Expected surprise / uncertainty (in bits) | `-np.sum(p[p > 0] * np.log2(p[p > 0]))` |
| **Binary Entropy $H_b(q)$** | $-q \log_2 q - (1-q) \log_2(1-q)$ | Uncertainty of coin toss (max at $q=0.5$) | `binary_entropy(q)` |

---

## 2. Topic 1: Expectation & Moments

### 2.1 Theoretical Foundations
- **Definition (Discrete)**: 
  $$\mathbb{E}[X] = \sum_x x \, f_X(x) = \sum_x x \, P(X=x)$$
- **Function of a Random Variable (LOTUS)**:
  $$\mathbb{E}[g(X)] = \sum_x g(x) \, f_X(x)$$
- **CRITICAL WARNING**:
  $$\mathbb{E}[g(X)] \neq g(\mathbb{E}[X]) \quad \text{(unless } g \text{ is linear/affine)}$$
  *Example*: $\mathbb{E}[X^2] \neq (\mathbb{E}[X])^2$.
- **Linearity of Expectation (Holds unconditionally — even if dependent!)**:
  $$\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$$
- **Variance**:
  $$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
  $$\text{Var}(aX + b) = a^2 \text{Var}(X)$$
  *If $X, Y$ independent*: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 2.2 Reusable NumPy Patterns
```python
import numpy as np

# Given outcomes `values` and probabilities `pmf` (must sum to 1.0)
def compute_expectation(values: np.ndarray, pmf: np.ndarray) -> float:
    assert np.isclose(np.sum(pmf), 1.0), "PMF must sum to 1"
    return float(np.sum(values * pmf))

def compute_variance(values: np.ndarray, pmf: np.ndarray) -> float:
    e_x = compute_expectation(values, pmf)
    e_x2 = compute_expectation(values**2, pmf)
    return float(e_x2 - e_x**2)

def compute_expectation_func(values: np.ndarray, pmf: np.ndarray, func) -> float:
    # E[g(X)] = sum( g(x) * P(X=x) )
    return float(np.sum(func(values) * pmf))
```

---

## 3. Topic 2: Joint, Marginal & Conditional Probabilities

### 3.1 Mental Model: 2D Table of Outcomes
Consider two discrete random variables: row variable $R$ (size $M$) and column variable $C$ (size $N$).
The joint probability matrix is $J \in \mathbb{R}^{M \times N}$, where $J[r, c] = P(R=r, C=c)$.
- **Unitarity**: $\sum_{r=0}^{M-1} \sum_{c=0}^{N-1} J[r, c] = 1$.

```
              C = 0       C = 1       ...     C = N-1     Row Marginal P(R)
R = 0     |  J[0,0]      J[0,1]      ...     J[0,N-1]  |  --> sum across columns (axis=1)
R = 1     |  J[1,0]      J[1,1]      ...     J[1,N-1]  |  --> sum across columns (axis=1)
...       |   ...         ...        ...       ...     |
Col Marg. |  --> sum across rows (axis=0)              |  Total Sum = 1.0
 P(C)
```

### 3.2 Marginals (Sum Rule)
- Marginalizing means summing over unwanted variables to remove them:
  $$P(R=r) = \sum_{c} P(R=r, C=c) \implies \texttt{P\_R = np.sum(J, axis=1)}$$
  $$P(C=c) = \sum_{r} P(R=r, C=c) \implies \texttt{P\_C = np.sum(J, axis=0)}$$

### 3.3 Conditionals (Product Rule & Division)
- By definition:
  $$P(C=c \mid R=r) = \frac{P(R=r, C=c)}{P(R=r)}$$
  Each row sums to 1.
- In NumPy:
  ```python
  # P(C | R): divide each row r by scalar P(R=r)
  # J is (M, N), P_R is (M,). Use broadcasting with P_R[:, np.newaxis]
  cond_C_given_R = J / P_R[:, np.newaxis]
  
  # Or via transpose (as in Lecture 7 notebook):
  # cond_C_given_R = (J.T / P_R).T
  
  # P(R | C): divide each column c by scalar P(C=c)
  # P_C is (N,). Broadcast with P_C[np.newaxis, :]
  cond_R_given_C = J / P_C[np.newaxis, :]
  ```

### 3.4 Independence Test
$R$ and $C$ are statistically independent **if and only if**:
$$P(R=r, C=c) = P(R=r) \cdot P(C=c) \quad \forall r, c$$
In NumPy: `np.allclose(J, np.outer(P_R, P_C))`

### 3.5 Bigram / Text Transition Matrix Pattern (From Lecture 7)
```python
def build_bigram_matrices(sequence, vocab_size):
    # Count transitions: c1 -> c2 (c1 at t-1, c2 at t)
    joint = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for prev_c, curr_c in zip(sequence[:-1], sequence[1:]):
        joint[prev_c, curr_c] += 1.0
        
    # 1. Normalize joint
    joint = joint / np.sum(joint)
    
    # 2. Marginals
    p_prev = np.sum(joint, axis=1) # P(c_{t-1})
    p_curr = np.sum(joint, axis=0) # P(c_t)
    
    # 3. Conditional: P(curr | prev)
    # Avoid div by zero if a character never appeared
    cond_curr_given_prev = np.zeros_like(joint)
    non_zero = p_prev > 0
    cond_curr_given_prev[non_zero] = joint[non_zero] / p_prev[non_zero, None]
    
    return joint, p_prev, p_curr, cond_curr_given_prev
```

---

## 4. Topic 3: Bayes' Rule & Evidential Reasoning

### 4.1 The Formula & Terminology
$$P(H \mid D) = \frac{P(D \mid H) \, P(H)}{P(D)}$$
- **$P(H)$ (Prior)**: Initial degree of belief in hypothesis $H$ before evidence.
- **$P(D \mid H)$ (Likelihood)**: Probability that data $D$ occurs if $H$ is true.
- **$P(D)$ (Evidence / Marginal Likelihood)**: Overall probability of seeing data $D$ across all mutually exclusive hypotheses $H_k$:
  $$P(D) = \sum_{k} P(D \mid H_k) \, P(H_k)$$
- **$P(H \mid D)$ (Posterior)**: Updated belief in $H$ after observing $D$.

### 4.2 The Canonical Disease / Diagnostic Test Problem
Let $H=1$ denote having the disease, $H=0$ healthy. Let $T=1$ denote a positive test.
- **Prevalence (Prior)**: $P(H=1)$
- **Sensitivity (True Positive Rate)**: $P(T=1 \mid H=1)$
- **Specificity (True Negative Rate)**: $P(T=0 \mid H=0)$
- **False Positive Rate**: $P(T=1 \mid H=0) = 1 - \text{Specificity}$
- **False Negative Rate**: $P(T=0 \mid H=1) = 1 - \text{Sensitivity}$

**The Question**: What is $P(H=1 \mid T=1)$?
$$P(H=1 \mid T=1) = \frac{P(T=1 \mid H=1) \, P(H=1)}{P(T=1 \mid H=1) P(H=1) + P(T=1 \mid H=0) P(H=0)}$$

#### Lab Implementation Template:
```python
def bayes_disease_test(prior_disease: float, sensitivity: float, specificity: float) -> float:
    p_d1 = prior_disease
    p_d0 = 1.0 - prior_disease
    
    p_pos_given_d1 = sensitivity
    p_pos_given_d0 = 1.0 - specificity  # false positive rate
    
    # Law of total probability for evidence P(Test = +)
    p_test_pos = (p_pos_given_d1 * p_d1) + (p_pos_given_d0 * p_d0)
    
    # Posterior
    p_d1_given_pos = (p_pos_given_d1 * p_d1) / p_test_pos
    return float(p_d1_given_pos)
```

### 4.3 The Natural Frequency Technique (Sanity Check Tool!)
If intuition feels counter-intuitive due to base-rate neglect, convert to counts out of $N = 100,000$:
- Disease prevalence = $1/10,000 \implies 10$ have disease, $99,990$ healthy.
- Test accuracy = $95\%$ (sensitivity $0.95$, false positive rate $0.05$):
  - True Positives = $10 \times 0.95 = 9.5$
  - False Positives = $99,990 \times 0.05 = 4,999.5$
- Total Positives = $9.5 + 4,999.5 = 5,009$
- $P(\text{Disease} \mid +) = \frac{9.5}{5009} \approx 0.0019$ (less than $0.2\%$, despite $95\%$ test accuracy!).

### 4.4 Sequential Bayes Updating
If multiple independent pieces of evidence $D_1, D_2, \dots$ are observed:
$$\text{Prior}_1 = P(H)$$
$$\text{Posterior}_1 = P(H \mid D_1) \propto P(D_1 \mid H) P(H)$$
$$\text{Posterior}_2 = P(H \mid D_1, D_2) \propto P(D_2 \mid H) \cdot \text{Posterior}_1$$
*The posterior of step $t$ becomes the prior of step $t+1$!*

---

## 5. Topic 4: Entropy & Information Theory

### 5.1 Definition & Intuition
$$H(X) = \mathbb{E}[-\log_2 P(X)] = -\sum_{x} P(X=x) \log_2 P(X=x)$$
- Measured in **bits** (log base 2) or **nats** (natural log $\ln$). Lab uses $\log_2$.
- **Intuition**: 
  - Degree of surprise / average number of yes/no questions needed to determine outcome.
  - Narrow / peaked distribution $\implies$ low surprise $\implies$ low entropy.
  - Flat / uniform distribution $\implies$ high surprise $\implies$ high entropy.

### 5.2 Key Properties & Bounds
1. **Non-negativity**: $H(X) \ge 0$.
2. **Deterministic Minimum**: $H(X) = 0$ if and only if $P(X=x_0) = 1$ for some single state $x_0$.
3. **Maximum Entropy**: For discrete distribution with $K$ possible states:
   $$\max H(X) = \log_2(K) \quad \text{achieved if and only if } P(x) = \frac{1}{K} \text{ (Uniform)}$$
   *Example*: Fair 6-sided die: $H = \log_2(6) \approx 2.585$ bits.

### 5.3 Binary Entropy Function $H_b(q)$
For a Bernoulli coin toss with $P(\text{heads}) = q$:
$$H_b(q) = -q \log_2 q - (1-q) \log_2(1-q)$$
- Symmetric around $q=0.5$.
- Maximum at $q=0.5$ ($H_b(0.5) = 1.0$ bit).
- Minimum at $q=0$ or $q=1$ ($H_b = 0$ bits).

### 5.4 Conditional Entropy
Uncertainty of $Y$ remaining after observing $X$:
$$H(Y \mid X) = \sum_x P(X=x) \, H(Y \mid X=x) = -\sum_x \sum_y P(x, y) \log_2 P(y \mid x)$$
- Bigram text example: If preceding letter is `'q'`, following letter `'u'` has near $P=1 \implies H(C_t \mid C_{t-1}=\text{'q'}) \approx 0$. If preceding letter is `' '` (space), following letter can be anything $\implies$ high entropy.

### 5.5 Handling Zero Probabilities in NumPy (Avoid `NaN`!)
Mathematically: $\lim_{p \to 0} p \log_2 p = 0$.  
In NumPy: `0.0 * np.log2(0.0)` produces `NaN` and throws a `RuntimeWarning`.

**Two Safe Implementations**:
```python
# Method A: Boolean masking non-zero values (Exact & Cleanest!)
def entropy_masked(p: np.ndarray) -> float:
    p_nonzero = p[p > 0.0]
    return float(-np.sum(p_nonzero * np.log2(p_nonzero)))

# Method B: Small epsilon smoothing (matches Lecture 7 notebook)
def entropy_eps(p: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.sum(p * np.log2(p + eps)))

# Per-row entropy of transition matrix (axis=1)
def transition_matrix_entropy(cond_matrix: np.ndarray) -> np.ndarray:
    # cond_matrix shape: (M, N), each row sums to 1
    # Use where to mask zeros without warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        log_terms = np.where(cond_matrix > 0, np.log2(cond_matrix), 0.0)
    return -np.sum(cond_matrix * log_terms, axis=1)
```

---

## 6. Bridge: Discrete vs Continuous (Week 8 Preview)

To understand continuous random variables (Week 8 / 3Blue1Brown video):

| Concept | Discrete RV ($X \in \{x_1, \dots, x_k\}$) | Continuous RV ($X \in \mathbb{R}$) |
| :--- | :--- | :--- |
| **Probability Function** | PMF: $f_X(x) = P(X=x)$ | PDF: $f_X(x)$ (Probability Density) |
| **Probability at exact point** | $P(X=x) \ge 0$ (can be non-zero) | $P(X=x) = 0$ strictly! |
| **Probability in interval** | $\sum_{x \in [a, b]} P(X=x)$ | $\int_a^b f_X(x) \, dx$ (Area under curve) |
| **Normalization (Unitarity)**| $\sum_x f_X(x) = 1$ | $\int_{-\infty}^\infty f_X(x) \, dx = 1$ |
| **PDF magnitude** | Max value $\le 1.0$ | **Can exceed 1.0!** (Height $\times$ width = 1) |
| **Expectation $\mathbb{E}[X]$**| $\sum_x x \, f_X(x)$ | $\int_{-\infty}^\infty x \, f_X(x) \, dx$ |
| **Expectation $\mathbb{E}[g(X)]$**| $\sum_x g(x) \, f_X(x)$ | $\int_{-\infty}^\infty g(x) \, f_X(x) \, dx$ |
| **Marginalization** | $P(X) = \sum_y P(X, Y)$ | $f_X(x) = \int f_{X,Y}(x, y) \, dy$ |
| **Bayes' Rule Denominator** | $\sum_k P(D \mid H_k) P(H_k)$ | $\int P(D \mid H) f(H) \, dH$ |

---

## 7. Master Cheatsheet Code Snippets (Ready for Copy-Paste)

```python
import numpy as np

# 1. Empirical PMF from raw samples
def empirical_pmf(samples: np.ndarray, num_bins: int) -> np.ndarray:
    counts = np.bincount(samples, minlength=num_bins)
    return counts / np.sum(counts)

# 2. Expectation & Variance
def expectation(values: np.ndarray, probs: np.ndarray) -> float:
    return float(np.sum(values * probs))

def variance(values: np.ndarray, probs: np.ndarray) -> float:
    return float(np.sum(probs * (values**2)) - (np.sum(probs * values))**2)

# 3. 2D Joint to Marginals & Conditionals
def analyze_joint(joint: np.ndarray):
    p_x = np.sum(joint, axis=1) # row marginal (M,)
    p_y = np.sum(joint, axis=0) # col marginal (N,)
    
    # Conditional P(Y | X)
    p_y_given_x = joint / p_x[:, np.newaxis]
    # Conditional P(X | Y)
    p_x_given_y = joint / p_y[np.newaxis, :]
    
    return p_x, p_y, p_y_given_x, p_x_given_y

# 4. Bayes' Rule with arbitrary discrete hypotheses
def bayes_rule(priors: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    # priors: P(H_i), shape (K,)
    # likelihoods: P(D | H_i), shape (K,)
    unnormalized = likelihoods * priors
    evidence = np.sum(unnormalized)
    assert evidence > 0, "Evidence probability cannot be zero"
    return unnormalized / evidence

# 5. Shannon Entropy
def shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))

# 6. Binary Entropy
def binary_entropy(q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return float(-q * np.log2(q) - (1.0 - q) * np.log2(1.0 - q))

# 7. Log-Likelihood of observed sequence under PMF
def log_likelihood(observations: np.ndarray, pmf: np.ndarray) -> float:
    # observations contains outcome indices 0 .. len(pmf)-1
    probs = pmf[observations]
    assert np.all(probs > 0), "Observed impossible event under model"
    return float(np.sum(np.log(probs)))
```

---

## 8. Common Traps & Autograder Failure Modes Checklist

- [ ] **Forgot `axis` in `np.sum`**:
  - `axis=1` sums across columns $\rightarrow$ returns vector along rows ($M$).
  - `axis=0` sums across rows $\rightarrow$ returns vector along columns ($N$).
- [ ] **Broadcasting Shape Mismatch**:
  - Dividing `(M, N)` matrix by row marginal `(M,)`: must write `joint / p_x[:, None]` (or `p_x.reshape(-1, 1)`). Writing `joint / p_x` will attempt to broadcast against columns and fail or produce corrupt values!
- [ ] **`np.log2` vs `np.log`**:
  - Entropy is standardly in bits ($\log_2$). If you use `np.log` (natural base $e$), your answer will be scaled by $\ln(2) \approx 0.693$ and fail test assertions!
- [ ] **`NaN` from zero probabilities in entropy**:
  - Always filter zeros `p[p > 0]` or add small $\epsilon$ (`1e-12`).
- [ ] **Confusing Likelihood with Posterior**:
  - $P(D \mid H)$ is likelihood. $P(H \mid D)$ is posterior. Never return likelihood when asked for probability of hypothesis!
- [ ] **Base Rate Neglect in Bayes**:
  - In disease tests, always multiply by the prior $P(\text{disease})$ before normalizing by total evidence.
- [ ] **$\mathbb{E}[g(X)] \neq g(\mathbb{E}[X])$**:
  - Never plug the mean into non-linear functions (e.g., $\mathbb{E}[X^2] \neq (\mathbb{E}[X])^2$).
