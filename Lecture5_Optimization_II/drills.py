import numpy as np
def loss(theta):
    return (theta[0] - 3) ** 2 + 2 * (theta[1] + 1) ** 2

def grad(theta):
    return np.array([2 * (theta[0] - 3), 4 * (theta[1] + 1)])

def line_grad(theta, x, y):
    err = (theta[1] * x + theta[0]) - y
    return np.array([2 * np.mean(err), 2 * np.mean(err * x)])

def numerical_grad(f, theta, h=1e-6):
    theta = np.array(theta, dtype=float)
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        e = np.zeros(len(theta))
        e[i] = h
        g[i] = (f(theta + e) - f(theta - e)) / (2 * h)
    return g

def gradient_descent(loss, grad, theta0, lr, alpha, steps):
    theta = np.array(theta0, dtype=float)
    v = np.zeros_like(theta)
    history = [(theta.copy(), loss(theta))]
    for _ in range(steps):
        v = alpha * v + lr * grad(theta)
        theta = theta - v
        history.append((theta.copy(), loss(theta)))
    return theta, loss(theta), history

def poly_model(x, theta): 
    y_pred = np.zeros_like(x, dtype=float)

    for i in range(len(theta)):
        y_pred += theta[i] * x**i

    return y_pred

def poly_grad(theta, x, y):
    err = poly_model(x, theta) - y
    return np.array([2 * np.mean(err * x ** i) for i in range(len(theta))])

def sgd(grad, theta0, x, y, lr, batch_size=10, epochs=100):
    theta = np.array(theta0, dtype=float)
    n = len(x)
    for epoch in range(epochs):
        order = np.random.permutation(n)
        for start in range(0, n-batch_size + 1, batch_size):
            idx = order[start:start + batch_size]
            theta = theta - lr * grad(theta, x[idx], y[idx])

    return theta



# ---------- CHECKS ----------
np.random.seed(0)
x = np.linspace(0, 10, 60)
y = -1.2 * x + 7.5 + np.random.normal(0, 1, 60)      # made-up line + noise
L = lambda t: np.mean((y - (t[1] * x + t[0])) ** 2)

t = np.array([7.0, -1.0])
print("line_grad     :", line_grad(t, x, y))
print("numerical_grad:", numerical_grad(L, t))
print("match?", np.allclose(line_grad(t, x, y), numerical_grad(L, t), atol=1e-4))

print("bowl grad at [0,0]:", grad(np.array([0.0, 0.0])), "expected [-6, 4]")
assert np.allclose(grad(np.array([0.0, 0.0])), numerical_grad(loss, [0.0, 0.0]), atol=1e-4)
print("all checks passed")

theta, final_loss, h = gradient_descent(loss, grad, [0, 0], 0.1, 100)
print("theta:", theta, "loss:", final_loss)
assert np.allclose(theta, [3, -1], atol=1e-3)
assert len(h) == 101
assert all(h[i][1] >= h[i+1][1] for i in range(len(h) - 1))
print("gradient_descent OK")



