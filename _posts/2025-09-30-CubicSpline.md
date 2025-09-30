---
title: "Differentiable Cubic Spline Interpolation in JAX"
date: 2025-09-30
layout: single
classes: wide
categories: 
  -tutorials
permalink: /tutorials/CubicSpline/
use_math: true
author_profile: true
toc: false
toc_label: "Table of Contents"
toc_icon: "gear"
toc_sticky: true
---

## 1. Cubic Spline Interpolation

Cubic splines are an interpolation method that construct a smooth curve by joining together cubic polynomials between data points [^1]. More precisely, the interpolant is defined as a piecewise cubic polynomial $ f :[t_1, t_{n + 1}] \to \mathbb{R}$ defined as 

$$
\begin{equation}
f(t_{\text{query}}) =
\begin{cases}
  a_1(t_{\text{query}} - t_1)^3 + b_1(t_{\text{query}} - t_1)^2 + c_1(t_{\text{query}} - t_1) + d_1, & \text{if}\ t_{\text{query}} \in [t_1, t_2] \\
  a_2(t_{\text{query}} - t_2)^3 + b_2(t_{\text{query}} - t_2)^2 + c_2(t_{\text{query}} - t_2) + d_2, & \text{if}\ t_{\text{query}} \in (t_2, t_3] \\
  \qquad \qquad \quad \vdots \\
  a_n(t_{\text{query}} - t_n)^3 + b_n(t_{\text{query}} - t_n)^2 + c_n(t_{\text{query}} - t_n) + d_n, & \text{if}\ t_{\text{query}} \in (t_n, t_{n + 1}] \\
\end{cases}
\end{equation} 
$$

where $\{ a_i, \ b_i, \ c_i, \ d_i \} _{i = 1}^{n}$ are the coefficients of $\{f_i \} _{i = 1}^{n}$ polynomial respectively, determined using $n + 1$ measurements $\{t_i, \ y_i \}_{i = 1}^{n + 1}$. In this tutorial, we will implement cubic spline interpolation in JAX, ensuring that it is fully differentiable with respect to its arguments

## 2. Optimal Parameters

To determine the $4n$ (corresponding to $𝑛$ cubic polynomials, each with four coefficients), we must solve a system of $4n$ linear equations. We begin with the most fundamental condition that the interpolant must satisfy: it should reproduce the given data points. In other words, each polynomial segment $f_i$ must pass through its two endpoints, such that, $f_i(t_i) = y_i$ and $f_i(t_{i + 1}) = y_{i + 1}$. This will give us $2n$ equations. 

$$
\begin{equation}
\begin{aligned}
    f_1(t_1) & = a_1(t_1 - t_1)^3 + b_1(t_1 - t_1)^2 + c_1(t_1 - t_1) + d_1 = y_1 \\
    f_1(t_2) & = a_1(t_2 - t_1)^3 + b_1(t_2 - t_1)^2 + c_1(t_2 - t_1) + d_1 = y_2 \\
    f_2(t_2) & = a_2(t_2 - t_2)^3 + b_2(t_2 - t_2)^2 + c_2(t_2 - t_2) + d_1 = y_2 \\
    & \qquad \qquad \qquad \vdots \\
    f_n(t_n) & = a_n(t_n - t_n)^3 + b_n(t_n - t_n)^2 + c_n(t_n - t_n) + d_n = y_n \\
    f_n(t_{n + 1}) & = a_n(t_{n + 1} - t_n)^3 + b_n(t_{n + 1} - t_n)^2 + c_n(t_{n + 1} - t_n) + d_n = y_{n + 1}
\end{aligned}
\end{equation} 
$$

To ensure smoothness, we require that both the first and second derivatives of adjacent polynomials at their point of intersection be identical. This gives us another $2 (n - 1)$ equations 

$$
\begin{equation}
\begin{aligned}
    f^{\prime}_1(t_2) & = f^{\prime}_2(t_2) \\
    f^{\prime \prime}_1(t_2) & = f^{\prime \prime}_2(t_2) \\
    & \vdots \\
    f^{\prime}_{n - 1}(t_n) & = f^{\prime}_n(t_n) \\
    f^{\prime \prime}_{n - 1}(t_n) & = f^{\prime \prime}_n(t_n)
\end{aligned}
\end{equation} 
$$

The last two equations are the boundary conditions. For not-a-knot boundary condition, the third order derivatives of the first and second polynomial should be equal at the point where they intersect. Similar condition for the last two polynomial is enforced. This give us the following equations 

$$
\begin{equation}
\begin{aligned}
    f^{\prime \prime \prime}_1(t_2) & = f^{\prime \prime \prime}_2(t_2) \\
    f^{\prime \prime \prime}_{n - 1}(t_n) & = f^{\prime \prime \prime}_n(t_n)
\end{aligned}
\end{equation} 
$$

With these $4n$ equations — consisting of $2n$ evaluation constraints, $2(n−1)$ smoothness conditions, and $2$ boundary conditions — we have exactly enough to solve for the $4n$ unknown parameters. These parameters can be obtained by solving a linear system of equations. Let us now implement this procedure in JAX.

## 3. JAX Implemention

We start by importing necessary packages

```python
import time
import functools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import flatten_util
from scipy.interpolate import CubicSpline as SCubicSpline # Benchmarking
import matplotlib.pyplot as plt # Plotting
```

We will now create a function called `CubicSplineParameters` that will take the measurements $\{ t_i, \ y_i \} _{i = 1}^{n + 1} $ as input and return the optimal parameters of the polynomials as output

```python
def CubicSplineParameters(t, y) : 
    # Gives the optimal values of parameters of cubic polynomials given time range t and function values y
    
    npoints = len(t)
    cubic_poly = lambda t, tj, p : jnp.dot(p, jnp.array([(t - tj)**3, (t - tj)**2, (t - tj), 1.])) # Define the cubic polynomial
    _f = jax.vmap(cubic_poly) # vmap the polynomial evaluation over its measurements
    _jac = jax.vmap(lambda t, tj, p : jnp.dot(p, jnp.array([3*(t - tj)**2, 2*(t - tj), 1, 0]))) # first-order derivative w.r.t time
    _hess = jax.vmap(lambda t, tj, p : jnp.dot(p, jnp.array([6*(t - tj), 2, 0, 0.]))) # second-order derivative w.r.t time
    _ghess = jax.vmap(lambda t, tj, p : jnp.dot(p, jnp.array([6, 0, 0, 0.]))) # third-order derivative w.r.t time

    def hvp(v, t) : 
        # This function evaluates the matrix-vector product. Therefore, in order to get the matrix, we must multiply it with identity matrix
        # v is vector of all the parameters (4 * (n - 1))
        _v = v.reshape(-1, 4) # shape (n - 1, 4)
        
        return jnp.concatenate([
            _f(t[:-1], t[:-1], _v), # (n - 1) equations
            _f(t[1:], t[:-1], _v), # (n - 1) equations
            _jac(t[1:-1], t[:-2], _v[:-1]) - _jac(t[1:-1], t[1:-1], _v[1:]), # (n - 2) equations
            _hess(t[1:-1], t[:-2], _v[:-1]) - _hess(t[1:-1], t[1:-1], _v[1:]), # (n - 2) equations
            _ghess(t[1:2], t[:1], _v[:1]) - _ghess(t[1:2], t[1:2], _v[1:2]), # 1 equation. Not-a-Knot spline
            _ghess(t[-2:-1], t[-3:-2], _v[-2:-1]) - _ghess(t[-2:-1], t[-2:-1], _v[-1:]) # 1 equation. Not-a-Knot spline
        ])

    y = jnp.atleast_2d(y) # shape (n, ny). Multiple dimensions of y is handled by the Linear solver
    return jnp.linalg.solve(
        jax.vmap(hvp, in_axes = (0, None))(jnp.eye(4 * (npoints - 1)), t).T, 
        jnp.concatenate((y[:-1], y[1:], jnp.zeros(shape = (2 * npoints - 2, y.shape[-1]))))
    ) # shape (4 * (n - 1), ny)
```

Notice that we use `jnp.linalg.solve` to solve the linear system, which returns the optimal values of the polynomial coefficients. Once the optimal values are found, we will now define a function `CubicSplineSimulate` that will evaluate the cubic spline interpolation at different query points. 

```python
def CubicSplineSimulate(ti, t, p) :    
    # This function evaluates the cubic spline for parameters p at ti query points.
    
    cubic_poly = lambda t, tj, p : jnp.dot(p, jnp.array([(t - tj)**3, (t - tj)**2, (t - tj), 1.]))
    p = p.reshape(-1, 4) # shape (n - 1, 4)

    # Append duplicates of first and last set of parameters to account for edge cases (ti < t0) & (ti > tf)
    _p = jnp.vstack((p[:1, :], p, p[-1:, :]))
    _t = jnp.array([-jnp.inf, *t, jnp.inf])
    _tj = jnp.array([t[0], *t[:-1], t[-2]])
    return jnp.sum(
        jnp.where(
            (ti > _t[:-1]) & (ti <= _t[1:]),
            jax.vmap(cubic_poly, in_axes = (None, 0, 0))(ti, _tj, _p),
            jnp.zeros_like(ti)
        )
    )
```

The evaluation proceeds by first determining the time interval in which the query point lies. If the query is outside the domain, the function returns $0$. If it is inside, the corresponding polynomial is evaluated. This logic is implemented using the `jnp.where` function, and the contributions from all intervals are then summed.

Finally, we create a convenience function that combines the two process - finding optimal parameters and evaluation. 

```python
@jax.jit
def CubicSpline(ti, t, y) :
    # Fully differentiable Cubic Spline Interpolation
    # Given measurements y at time points t. The time arguments are ti
    _y = y if y.ndim == 2 else y[:, jnp.newaxis] # makes sure that array is 2D. 
    popt = CubicSplineParameters(t, _y) 
    return jax.vmap(
        jax.vmap(CubicSplineSimulate, in_axes = (None, None, 1)), 
        in_axes = (0, None, None)
    )(jnp.atleast_1d(ti), t, popt) 
```

## 4. Evaluation and Gradient Computation

Lets test the function and compare the results with SciPy's CubicSpline interpolation. 

```python
npoints = 5 # number of points
t = jnp.arange(npoints, dtype = jnp.float64) # time points
y = jnp.column_stack((2 * jnp.sin(t), 2 * jnp.cos(t), 2 * jnp.tan(t))) # measurements
targ = targ = jnp.concatenate((t[:1] - 0.2, t[-1:] + 0.2, t[::4] + 0.2)) # query points

jinterp = CubicSpline(targ, t, y) # results at query points from JAX Cubic Spline
sinterp = SCubicSpline(t, y)(targ) # results at query points form SciPy Cubic Spline
```

```python
>>> jnp.allclose(sinterp, jinterp)
Array(True, dtype=bool)
```

Lets compute gradients using JAX. We first define a simple objective function that we want to take gradient over. 

```python
def obj(ti, t, y):
    # simple objective function
    sol = CubicSpline(ti, t, y)
    return jnp.mean((sol - jnp.ones_like(sol))**2)

# Compute reverse-mode gradients using JAX
loss, (grad_ti, grad_t, grad_y) = jax.value_and_grad(obj, argnums = (0, 2))(targ, t, y)

# comparing gradients using finite-difference
def fd_grad(eps):
    vars, unravel = flatten_util.ravel_pytree((targ, t, y))
    grads = jax.vmap(
        lambda v : (obj(*unravel(vars + eps * v)) - loss) / eps
    )(jnp.eye(len(vars)))
    return unravel(grads)

fd_ti, _, fd_y = fd_grad(1e-5)
```
```python
>>> all((jnp.allclose(fd_ti, grad_ti, 1e-3), jnp.allclose(fd_y, grad_y, 1e-3))) 
True
```

## 5. Efficient Implementation

When comparing execution times with npoints = 100, which is a typical scale in many scientific applications, we observe a substantial difference between our JAX implementation and SciPy’s implementation. The resulting times are as follows :

```python
JAX Cubic spline (jinterp):  0.38259243965148926 sec
Scipy Cubic spline (sinterp):  0.0015339851379394531 sec
```

The reason for this discrepancy is that our JAX implementation relies on jnp.linalg.solve to handle a dense linear system, whereas SciPy takes advantage of the tridiagonal structure of the system, leading to a much faster computation. In this section, we will incorporate SciPy’s optimized approach into our workflow while keeping the process differentiable.

JAX provides a convenient function, `jax.pure_callback`, which allows us to incorporate external functions that are not written in pure JAX [^2]. In addition, we need to define custom gradient rules for these external functions to ensure they remain differentiable within JAX’s computation graph [^3]. 


```python
@jax.custom_jvp
def CubicSplineParametersScipy(t, y) :
    
    def _scipy_interp_params(t, y) : 
        # external scipy function that returns the optimal parameters
        return jnp.vstack(jnp.einsum("ijk->jik", SCubicSpline(t, y).c))
    
    return jax.pure_callback(
        _scipy_interp_params, 
        jax.ShapeDtypeStruct((4 * (y.shape[0] - 1), y.shape[1]), y.dtype), 
        t, y)

@CubicSplineParametersScipy.defjvp
def CubicSplineParametersScipy_fwd(primals, tangents):
    t, y = primals
    _, ydot = tangents
    n, ny = y.shape

    # This implementation is not linear in the tangent space. 
    # Therefore only jvp's can be computed
    # p = CubicSplineParametersScipy(t, y)
    # p_out = CubicSplineParametersScipy(t, ydot)

    # This impelmentation makes the function linear in tangent space.
    # Therefore both jvp's and vjp's can be computed
    p, AinvI = jnp.array_split(
        CubicSplineParametersScipy(
            t, jnp.concatenate((y, jnp.eye(n)), axis = 1)
        ), 
        [ny], 
        axis = 1
    )
    p_out = AinvI @ ydot
    return p, p_out
```

We then edit the `CubicSpline` function to account for this change 

```python
@functools.partial(jax.jit, static_argnums = (3, ))
def CubicSpline(ti, t, y, method = "jax") :
    # Fully differentiable Cubic Spline Interpolation
    # Given measurements y at time points t. The time arguments are ti
    _y = y if y.ndim == 2 else y[:, jnp.newaxis] # makes sure that array is 2D. 
    popt = CubicSplineParameters(t, _y) if method == "jax" else CubicSplineParametersScipy(t, _y)
    return jax.vmap(
        jax.vmap(CubicSplineSimulate, in_axes = (None, None, 1)), 
        in_axes = (0, None, None)
    )(jnp.atleast_1d(ti), t, popt) 
```

Note that this implementation only gives correct gradients for time queries and the measurements $\{y_i \}_{i = 1}^{n + 1}$ . Gradients with respect to $\{t_i \}_{i = 1}^{n + 1}$ will be incorrect using this method.

```python
start = time.time()
jax.block_until_ready(jax.jacfwd(obj, argnums = (0, 2))(targ, t, y, "scipy"))
end = time.time()
print("Gradients (forward mode) using callback :", end - start, "sec")

start = time.time()
jax.block_until_ready(jax.jacfwd(obj, argnums = (0, 2))(targ, t, y))
end = time.time()
print("Gradients (forward mode) using JAX :", end - start, "sec")

start = time.time()
jax.block_until_ready(jax.value_and_grad(obj, argnums = (0, 2))(targ, t, y, "scipy"))
end = time.time()
print("Gradients (reverse mode) using callback", end - start, "sec")

start = time.time()
jax.block_until_ready(jax.value_and_grad(obj, argnums = (0, 2))(targ, t, y))
end = time.time()
print("Gradients (reverse mode) using JAX", end - start, "sec")

start = time.time()
jax.block_until_ready(jax.hessian(obj, argnums = (0, 2))(targ, t, y, "scipy"))
end = time.time()
print("Hessian (fwd-rev mode) using callback", end - start, "sec")

start = time.time()
jax.block_until_ready(jax.hessian(obj, argnums = (0, 2))(targ, t, y))
end = time.time()
print("Hessian (fwd-rev mode) using JAX", end - start, "sec")
```

```python
Gradients (forward mode) using callback : 1.0925776958465576 sec
Gradients (forward mode) using JAX : 1.1279046535491943 sec
Gradients (reverse mode) using callback 0.5884313583374023 sec
Gradients (reverse mode) using JAX 0.9264490604400635 sec
Hessian (fwd-rev mode) using callback 1.9281635284423828 sec
Hessian (fwd-rev mode) using JAX 1.6729705333709717 sec
```

Note that we use `npoints = 100` for the above computations. Although the forward evaluation is now orders of magnitude faster, the time required to compute gradients has improved only marginally. This slowdown occurs because, in our custom-JVP implementation, we do not have access to the original matrix factorization. As a result we are forced to form (or implicitly apply) the matrix inverse — effectively multiplying the function by the identity to obtain an inverse — and then multiply that inverse by a vector, rather than solving a linear system directly.

## 6. References 

[^1]: [Cubic Spline Interpolation](https://sites.millersville.edu/rbuchanan/math375/CubicSpline.pdf)
[^2]: [JAX external callbacks](https://docs.jax.dev/en/latest/external-callbacks.html)
[^3]: [JAX custom derivatives](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)