---
title: "Differentiable Cubic Spline Interpolation in JAX"
date: 2025-09-30
layout: archive
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

## Table of Contents

1. [Cubic Spline Interpolation](#1-cubic-spline-interpolation)
2. [Optimal Parameters](#2-optimal-parameters)
3. [JAX Implementation](#3-jax-implemention)
4. [Evaluation and Gradient Computation](#4-evaluation-and-gradient-computation)
5. [Efficient Implementation](#5-efficient-implementation)


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

where $[a_i, \ b_i, \ c_i, \ d_i] _{i = 1}^{n} $ are the coefficients of $[f_i] _{i = 1}^{n}$ polynomial respectively, determined using $n + 1$ measurements $[t_i, \ y_i ] _{i = 1}^{n + 1}$. In this tutorial, we will implement cubic spline interpolation in JAX, ensuring that it is fully differentiable with respect to its arguments

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

With these $4n$ equations — consisting of $2n$ evaluation constraints, $2(n−1)$ smoothness conditions, and $2$ boundary conditions — we have exactly enough to solve for the $4n$ unknown parameters. These parameters can be obtained by solving a linear system of equations, which is given as follows. 

$$
\begin{equation}
A(t)p = b(y)
\end{equation} 
$$

where $p \in \mathbb{R}^{4n}$ are all the parameters of the polynomials, $A(t) \in \mathbb{R}^{4n}$ is a matrix of coefficients corresponding to these parameters and $b(y) \in \mathbb{R}^{4n}$ is a vector of measurements and zeros. Let us now implement this procedure in JAX.

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

We will now create a function called `CubicSplineParameters` that will take the measurements $[t_i, \ y_i] _{i = 1}^{n + 1} $ as input and return the optimal parameters of the polynomials as output

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

When comparing execution times with `npoints = 5000`, which is a typical scale in many scientific applications, we observe a substantial difference between our JAX implementation and SciPy’s implementation. The resulting times are as follows :

```python
Simulate using JAX :  20.629136323928833 sec
Simulate using Scipy Cubic spline :  0.0026857852935791016 sec
```

This discrepancy arises from two main factors. First, our JAX implementation constructs the matrix $A$ to be inverted by applying the function `hvp` to the identity matrix, since the function directly returns $Ax$ rather than $A$ itself. Second, we use `jnp.linalg.solve` to solve a dense linear system, whereas SciPy exploits the tridiagonal structure of the system, enabling much faster computations.

In this section, we will first use a sparse matrix solver and then integrate SciPy’s optimized approach into our workflow while ensuring the entire process remains differentiable. Instead of reimplementing everything in JAX, we can use the convenient `jax.pure_callback`, function, which allows us to incorporate external functions that are not written purely in JAX [^2]. However, to maintain differentiability, we must define custom gradient rules for these external functions so they remain compatible with JAX’s computation graph. [^3]. 

Instead of computing the entire matrix using the `hvp` function as we did before, we will now only store the nonzero elements of this matrix $A$ denoted by its row index, column index and values. Using `jax.pure_callback`, we will then form a sparse matrix and solve the linear system of equations. We also define a custom reverse-mode differentiation rule using `jax.custom_vjp`. Note that this implementation only gives correct gradients for time queries and the measurements $[y_i] _{i = 1}^{n + 1} $. Gradients with respect to $[t_i] _{i = 1}^{n + 1} $ are not implemented. The reverse-mode differentiation rule is as follows 


$$
\begin{equation}
\begin{aligned}
    p & = A^{-1}b \\
    v^T \partial p & = v^T A^{-1} \partial b \\
    \text{If} \quad w^T & = v^T A^{-1} \quad \text{then} \quad v^T \partial p = w^T \partial b \\
    w & = A^{T^{-1}} v \rightarrow \quad \text{Cubic Spline Solve with matrix transpose}
\end{aligned}
\end{equation}
$$


```python
@partial(jax.custom_vjp, nondiff_argnums = (4, ))
def SparseLinearSolve(rows, cols, values, b, transpose = False):
    # A is non differentiable
    def _spsolve(rows, cols, values, b):
        A = sparse.csr_matrix((values, (rows, cols)), shape = (len(b), len(b)))
        return sparse.linalg.spsolve(A.T, b) if transpose else sparse.linalg.spsolve(A, b)

    return jax.pure_callback(_spsolve, jax.ShapeDtypeStruct(b.shape, b.dtype), rows, cols, values, b)
    
def SparseLinearSolve_fwd(rows, cols, values, b, transpose):
    p = SparseLinearSolve(rows, cols, values, b, transpose)
    return p, (rows, cols, values, b)

def SparseLinearSolve_bwd(transpose, res, gdot):
    rows, cols, values, b = res
    return None, None, None, SparseLinearSolve(rows, cols, values, gdot, transpose)

SparseLinearSolve.defvjp(SparseLinearSolve_fwd, SparseLinearSolve_bwd)


def CubicSplineParametersSparse(t, y) : 
    # Gives the optimal values of parameters of cubic polynomial given time range t and function values y
    
    npoints = len(t)
    cubic_poly = lambda t, tj : jnp.array([(t - tj)**3, (t - tj)**2, (t - tj), 1.])
    _f = jax.vmap(cubic_poly) # polynomial evaluation
    _jac = jax.vmap(lambda t, tj : jnp.array([3*(t - tj)**2, 2*(t - tj), 1, 0])) # first-order derivative w.r.t time
    _hess = jax.vmap(lambda t, tj : jnp.array([6*(t - tj), 2, 0, 0.])) # second-order derivative w.r.t time
    _ghess = jax.vmap(lambda t, tj : jnp.array([6, 0, 0, 0.])) # third-order derivative w.r.t time

    rows = jnp.arange(npoints - 1)
    cols = jnp.arange(4 * (npoints - 1)).reshape(-1, 4)
    A = jnp.zeros(shape = (4 * (npoints - 1), 4 * (npoints - 1)))

    rows = jnp.concatenate((
        rows, 
        rows + (npoints - 1),
        rows[:-1] + 2 * (npoints - 1),
        rows[:-1] + 2 * (npoints - 1),
        rows[:-1] + 3 * (npoints - 1) - 1,
        rows[:-1] + 3 * (npoints - 1) - 1, 
        4 * (rows[-1:] + 1) - 2,
        4 * (rows[-1:] + 1) - 2,
        4 * (rows[-1:] + 1) - 1,
        4 * (rows[-1:] + 1) - 1 
    )).reshape(-1, 1)

    cols = jnp.vstack([
        cols, 
        cols, 
        cols[:-1], #(1)
        cols[1:], #(1)
        cols[:-1], #(2)
        cols[1:], #(2)
        cols[:1], #(3)
        cols[1:2], #(3)
        cols[-2:-1], #(4)
        cols[-1:] #(4)
    ])

    _rows, _cols = jnp.vstack(jax.vmap(lambda r, c : jnp.asarray(jnp.meshgrid(r, c)).T.reshape(-1, 2))(rows, cols)).T

    values = jnp.vstack(( 
        _f(t[:-1], t[:-1]), 
        _f(t[1:], t[:-1]), 
        _jac(t[1:-1], t[:-2]), 
        - _jac(t[1:-1], t[1:-1]), 
        _hess(t[1:-1], t[:-2]), 
        - _hess(t[1:-1], t[1:-1]),
        _ghess(t[1:2], t[:1]), 
        - _ghess(t[1:2], t[1:2]),
        _ghess(t[-2:-1], t[-3:-2]), 
        - _ghess(t[-2:-1], t[-2:-1])
    )).flatten()

    y = jnp.atleast_2d(y) # shape (n, ny)
    return SparseLinearSolve(
        _rows, _cols, values, 
        jnp.concatenate((y[:-1], y[1:], jnp.zeros(shape = (2 * npoints - 2, y.shape[-1]))))
    ) # shape (4 * (n - 1), ny)

@partial(jax.jit, static_argnums = (3, ))
def CubicSpline(ti : jnp.ndarray, t : jnp.ndarray, y : jnp.ndarray, method : str = "jax") -> jnp.ndarray :
    # https://sites.millersville.edu/rbuchanan/math375/CubicSpline.pdf
    # Fully differentiable Cubic Spline Interpolation
    # Given measurements y at time points t. The time arguments are ti
    _y = y if y.ndim == 2 else y[:, jnp.newaxis] # makes sure that array is 2D. 
    
    if method == "jax" : 
        popt = CubicSplineParameters(t, _y) # JAX + dense matrix inverse
    elif method == "sparse" :
        popt = CubicSplineParametersSparse(t, _y) # JAX + sparse matrix inverse
    else : 
        assert False, "Method not implemented"
    
    return jax.vmap(
        jax.vmap(CubicSplineSimulate, in_axes = (None, None, 1)), 
        in_axes = (0, None, None)
    )(jnp.atleast_1d(ti), t, popt) 
```

Lets compare the performance of this approach with our earlier approach

```python
start = time.time()
_ = jax.block_until_ready(CubicSpline(targ, t, y, "sparse"))
end = time.time()
print("Simulate using spsolve :", end - start)

start = time.time()
_ = jax.block_until_ready(CubicSpline(targ, t, y))
end = time.time()
print("Simulate using JAX :", end - start)

start = time.time()
loss, (tidot, ydot) = jax.block_until_ready(jax.value_and_grad(obj, argnums = (0, 2))(targ, t, y, "sparse"))
end = time.time()
print("Gradients (reverse mode) using Spsolve", end - start)

start = time.time()
loss, (tidot, ydot) = jax.block_until_ready(jax.value_and_grad(obj, argnums = (0, 2))(targ, t, y))
end = time.time()
print("Gradients (reverse mode) using JAX", end - start)
```

```python
Simulate using spsolve : 0.12991094589233398 sec
Simulate using JAX : 20.629136323928833 sec
Simulate using Scipy Cubic spline :  0.0026857852935791016 sec
Gradients (reverse mode) using Spsolve 10.910008668899536 sec
Gradients (reverse mode) using JAX 35.83856439590454 sec
```

We observe that our current approach is significantly faster than the previous one; however, it is still not as fast as the pure SciPy implementation. While we can implement the SciPy version using `jax.pure_callback`, we cannot define a custom VJP rule in this case, since we do not have access to the matrix or its inverse/decomposition. Doing so makes the differentiation rule linear in the (co-)tangent space, allowing a custom JVP rule to suffice. 

$$
\begin{equation}
\begin{aligned}
    p & = A^{-1}b \\
    \partial p \ v & = A^{-1} \partial b \ v \\
\end{aligned}
\end{equation}
$$


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
@partial(jax.jit, static_argnums = (3, ))
def CubicSpline(ti : jnp.ndarray, t : jnp.ndarray, y : jnp.ndarray, method : str = "jax") -> jnp.ndarray :
    # https://sites.millersville.edu/rbuchanan/math375/CubicSpline.pdf
    # Fully differentiable Cubic Spline Interpolation
    # Given measurements y at time points t. The time arguments are ti
    _y = y if y.ndim == 2 else y[:, jnp.newaxis] # makes sure that array is 2D. 
    
    if method == "jax" : 
        popt = CubicSplineParameters(t, _y) # JAX + dense matrix inverse
    elif method == "sparse" :
        popt = CubicSplineParametersSparse(t, _y) # JAX + sparse matrix inverse
    else : 
        popt = CubicSplineParametersScipy(t, _y) # JAX + scipy
    
    return jax.vmap(
        jax.vmap(CubicSplineSimulate, in_axes = (None, None, 1)), 
        in_axes = (0, None, None)
    )(jnp.atleast_1d(ti), t, popt) 
```

Comparing its performance with the rest of the approaches. 


```python
start = time.time()
_ = jax.block_until_ready(CubicSpline(targ, t, y, "scipy"))
end = time.time()
print("Simulate using SciPy callback :", end - start)

start = time.time()
jax.block_until_ready(jax.value_and_grad(obj, argnums = (0, 2))(targ, t, y, "scipy"))
end = time.time()
print("Gradients (reverse mode) using SciPy callback", end - start, "sec")
```

```python
Simulate using scipy callback : 0.11706137657165527 sec
Gradients (reverse mode) using SciPy callback 13.912312746047974 sec
```

We observe that this approach still performs better than our initial method, where we solved a dense linear system, and performs comparably to our sparse linear system approach. However, it does not perform as well as the SciPy implementation, likely due to the additional overhead costs in JAX. A major advantage of this method is that it enables efficient computation of higher-order derivatives. For instance, the Hessian can be computed using forward-over-reverse mode, whereas in the previous approach it could only be obtained using the less efficient reverse-over-reverse mode.

```python
start = time.time()
tidot, ydot = jax.block_until_ready(jax.hessian(obj, argnums = (0, 2))(targ, t, y, "scipy"))
end = time.time()
print("Hessian (fwd-rev mode) using SciPy callback", end - start)

start = time.time()
tidot, ydot = jax.block_until_ready(jax.hessian(obj, argnums = (0, 2))(targ, t, y))
end = time.time()
print("Hessian (fwd-rev mode) using JAX", end - start)

start = time.time()
tidot, ydot = jax.block_until_ready(jax.jacrev(jax.jacrev(obj, argnums = (0, 2)), argnums = (0, 2))(targ, t, y))
end = time.time()
print("Hessian (rev-rev mode) using Spsolve", end - start)
```

```python
Hessian (fwd-rev mode) using SciPy callback 20.535470724105835 sec
Hessian (fwd-rev mode) using JAX 10.610387802124023 sec
Hessian (rev-rev mode) using Spsolve 13.807488918304443 sec
```

Note that we use `npoints = 500` in our Hessian computations to manage computational and memory constraints. We observe that although our latest approach enables forward-over-reverse computation of the Hessian, it remains computationally expensive because the inverse is computed within the differentiation rule. Interestingly, our original JAX implementation turns out to be faster.


## 6. References 

[^1]: [Cubic Spline Interpolation](https://sites.millersville.edu/rbuchanan/math375/CubicSpline.pdf)
[^2]: [JAX external callbacks](https://docs.jax.dev/en/latest/external-callbacks.html)
[^3]: [JAX custom derivatives](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)