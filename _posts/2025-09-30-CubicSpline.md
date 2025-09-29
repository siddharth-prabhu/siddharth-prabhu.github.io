---
title: "Differentiable Cubic Spline Interpolation in JAX"
date: 2025-09-30
permalink: /tutorials/CubicSpline/
categories: [tutorials]
---

## 1. Cubic Spline

Parameter estimation is a process of finding the optimal parameters of a given model using experimental data. In this tutorial, we will estimate the parameters of an ordinary differential equation (ODE) using three different methods 

- Single shooting (Sequential optimization)
- Multiple shooting (Simultaneous optimization)
- Orthogonal Collocation (Simultaneous optimization)

We will implement these methods using CasADi 

## 2. CasADi

CasADi is a open-source python package for automatic differentiation and nonlinear optimization [^1]. A lot of problems including parameter estimation of ODE's can be efficiently solved using CasADi. In this section we will quicklly walk through the basic syntax of CasADi. We begin by importing required packages. 

```python
import numpy as np
import casadi as cd 
import matplotlib.pyplot as plt
```

CasADi essentially creates symbolic expressions by treating everyting as matrix operation i.e vectors are treated as $n\times1$ matrix and scalar as $1\times1$ matrix. It has two basic data structures

- **The SX symbolics** : used to define matrices where mathematical operations are performed element wise.

    ```python
    >>> x = cd.SX.sym("x", 2, 2)
    >>> 2 * x + 1
    SX(@1=2, @2=1, 
    [[((@1*x_0)+@2), ((@1*x_2)+@2)], 
    [((@1*x_1)+@2), ((@1*x_3)+@2)]])
    ```

- **The MX symbolics** : used to define matrices where mathematical operations are performed over the entire matrix. 

    ```python
    >>> x = cd.MX.sym("x", 2, 2)
    >>> 2 * x + 1
    MX((ones(2x2)+(2.*x)))
    ```

Once symbolic variables are defined, you can create mathematical expressions directly using CasADi primitives:

- **Mathematical operations** such as addition (`x + x`), multiplication (`x * x`), trignometric functions (`cd.sin(x)`), matrix multiplications (`cd.mtimes(x, x)`)
- **Linear algebra** such as solving linear systems (`cd.solve(A, b)`), root finding problem (`cd.rootfinder(g)`)
- **Control flow** such as if-else statements (`cd.if_else(*args)`)
- **Automatic differentiation** using forward (Jacobian-vector-product) and reverse mode (vector-Jacobian-product) (`cd.jacobian(A@x, x)`)

Using these mathematical expressions, CasADi can also be used for optimization. Lets look at the CasADi syntax for solving a simple optimization problem given below using the `cd.Opti` optimization helper class

$$
\begin{equation}
\begin{aligned}
    & \min _{p_1, \ p_2} \ \ (p_1 - 1)^2 + (p_2 - 2)^2 \\ 
    \text{subject to} & \\
    & p_1 \geq 2
\end{aligned}
\end{equation}
$$

```python
opti = cd.Opti() # Initialize CasADi optimization helper class
p = opti.variable(2) # Define decision variables

opti.minimize((p[0] - 1)**2 + (p[1] - 2)**2) # Define objective function
opti.subject_to(p[0] >= 2) # Define constraints
opti.set_initial(p, np.array([0, 0])) # Define initial conditions

plugin_options = {} # Plugin options
solver_options = {"max_iter" : 100, "tol" : 1e-5} # solver options
opti.solver("ipopt", plugin_options, solver_options) # Initialize optimization solver
optimal = opti.solve() # Solve the problem
print("Optimal Parameters : ", optimal.value(p)) # Get the optimal parameters
```