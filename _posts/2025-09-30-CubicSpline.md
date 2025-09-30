---
title: "Differentiable Cubic Spline Interpolation in JAX"
date: 2025-09-30
layout: single
classes: wide
categories: 
  -tutorials
permalink: /tutorials/CubicSpline/
use_math: true
author_profile: false
toc: true
toc_label: "Table of Contents"
toc_icon: "gear"
toc_sticky: true
---

## 1. Cubic Spline Interpolation

Cubic splines are an interpolation method that construct a smooth curve by joining together cubic polynomials between data points. More precisely, the interpolant is defined as a piecewise cubic polynomial $ f :[t_1, t_{n + 1}] \to \mathbb{R}$ defined as 

$$
\begin{equation}
f(t_{\text{query}}) =
\begin{cases}
  a_1t^3 + b_1t^2 + c_1t + d_1, & \text{if}\ t_{\text{query}} \in [t_1, t_2] \\
  a_2t^3 + b_2t^2 + c_2t + d_2, & \text{if}\ t_{\text{query}} \in (t_2, t_3] \\
  \qquad \qquad \quad \vdots \\
  a_nt^3 + b_nt^2 + c_nt + d_n, & \text{if}\ t_{\text{query}} \in (t_n, t_{n + 1}] \\
\end{cases}
\end{equation} 
$$

where $\{a_i, \ b_i, \ c_i, \ d_i\}_{i = 1}^{n}$ are the coefficients of $n$ pieciwise polynomials, determined using $n + 1$ measurements $\{t_i, \ f(t_i)\}_{i = 1}^{n + 1}$. In this tutorial, n this tutorial, we will implement cubic spline interpolation in JAX, ensuring that it is fully differentiable with respect to its arguments

## 2. Optimal Parameters

## 3. 