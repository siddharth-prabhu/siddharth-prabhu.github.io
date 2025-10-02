---
title: "Tutorials"
layout: archive
permalink: /tutorials/
author_profile: true
---

- <a href="{{page.url}}ParameterEstimation/"> Parameter estimation of ordinary differential equations using CasADi </a>
- <a href="{{page.url}}CubicSpline/"> Differentiable Cubic Spline Interpolation in JAX </a>
- <a href="{{page.url}}DOpti/"> Differentiable Optimization in JAX </a>
- <a href="{{page.url}}ODEvent/"> Sensitivity Analysis of Hybrid Dynamical Systems </a>


{% for post in site.categories.tutorials %}
<li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
</li>
{% endfor %}