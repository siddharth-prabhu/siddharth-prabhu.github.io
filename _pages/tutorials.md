---
title: "Tutorials"
layout: single
permalink: /tutorials/
author_profile: true
---

- <a href="{{page.url}}ParameterEstimation/"> Parameter estimation of ordinary differential equations using CasADi </a>
- <a href="{{page.url}}CubicSpline/"> Differentiable Cubic Spline Interpolation in JAX </a>
- <a href="{{page.url}}DOpti/"> Differentiable Optimization in JAX </a>
- <a href="{{page.url}}ODEvent/"> Sensitivity Analysis of Hybrid Dynamical Systems </a>
- <a href=""> Condensing Approach to Parameter Estimation in JAX </a>

{% assign tutorials = site.categories.tutorials %}
{% if tutorials.size > 0 %}
  <div class="entries-list">
    {% for post in tutorials %}
      {% include archive-single.html type="post" %}
    {% endfor %}
  </div>
{% else %}
  <p>No posts found in the <code>tutorial</code> category.</p>
{% endif %}