---
title: "Tutorials"
layout: archive
permalink: /tutorials/
author_profile: true
---

<h2>All posts detected in 'tutorials'</h2>
<ul>
{% assign tposts = site.categories.tutorials %}
{% if tposts %}
  {% for post in tposts %}
    <li>{{ post.date }} - {{ post.title }} - {{ post.url }}</li>
  {% endfor %}
{% else %}
  <li>No posts detected in tutorials category!</li>
{% endif %}
</ul>

## Custom
- <a href="{{page.url}}ParameterEstimation"> Parameter estimation of ordinary differential equations using CasADi </a>
- <a href="{{page.url}}CubicSpline"> Differentiable Cubic Spline Interpolation in JAX </a>
- <a href="{{page.url}}ODEvent"> Differentiable ordinary differential equation solver with events in JAX </a>