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


{% assign posts = site.categories.tutorials | sort: "date" | reverse %}
<ul>
{% for post in posts %}
<li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small>— {{ post.date | date: "%b %-d, %Y" }}</small>
    {% if post.excerpt %}
    <div class="excerpt">{{ post.excerpt | strip_html | truncatewords: 25 }}</div>
    {% endif %}
</li>
{% endfor %}
</ul>