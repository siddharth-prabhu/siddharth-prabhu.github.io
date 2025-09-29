---
title: "Tutorials"
layout: archive
permalink: /tutorials/
author_profile: true
category: tutorials
entries_layout: list
show_excerpts: true
---

<ul>
{% for post in site.categories.tutorials %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>

- <a href="{{site.categories.tutorials | CubicSpline}}"> Differentiable Cubic Spline Interpolation in JAX </a>
- <a href="{{site.categories.tutorials | ODEvent}}"> Differentiable ordinary differential equation solver with events in JAX </a>