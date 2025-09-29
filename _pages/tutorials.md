---
title: "Tutorials"
layout: archive
permalink: /tutorials/
author_profile: true
category: tutorials
entries_layout: list
show_excerpts: true
---

<h2>Debug list</h2>
<ul>
{% for post in site.categories.tutorials %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>