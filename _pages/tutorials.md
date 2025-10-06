---
title: "Tutorials"
layout: single
permalink: /tutorials/
author_profile: true
toc: false
toc_label: "Table of Contents"
toc_icon: "gear"
toc_sticky: true
---

{% assign tutorials = site.categories.tutorials | sort: "date" | reverse %}
{% if tutorials.size > 0 %}
  <div class="entries-list">
    {% for post in tutorials %}
      {% include archive-single.html type="post" %}
      {% if post.excerpt %}
        <p class="tutorial-intro">{{ post.excerpt | strip_html | truncate: 200 }}</p>
      {% endif %}
    {% endfor %}
  </div>
{% else %}
  <p>No posts found in the <code>tutorial</code> category.</p>
{% endif %}