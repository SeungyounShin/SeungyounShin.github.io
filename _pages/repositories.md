---
layout: page
permalink: /repositories/
title: repositories
description: Open-source work, mostly agents and the training around them.
nav: true
nav_order: 3
---

{% if site.data.repositories.github_repos %}
<div class="c-repo-list">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.html repository=repo %}
  {% endfor %}
</div>
{% endif %}

<p class="c-repo-more">
  More at <a href="https://github.com/{{ site.github_username }}" target="_blank" rel="noopener">github.com/{{ site.github_username }}</a>.
</p>
