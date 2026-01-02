---
layout: page
permalink: /repositories/
title: projects
description:
nav: true
nav_order: 4
---

{% if site.data.repositories.projects %}
<div class="projects">
  {% for project in site.data.repositories.projects %}
    {% include repository/project.liquid project=project %}
  {% endfor %}
</div>
{% endif %}
