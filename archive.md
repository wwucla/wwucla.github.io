---
layout: page
title: Blog Archive
---

{%- comment -%}
  Tag cloud at the top: count posts per tag and render as chips that jump to
  the section below. site.tags is a hash { tag_name => [posts...] }.
{%- endcomment -%}

<ul class="tag-cloud">
{%- assign tag_names = site.tags | sort -%}
{%- for tag in tag_names -%}
  {%- assign tag_slug = tag[0] | slugify -%}
  <li><a class="tag-chip" href="#tag-{{ tag_slug }}">
    {{ tag[0] }} <span class="tag-count">{{ tag[1].size }}</span>
  </a></li>
{%- endfor -%}
</ul>

{%- for tag in tag_names -%}
  {%- assign tag_slug = tag[0] | slugify -%}

### {{ tag[0] }} <span class="tag-count">({{ tag[1].size }})</span>
{: id="tag-{{ tag_slug }}" }

{% for post in tag[1] %}- [{{ post.date | date: "%B %Y" }} &mdash; {{ post.title }}]({{ post.url | relative_url }})
{% endfor %}

{% endfor %}
