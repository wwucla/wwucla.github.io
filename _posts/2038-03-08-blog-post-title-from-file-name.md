---
layout: post
title: "Sample Post: Code Block Rendering"
date: 2038-03-08
categories: [Demo]
tags: [Demo]
description: A future-dated sample post used to verify code block rendering and syntax highlighting across the supported languages.
---

This is a future-dated sample post used to verify that fenced code blocks
render correctly and that highlight.js applies syntax highlighting for each
language module loaded in `_includes/head.html` (currently: `plaintext`,
`powershell`, and `tsql`).

## T-SQL

```tsql
SELECT This, [Is], A, Code, Block -- Using SSMS style syntax highlighting
    , REVERSE('abc')
FROM dbo.SomeTable s
    CROSS JOIN dbo.OtherTable o;
```

## PowerShell

```powershell
Write-Host "This is a PowerShell code block";

# There are many other languages you can use, but the style has to be loaded first.
ForEach ($thing in $things) {
    Write-Output "It highlights it using the GitHub style"
}
```

## Plain text

```plaintext
No syntax highlighting here — just a fenced code block.
The plaintext language module ensures this falls back cleanly
instead of being marked up as "unknown" by highlight.js.
```

## Inline code

You can also use `inline code` mid-paragraph; inline code should NOT be
syntax-highlighted, and the override CSS keeps its background transparent
so it reads as part of the prose.

## Adding more languages

To highlight another language (e.g., Python or JavaScript), add the matching
module under `js/highlightjs/languages/` and load it in `_includes/head.html`
alongside the existing `<script>` tags. The Markdown side stays the same —
just fence the block with the language name after the opening backticks.
