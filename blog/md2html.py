#!/usr/bin/env python3

import sys
import re
import os
import markdown

template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@font-face {{
    font-family: 'matisse bold';
    src: url('assets/webfonts/fotmatissebold.otf') format('opentype');
    font-display: optional;
}}
body {{
    font-family: Helvetica, Arial, sans-serif;
    margin: 20px;
    transition: background-color 0.3s ease, color 0.3s ease;
    display: flex;
    justify-content: center;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
.wrapper {{
    max-width: 700px;
    width: 100%;
}}
.light-mode {{
    background-color: #f8f8f0;
    color: #2d2d2d;  /* Slightly softer than #333 */
}}
.dark-mode {{
    background-color: #1f1f1f;
    color: #f0f0f0;  /* Slightly softer than #eee */
}}
.light-mode h1, .light-mode h2 {{
    font-family: 'matisse bold', "Times New Roman", Times, serif;
    color: #333;
}}
.dark-mode h1, .dark-mode h2 {{
    font-family: 'matisse bold', "Times New Roman", Times, serif;
    color: #eee;
}}
code {{
    font-family: Menlo, Consolas, "Liberation Mono", "Courier New", monospace;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-size: 0.9em;
    white-space: pre-wrap;
    word-wrap: break-word;
}}

/* Separate styles for code blocks vs inline code */
pre code {{
    padding: 0;  /* Remove padding for code inside pre blocks */
    display: block;
    overflow-x: auto;
    margin: 1em 0;
}}

pre {{
    padding: 1em;
    border-radius: 6px;
    margin: 1em 0;
}}
.light-mode code, .light-mode pre {{
    background-color: #f0f0f0;
    color: #333;
}}
.dark-mode code, .dark-mode pre {{
    background-color: #333;
    color: #ccc;
}}
.toggle-btn {{
    cursor: pointer;
    position: fixed;
    top: 20px;
    right: 20px;
    background: none;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 14px;
    min-width: 100px;  /* Set consistent width */
    text-align: center;  /* Center the text within the button */
    transform: scale(1);
    transition: transform 0.15s ease, color 0.3s ease, border-color 0.3s ease;
}}
.light-mode .toggle-btn {{
    color: #333;
    border: 1px solid #333;
}}
.dark-mode .toggle-btn {{
    color: #eee;
    border: 1px solid #eee;
}}
.light-mode a:hover {{
    text-decoration: underline;
}}

.light-mode a:visited {{
    color: #551A8B;
}}

/* Link styles for dark mode */
.dark-mode a {{
    color: #58a6ff;  /* Brighter blue for dark mode */
    text-decoration: none;
}}

.dark-mode a:hover {{
    text-decoration: underline;
    color: #79b8ff;  /* Even brighter on hover */
}}

.dark-mode a:visited {{
    color: #b392f0;  /* Purple for visited links */
}}

.dark-mode a:active {{
    color: #ffab70;  /* Orange flash when clicked */
}}
</style>
</head>
<body class="light-mode">
<button class="toggle-btn" onclick="toggleMode()">Dark Mode</button>
<div class="wrapper">
<h1>{title}</h1>
{content}
</div>
<script>
function toggleMode() {{
  var body = document.body;
  body.classList.toggle('dark-mode');
  body.classList.toggle('light-mode');
  document.querySelector('.toggle-btn').textContent =
    body.classList.contains('dark-mode') ? 'Light Mode' : 'Dark Mode';
}}

</script>
</body>
</html>
"""

def main(md_file):
    base_name = os.path.splitext(md_file)[0]
    safe_base = re.sub(r'\s+', '-', base_name)
    output_file = f"{safe_base}.html".lower()

    doc_title = base_name
    with open(md_file, 'r', encoding='utf-8') as f:
        md_body = f.read()

    html_content = markdown.markdown(md_body, extensions=['fenced_code', 'tables'])
    final_html = template.format(title=doc_title, content=html_content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"Output written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py input.md")
        sys.exit(1)
    main(sys.argv[1])
