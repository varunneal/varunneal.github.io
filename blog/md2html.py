#!/usr/bin/env python3

import sys
import re
import os
import markdown


def main(md_file):

    base_name = os.path.splitext(md_file)[0]
    safe_base = re.sub(r'\s+', '-', base_name)
    output_file = f"{safe_base}.html".lower()

    doc_title = base_name
    with open(md_file, 'r', encoding='utf-8') as f:
        md_body = f.read()

    html_content = markdown.markdown(md_body, extensions=['fenced_code', 'tables'])
    with open('template.html', 'r', encoding='utf-8') as f:
        template = f.read()

    final_html = (
        template
        .replace("__TITLE__", doc_title)
        .replace("__CONTENT__", html_content)
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"Output written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py input.md")
        sys.exit(1)
    main(sys.argv[1])
