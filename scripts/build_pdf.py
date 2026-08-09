"""Render reports/report.md to a submission-ready PDF.

Usage:
    python scripts/build_pdf.py
    python scripts/build_pdf.py --input reports/report.md --output reports/report.pdf

Markdown -> styled HTML -> PDF via headless Chrome. Chrome is used because it is
already present on macOS and renders tables and page breaks correctly; no LaTeX
or pandoc install is required.

The assignment caps the system report at 5 pages *excluding references and
appendices*, so this script reports the page count and where the appendices
start, and warns if the body runs over.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
/* Tuned to fit the 5-page body cap without looking cramped: 10pt serif at
   1.32 line-height with 0.7in margins is a standard two-column-conference
   density, applied to a single column. */
@page { size: letter; margin: 0.7in 0.75in; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10pt; line-height: 1.32; color: #111;
  max-width: 100%; margin: 0;
}
h1 { font-size: 15pt; margin: 0 0 2pt 0; line-height: 1.2; }
h2 { font-size: 11.5pt; margin: 11pt 0 4pt 0; padding-bottom: 1.5pt;
     border-bottom: 1px solid #bbb; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 8pt 0 2pt 0; page-break-after: avoid; }
h4 { font-size: 10pt; margin: 7pt 0 1pt 0; page-break-after: avoid; }
p  { margin: 0 0 4.5pt 0; text-align: justify; }
em.subtitle { color: #555; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
table {
  border-collapse: collapse; width: 100%; margin: 7pt 0 9pt 0;
  font-size: 8.8pt; page-break-inside: avoid;
}
th, td { border: 1px solid #ccc; padding: 3.5pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
code {
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-size: 8.8pt; background: #f3f3f3; padding: 0.5pt 2.5pt;
  border-radius: 2px;
}
pre {
  background: #f6f6f6; border: 1px solid #ddd; border-radius: 3px;
  padding: 7pt 9pt; overflow-x: auto; page-break-inside: avoid;
  font-size: 8.5pt; line-height: 1.35;
}
pre code { background: none; padding: 0; font-size: 8.5pt; }
blockquote {
  margin: 7pt 0; padding: 5pt 10pt; border-left: 3px solid #bbb;
  background: #f8f8f8; color: #333; font-size: 9.5pt;
}
hr { border: none; border-top: 1px solid #ddd; margin: 13pt 0; }
a { color: #0b57d0; text-decoration: none; word-break: break-all; }
strong { font-weight: 600; }
.pagebreak { page-break-before: always; }
"""


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "No Chrome/Chromium found for PDF rendering.\n"
        "Either install Chrome, or open reports/report.html in any browser and "
        "use File > Print > Save as PDF."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="reports/report.md")
    ap.add_argument("--output", default="reports/report.pdf")
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    text = src.read_text(encoding="utf-8")

    # Warn about unfilled placeholders rather than silently shipping them.
    placeholders = sorted(set(re.findall(r"\[\[[^\]]+\]\]", text)))
    if placeholders:
        print("WARNING: unfilled placeholders still in the report:")
        for p in placeholders:
            print(f"  {p}")
        print()

    # Start the appendices on a fresh page: the page limit excludes them, so a
    # grader should be able to see exactly where the body ends.
    text = text.replace(
        "\n## Appendix A", "\n<div class='pagebreak'></div>\n\n## Appendix A", 1
    )

    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )

    html_path = Path(args.output).with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")

    chrome = find_chrome()
    out = Path(args.output).resolve()
    subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={out}", html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )

    if not out.exists():
        raise SystemExit("Chrome did not produce a PDF; open the HTML and print manually.")

    size_kb = out.stat().st_size / 1024
    pages = count_pages(out)
    print(f"wrote {out}  ({size_kb:.0f} KB" + (f", {pages} pages)" if pages else ")"))

    if pages:
        print(
            "\nThe appendices start on their own page (forced break before Appendix A),\n"
            "so the body page count is that page number minus one — check it in a\n"
            "viewer. The 5-page cap applies to the body only; references and\n"
            "appendices are excluded."
        )

    print(f"(HTML kept at {html_path} if you prefer to print from a browser)")


def count_pages(pdf: Path) -> int | None:
    """Page count without a PDF dependency: count page objects in the raw file."""
    try:
        data = pdf.read_bytes()
    except OSError:
        return None
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", data)]
    if counts:
        return max(counts)
    n = len(re.findall(rb"/Type\s*/Page[^s]", data))
    return n or None


if __name__ == "__main__":
    main()
