#!/usr/bin/env python3
"""
Build script to combine all ensureStudy documentation into a single PDF.
Uses Pandoc + XeLaTeX + mermaid-filter for Mermaid diagram rendering.
"""

import os
import re
import glob
import subprocess
import sys

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DOCS_DIR, "ensureStudy_Complete_Documentation.pdf")
COMBINED_MD = os.path.join(DOCS_DIR, "_combined_documentation.md")

# Emoji regex pattern (covers most Unicode emoji ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001f926-\U0001f937"  # supplemental
    "\U00010000-\U0010ffff"  # supplemental
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)


def get_ordered_files():
    """Get all markdown files in proper order: numbered first, then legacy."""
    numbered = []
    legacy = []

    for f in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        basename = os.path.basename(f)
        # Skip build artifacts and this script's output
        if basename.startswith("_") or basename == "build_pdf.py":
            continue
        # Check if it starts with a number
        if re.match(r"^\d+_", basename):
            numbered.append(f)
        else:
            legacy.append(f)

    # Sort numbered files naturally
    numbered.sort(key=lambda x: int(re.match(r"(\d+)", os.path.basename(x)).group(1)))
    # Sort legacy alphabetically
    legacy.sort()

    return numbered + legacy


def strip_emojis(text):
    """Remove all emoji characters from text."""
    return EMOJI_PATTERN.sub("", text)


def clean_markdown(content, filename):
    """Clean a markdown file for professional PDF output."""
    # Strip emojis
    content = strip_emojis(content)

    # Remove standalone horizontal rules that are just "---" on their own line
    # (Pandoc handles section breaks via headers)
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        # Keep horizontal rules but clean up excessive ones
        cleaned_lines.append(line)

    content = "\n".join(cleaned_lines)

    # Ensure proper page break before each top-level header (# Page ...)
    # by adding a LaTeX pagebreak command
    content = re.sub(
        r"\n(# Page \d+)",
        r"\n\\newpage\n\1",
        content,
    )

    return content


def combine_files(files):
    """Combine all markdown files into a single document."""
    combined = []

    for filepath in files:
        basename = os.path.basename(filepath)
        print(f"  Adding: {basename}")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Clean the content
        content = clean_markdown(content, basename)

        # Add page break between files
        if combined:
            combined.append("\n\\newpage\n")

        combined.append(content)

    return "\n\n".join(combined)


def create_metadata_yaml():
    """Create YAML metadata block for Pandoc."""
    return """---
title: "ensureStudy - Complete Technical Documentation"
subtitle: "Production-Grade EdTech Platform with AI-Powered Learning"
author: "ensureStudy Engineering Team"
date: "February 2026"
documentclass: report
classoption:
  - a4paper
  - 11pt
geometry:
  - top=25mm
  - bottom=25mm
  - left=20mm
  - right=20mm
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: "blue"
urlcolor: "blue"
toccolor: "black"
header-includes:
  - |
    \\usepackage{fancyhdr}
    \\pagestyle{fancy}
    \\fancyhf{}
    \\fancyhead[L]{\\small ensureStudy Technical Documentation}
    \\fancyhead[R]{\\small \\leftmark}
    \\fancyfoot[C]{\\thepage}
    \\fancyfoot[R]{\\small February 2026}
    \\renewcommand{\\headrulewidth}{0.4pt}
    \\renewcommand{\\footrulewidth}{0.2pt}
  - |
    \\usepackage{titling}
    \\pretitle{\\begin{center}\\LARGE\\bfseries}
    \\posttitle{\\end{center}}
  - |
    \\usepackage{listings}
    \\lstset{
      basicstyle=\\ttfamily\\small,
      breaklines=true,
      frame=single,
      numbers=none,
      backgroundcolor=\\color[gray]{0.95},
      xleftmargin=2em,
      framexleftmargin=1.5em
    }
  - |
    \\usepackage{graphicx}
    \\usepackage{float}
    \\floatplacement{figure}{H}
  - |
    \\setlength{\\parskip}{0.5em}
    \\setlength{\\parindent}{0pt}
---

"""


def build_pdf():
    """Main build function."""
    print("=" * 60)
    print("ensureStudy Documentation PDF Builder")
    print("=" * 60)

    # Step 1: Get ordered files
    print("\n[1/4] Collecting documentation files...")
    files = get_ordered_files()
    print(f"  Found {len(files)} markdown files")

    # Step 2: Combine files
    print("\n[2/4] Combining and cleaning files...")
    metadata = create_metadata_yaml()
    combined_content = combine_files(files)

    # Write combined markdown
    with open(COMBINED_MD, "w", encoding="utf-8") as f:
        f.write(metadata)
        f.write(combined_content)
    print(f"  Combined file: {os.path.basename(COMBINED_MD)}")

    # Step 3: Build PDF with Pandoc
    print("\n[3/4] Building PDF with Pandoc + XeLaTeX + mermaid-filter...")
    print("  This may take several minutes due to Mermaid rendering...")

    pandoc_cmd = [
        "pandoc",
        COMBINED_MD,
        "-o",
        OUTPUT_FILE,
        "--pdf-engine=xelatex",
        "--filter=mermaid-filter",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--highlight-style=tango",
        "--variable=mainfont:Helvetica Neue",
        "--variable=monofont:Menlo",
        "--wrap=auto",
        "-V",
        "geometry:margin=25mm",
    ]

    try:
        result = subprocess.run(
            pandoc_cmd,
            capture_output=True,
            text=True,
            cwd=DOCS_DIR,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"\n  WARNING: Pandoc returned code {result.returncode}")
            if result.stderr:
                # Print only the last 30 lines of stderr to avoid overwhelming output
                stderr_lines = result.stderr.strip().split("\n")
                print("  Last errors:")
                for line in stderr_lines[-30:]:
                    print(f"    {line}")

            # If xelatex fails, try with a simpler engine
            if "xelatex" in result.stderr or result.returncode != 0:
                print("\n  Retrying without mermaid-filter (simpler build)...")
                pandoc_cmd_simple = [
                    "pandoc",
                    COMBINED_MD,
                    "-o",
                    OUTPUT_FILE,
                    "--pdf-engine=xelatex",
                    "--toc",
                    "--toc-depth=3",
                    "--number-sections",
                    "--highlight-style=tango",
                    "--wrap=auto",
                    "-V",
                    "geometry:margin=25mm",
                ]
                result2 = subprocess.run(
                    pandoc_cmd_simple,
                    capture_output=True,
                    text=True,
                    cwd=DOCS_DIR,
                    timeout=600,
                )
                if result2.returncode != 0:
                    print(f"  Simple build also failed: {result2.stderr[-500:]}")
                    return False
        print("  PDF generation complete!")

    except subprocess.TimeoutExpired:
        print("  ERROR: Build timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # Step 4: Verify output
    print("\n[4/4] Verifying output...")
    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"  Output: {os.path.basename(OUTPUT_FILE)}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Path: {OUTPUT_FILE}")
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL")
        print("=" * 60)
        return True
    else:
        print("  ERROR: Output file not created")
        return False


if __name__ == "__main__":
    success = build_pdf()
    sys.exit(0 if success else 1)
