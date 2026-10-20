
from pathlib import Path
import html
import json
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SITE_DIR = PROJECT_ROOT / "_site"

CHAPTER_NUMBERS_FILE = (
    PROJECT_ROOT / "chapter_numbers.yml"
)

with CHAPTER_NUMBERS_FILE.open(
    "r", encoding="utf-8"
) as file:
    configuration = yaml.safe_load(file)

CHAPTER_NUMBERS = configuration["chapters"]

CONFIG_FILE = (
    PROJECT_ROOT
    / "extra"
    / "javascript"
    / "maths-admonitions.config.js"
)

CDN_URL = (
    "https://cdn.jsdelivr.net/npm/"
    "@rod2ik/mkdocs-maths-admonitions@0.2.0/"
    "dist/mkdocs-maths-admonitions.min.js"
)

MARKER_START = "<!-- ste131-maths-admonitions:start -->"
MARKER_END = "<!-- ste131-maths-admonitions:end -->"


def get_chapter_number(html_file):
    """Determine the chapter number from the generated HTML path."""

    # marimo-book uses chapter title directories, for example:
    # _site/Linear Optimization Part I/index.html

    page_name = html_file.parent.name

    for filename, number in CHAPTER_NUMBERS.items():
        if Path(filename).stem == page_name:
            return number

    return None


def numbering_script(chapter_number):
    """JavaScript to add chapter prefixes to generated numbers."""

    chapter = json.dumps(chapter_number)

    return f"""
<script>
(function () {{
    const chapterNumber = {chapter};

    function updateNumbers() {{
        // Ensure the extension has processed the page.
        window.MkDocsMathsAdmonitions?.init?.(document);

        const statements = document.querySelectorAll(
            '[data-mkdocs-maths-admonitions-number]'
        );

        for (const statement of statements) {{
            if (statement.dataset.ste131ChapterNumbered === "true") {{
                continue;
            }}

            const localNumber = statement.getAttribute(
                "data-mkdocs-maths-admonitions-number"
            );

            const title = statement.querySelector(
                ".admonition-title, .summary"
            );

            if (!title || !localNumber) {{
                continue;
            }}

            // Locate the generated integer in the title.
            const walker = document.createTreeWalker(
                title,
                NodeFilter.SHOW_TEXT
            );

            let node;
            let replaced = false;

            while ((node = walker.nextNode())) {{
                const pattern = new RegExp(
                    "(^|\\\\s)(" + localNumber + ")(?=[.\\\\s:—-])"
                );

                if (pattern.test(node.textContent)) {{
                    node.textContent = node.textContent.replace(
                        pattern,
                        "$1" + chapterNumber + "." + localNumber
                    );

                    replaced = true;
                    break;
                }}
            }}

            if (replaced) {{
                statement.dataset.ste131ChapterNumbered = "true";
                statement.setAttribute(
                    "data-ste131-number",
                    chapterNumber + "." + localNumber
                );
            }}
        }}
    }}

    window.addEventListener("load", updateNumbers);
}})();
</script>
"""


def enable_math_admonitions():

    if not SITE_DIR.is_dir():
        raise FileNotFoundError(
            "Build the book before running this script."
        )

    config = CONFIG_FILE.read_text(encoding="utf-8")
    modified = 0

    for html_file in SITE_DIR.rglob("*.html"):

        content = html_file.read_text(encoding="utf-8")

        # Remove previous injection, making rebuilds repeatable.
        if MARKER_START in content and MARKER_END in content:
            before, remainder = content.split(
                MARKER_START, 1
            )
            _, after = remainder.split(MARKER_END, 1)
            content = before + after

        if "</body>" not in content:
            continue

        chapter_number = get_chapter_number(html_file)

        chapter_script = (
            numbering_script(chapter_number)
            if chapter_number is not None
            else ""
        )

        injection = f"""
{MARKER_START}
<script>
{config}
</script>
<script defer src="{html.escape(CDN_URL, quote=True)}"></script>
{chapter_script}
{MARKER_END}
"""

        content = content.replace(
            "</body>",
            injection + "\\n</body>",
            1,
        )

        html_file.write_text(
            content,
            encoding="utf-8",
        )

        modified += 1

        if chapter_number is not None:
            print(
                f"Chapter {chapter_number}: "
                f"{html_file.relative_to(SITE_DIR)}"
            )

    print(f"Processed {modified} HTML pages.")


if __name__ == "__main__":
    enable_math_admonitions()
