
from pathlib import Path

SITE_DIR = Path("_site")
CONFIG_FILE = Path(
    "extra/javascript/maths-admonitions.config.js"
)

CDN_URL = (
    "https://cdn.jsdelivr.net/npm/"
    "@rod2ik/mkdocs-maths-admonitions@0.2.0/"
    "dist/mkdocs-maths-admonitions.min.js"
)

MARKER = "<!-- ste131-maths-admonitions -->"


def enable_math_admonitions():
    if not SITE_DIR.is_dir():
        raise FileNotFoundError(
            "Build the book before running this script."
        )

    config = CONFIG_FILE.read_text(encoding="utf-8")

    injection = f"""
{MARKER}
<script>
{config}
</script>
<script defer src="{CDN_URL}"></script>
"""

    modified = 0

    for html_file in SITE_DIR.rglob("*.html"):
        html = html_file.read_text(encoding="utf-8")

        if MARKER in html:
            continue

        if "</body>" not in html:
            continue

        html = html.replace(
            "</body>",
            injection + "\n</body>",
            1,
        )

        html_file.write_text(
            html,
            encoding="utf-8",
        )

        modified += 1

    print(
        f"Maths admonitions enabled on {modified} pages."
    )


if __name__ == "__main__":
    enable_math_admonitions()
