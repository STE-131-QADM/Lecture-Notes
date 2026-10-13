import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    print("Hello, world!")
    return


if __name__ == "__main__":
    app.run()
