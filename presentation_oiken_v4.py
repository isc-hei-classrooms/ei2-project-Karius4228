import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    Get-Content "presentation_oiken_v4.py" | Measure-Object -Line
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
