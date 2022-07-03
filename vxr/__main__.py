"""CLI commands."""

import typer

app = typer.Typer(
    name='vxr',
    add_completion=False,
    help='Generate chest X-ray reports using Vision Transformers',
)


if __name__ == '__main__':
    app()
