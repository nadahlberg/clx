import click

from .manage import manage


@click.group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(manage)
