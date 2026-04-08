import click

from .autopilot import autopilot
from .manage import manage


@click.group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(manage)
cli.add_command(autopilot)
