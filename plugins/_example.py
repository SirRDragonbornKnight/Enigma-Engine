"""
Example Plugin — rename to ``example.py`` to activate.

Demonstrates how to add custom [CMD] commands to Enigma Engine.
"""

from enigma_engine.core.commands import CommandResult


def register(registry):
    """Called automatically when this plugin is loaded."""

    def greet(args, ctx):
        """Say hello to someone."""
        name = args[0] if args else "world"
        return CommandResult(True, f"[OK] Hello, {name}!")

    def roll_dice(args, ctx):
        """Roll a random number between 1 and N (default 6)."""
        import random

        sides = int(args[0]) if args else 6
        result = random.randint(1, sides)
        return CommandResult(True, f"[OK] Rolled a {result} (d{sides})")

    registry.register(
        "example.greet",
        greet,
        description="Greet someone by name",
        usage="example.greet [name]",
    )
    registry.register(
        "example.roll",
        roll_dice,
        description="Roll a dice",
        usage="example.roll [sides]",
    )
