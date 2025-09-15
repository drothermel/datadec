"""
Custom Rich renderable components for consistent console formatting.

This module provides reusable Rich components that encapsulate styling and layout
patterns used throughout the DataDec console interface.
"""

from typing import Optional

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.rule import Rule


class TitlePanel:
    """
    A full-width title panel with consistent styling.

    Automatically includes:
    - Bold blue text and matching border
    - Full terminal width expansion
    - Newlines before and after for spacing
    """

    def __init__(self, title: str):
        self.title = title

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield ""  # Newline before
        yield Panel(
            f"[bold blue]{self.title}[/bold blue]",
            expand=True,
            border_style="bold blue"
        )
        yield ""  # Newline after


class SectionRule:
    """
    A styled rule for major section divisions.

    Provides consistent formatting for section breaks with:
    - Title case text with colored background
    - Bold blue rule styling
    - Newlines before and after for spacing
    """

    def __init__(self, title: str, style: str = "bold blue"):
        self.title = title.title()  # Convert to Title Case
        self.style = style
        # Create background color based on rule style
        if "blue" in style:
            self.title_style = "bold white on blue"
        elif "yellow" in style:
            self.title_style = "bold black on yellow"
        elif "green" in style:
            self.title_style = "bold white on green"
        elif "red" in style:
            self.title_style = "bold white on red"
        else:
            self.title_style = f"bold white on {style.split()[-1] if ' ' in style else style}"

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield ""  # Newline before
        yield Rule(f"[{self.title_style}] {self.title} [/{self.title_style}]", style=self.style)
        yield ""  # Newline after


class MetricPanel:
    """
    A fitted panel for containing metric analysis content.

    Creates a bordered container with:
    - Metric name as panel title
    - Fitted sizing (not full width)
    - Spacing before and after for separation
    """

    def __init__(self, metric_name: str, *content):
        self.metric_name = metric_name
        self.content = content

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Add newlines inside the panel content
        panel_content = Group("", *self.content, "")
        yield Panel.fit(panel_content, title=f"METRIC: {self.metric_name}")
        yield ""  # Spacing after


class InfoBlock:
    """
    Simple informational text block with optional styling.

    For displaying contextual information like data counts, formatting notes, etc.
    """

    def __init__(self, text: str, style: Optional[str] = None):
        self.text = text
        self.style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.style:
            yield f"[{self.style}]{self.text}[/{self.style}]"
        else:
            yield self.text