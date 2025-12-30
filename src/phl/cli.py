"""CLI output formatting and progress tracking utilities."""

import json
import sys
from typing import Any
from io import StringIO


class OutputFormatter:
    """Format data for CLI output in various formats.

    Supports: json, jsonl, csv, tsv, table, and auto (intelligent selection).
    """

    def __init__(self, format: str = "auto", compact: bool = False):
        """Initialize the output formatter.

        Args:
            format: Output format. One of: auto, json, jsonl, csv, tsv, table
            compact: If True, use compact formatting where applicable
        """
        self.format: str = format.lower()
        self.compact: bool = compact

        # Validate format
        valid_formats = {"auto", "json", "jsonl", "csv", "tsv", "table"}
        if self.format not in valid_formats:
            raise ValueError(
                f"Invalid format '{self.format}'. "
                "Must be one of: " + ", ".join(valid_formats)
            )

    def format_output(self, data: Any) -> str:
        """Route to appropriate formatter based on format setting.

        Args:
            data: Data to format. Can be dict, list, DataFrame, or other types.

        Returns:
            Formatted string ready for output
        """
        formatters = {
            "json": self._format_json,
            "jsonl": self._format_jsonl,
            "csv": self._format_csv,
            "tsv": self._format_tsv,
            "table": self._format_table,
            "auto": self._format_auto,
        }

        formatter = formatters[self.format]
        return formatter(data)

    def _format_json(self, data: Any) -> str:
        """Format data as pretty JSON.

        Args:
            data: Data to format

        Returns:
            JSON string with indent=2 (or compact if self.compact=True)
        """
        if self.compact:
            return json.dumps(data, default=str, separators=(",", ":"))
        return json.dumps(data, indent=2, default=str)

    def _format_jsonl(self, data: Any) -> str:
        """Format data as JSON Lines (one JSON object per line).

        Args:
            data: Data to format. If list, each item becomes a line.

        Returns:
            JSONL string
        """
        # If data is not a list, wrap it
        if not isinstance(data, list):
            data = [data]

        lines = []
        for item in data:
            lines.append(json.dumps(item, default=str))

        return "\n".join(lines)

    def _format_csv(self, data: Any) -> str:
        """Format data as CSV.

        Args:
            data: Data to format. Works best with DataFrame or list of dicts.

        Returns:
            CSV string
        """
        try:
            import pandas as pd

            # Convert to DataFrame if not already
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Single dict - convert to single-row DataFrame
                df = pd.DataFrame([data])
            else:
                # Try to convert directly
                df = pd.DataFrame(data)

            # Use StringIO to capture CSV output
            output = StringIO()
            df.to_csv(output, index=False)
            return output.getvalue()

        except ImportError:
            # pandas not available - provide basic CSV formatting
            return self._format_csv_basic(data)

    def _format_csv_basic(self, data: Any) -> str:
        """Basic CSV formatting without pandas.

        Args:
            data: Data to format (list of dicts expected)

        Returns:
            Basic CSV string
        """
        if not data:
            return ""

        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return str(data)

        # Get headers from first item
        if not isinstance(data[0], dict):
            return "\n".join(str(item) for item in data)

        headers = list(data[0].keys())
        lines = [",".join(headers)]

        for item in data:
            values = [str(item.get(h, "")) for h in headers]
            # Basic CSV escaping - wrap in quotes if contains comma
            escaped = [f'"{v}"' if "," in v or '"' in v else v for v in values]
            lines.append(",".join(escaped))

        return "\n".join(lines)

    def _format_tsv(self, data: Any) -> str:
        """Format data as TSV (Tab-Separated Values).

        Args:
            data: Data to format

        Returns:
            TSV string
        """
        try:
            import pandas as pd

            # Convert to DataFrame if not already
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)

            # Use StringIO to capture TSV output
            output = StringIO()
            df.to_csv(output, sep="\t", index=False)
            return output.getvalue()

        except ImportError:
            # pandas not available - provide basic TSV formatting
            return self._format_tsv_basic(data)

    def _format_tsv_basic(self, data: Any) -> str:
        """Basic TSV formatting without pandas.

        Args:
            data: Data to format (list of dicts expected)

        Returns:
            Basic TSV string
        """
        if not data:
            return ""

        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return str(data)

        # Get headers from first item
        if not isinstance(data[0], dict):
            return "\n".join(str(item) for item in data)

        headers = list(data[0].keys())
        lines = ["\t".join(headers)]

        for item in data:
            values = [str(item.get(h, "")) for h in headers]
            lines.append("\t".join(values))

        return "\n".join(lines)

    def _format_table(self, data: Any) -> str:
        """Format data as a pretty table.

        Tries to use tabulate library if available, otherwise falls back to
        simple column alignment.

        Args:
            data: Data to format

        Returns:
            Table string
        """
        try:
            from tabulate import tabulate  # pyright: ignore[reportMissingModuleSource]

            # Convert various data types to table format
            if hasattr(data, "to_dict"):  # DataFrame
                return tabulate(data, headers="keys", tablefmt="simple")
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                return tabulate(data, headers="keys", tablefmt="simple")
            elif isinstance(data, dict):
                # Single dict - show as key-value table
                table_data = [[k, v] for k, v in data.items()]
                return tabulate(table_data, headers=["Key", "Value"], tablefmt="simple")
            else:
                return tabulate(data, tablefmt="simple")

        except ImportError:
            # tabulate not available - use simple formatting
            return self._format_table_simple(data)

    def _format_table_simple(self, data: Any) -> str:
        """Simple table formatting without tabulate library.

        Args:
            data: Data to format

        Returns:
            Simple aligned table string
        """
        if not data:
            return ""

        # Convert DataFrame to list of dicts
        if hasattr(data, "to_dict"):
            data = data.to_dict("records")

        # Handle single dict
        if isinstance(data, dict):
            # Show as key-value pairs
            max_key_len = max(len(str(k)) for k in data.keys())
            lines = []
            lines.append(f"{'Key':<{max_key_len}}  Value")
            lines.append("-" * (max_key_len + 2 + 20))
            for k, v in data.items():
                lines.append(f"{str(k):<{max_key_len}}  {v}")
            return "\n".join(lines)

        # Handle list of dicts
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                headers = list(data[0].keys())

                # Calculate column widths
                col_widths = {h: len(str(h)) for h in headers}
                for item in data:
                    for h in headers:
                        col_widths[h] = max(col_widths[h], len(str(item.get(h, ""))))

                # Build table
                lines = []

                # Header row
                header_row = "  ".join(str(h).ljust(col_widths[h]) for h in headers)
                lines.append(header_row)

                # Separator
                separator = "  ".join("-" * col_widths[h] for h in headers)
                lines.append(separator)

                # Data rows
                for item in data:
                    row = "  ".join(
                        str(item.get(h, "")).ljust(col_widths[h]) for h in headers
                    )
                    lines.append(row)

                return "\n".join(lines)
            else:
                # List of non-dict items
                return "\n".join(str(item) for item in data)

        # Fallback
        return str(data)

    def _format_auto(self, data: Any) -> str:
        """Automatically select best format based on context.

        Logic:
        - If in a terminal AND data is small: use table format
        - Otherwise: use JSON format

        Args:
            data: Data to format

        Returns:
            Formatted string using auto-selected format
        """
        is_terminal = sys.stdout.isatty()

        # Determine if data is "small" (suitable for table display)
        is_small = False

        if isinstance(data, list):
            is_small = len(data) <= 100
        elif isinstance(data, dict):
            is_small = len(data) <= 50
        elif hasattr(data, "__len__"):
            try:
                is_small = len(data) <= 100
            except TypeError:
                is_small = False

        # Use table for small data in terminal, otherwise JSON
        if is_terminal and is_small:
            return self._format_table(data)
        else:
            return self._format_json(data)


class ProgressTracker:
    """Track and display progress and status messages for CLI operations.

    Messages are sent to stderr to keep stdout clean for data output.
    """

    def __init__(self, show_progress: bool = True, quiet: bool = False):
        """Initialize the progress tracker.

        Args:
            show_progress: If True, show progress messages
            quiet: If True, suppress all non-error output
        """
        self.show_progress: bool = show_progress
        self.quiet: bool = quiet

    def info(self, message: str) -> None:
        """Print an info message to stderr.

        Args:
            message: Message to print
        """
        if not self.quiet:
            print(f"ℹ {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Print an error message to stderr.

        Always printed, even in quiet mode.

        Args:
            message: Error message to print
        """
        print(f"✗ Error: {message}", file=sys.stderr)

    def success(self, message: str) -> None:
        """Print a success message to stderr.

        Args:
            message: Success message to print
        """
        if not self.quiet:
            print(f"✓ {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        """Print a warning message to stderr.

        Args:
            message: Warning message to print
        """
        if not self.quiet:
            print(f"⚠ {message}", file=sys.stderr)

    def verbose(self, message: str, is_verbose: bool = False) -> None:
        """Print a message only if verbose mode is enabled.

        Args:
            message: Message to print
            is_verbose: Whether verbose mode is enabled
        """
        if is_verbose and not self.quiet:
            print(f"  {message}", file=sys.stderr)

    def progress(self, message: str) -> None:
        """Print a progress message to stderr.

        Only shown if show_progress is True and not in quiet mode.

        Args:
            message: Progress message to print
        """
        if self.show_progress and not self.quiet:
            print(f"⋯ {message}", file=sys.stderr)


# Convenience functions for one-off formatting
def format_output(data: Any, format: str = "auto", compact: bool = False) -> str:
    """Convenience function to format output without creating a formatter instance.

    Args:
        data: Data to format
        format: Output format (auto, json, jsonl, csv, tsv, table)
        compact: Use compact formatting if applicable

    Returns:
        Formatted string
    """
    formatter = OutputFormatter(format=format, compact=compact)
    return formatter.format_output(data)


def print_output(data: Any, format: str = "auto", compact: bool = False) -> None:
    """Format and print data to stdout.

    Args:
        data: Data to format and print
        format: Output format (auto, json, jsonl, csv, tsv, table)
        compact: Use compact formatting if applicable
    """
    output = format_output(data, format=format, compact=compact)
    print(output)
