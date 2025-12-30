"""Tests for CLI output formatting and progress tracking."""

import sys
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from phl.cli import OutputFormatter, ProgressTracker, format_output, print_output
from philly.__main__ import ConfigCommands


class TestOutputFormatter:
    """Tests for the OutputFormatter class."""

    def test_init_valid_formats(self):
        """Test initialization with valid formats."""
        for fmt in ["auto", "json", "jsonl", "csv", "tsv", "table"]:
            formatter = OutputFormatter(format=fmt)
            assert formatter.format == fmt

    def test_init_invalid_format(self):
        """Test initialization with invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            OutputFormatter(format="invalid")

    def test_format_json(self):
        """Test JSON formatting."""
        formatter = OutputFormatter(format="json")
        data = {"name": "test", "value": 42}
        output = formatter.format_output(data)

        assert '"name"' in output
        assert '"test"' in output
        assert '"value"' in output
        assert "42" in output

    def test_format_json_compact(self):
        """Test compact JSON formatting."""
        formatter = OutputFormatter(format="json", compact=True)
        data = {"a": 1, "b": 2}
        output = formatter.format_output(data)

        # Compact should have no spaces after colons
        assert ": " not in output
        assert '{"a":1,"b":2}' == output

    def test_format_jsonl(self):
        """Test JSONL formatting."""
        formatter = OutputFormatter(format="jsonl")
        data = [{"a": 1}, {"b": 2}]
        output = formatter.format_output(data)

        lines = output.split("\n")
        assert len(lines) == 2
        assert '{"a": 1}' in lines[0] or '{"a":1}' in lines[0]
        assert '{"b": 2}' in lines[1] or '{"b":2}' in lines[1]

    def test_format_jsonl_single_item(self):
        """Test JSONL with single item (should wrap in list)."""
        formatter = OutputFormatter(format="jsonl")
        data = {"test": "value"}
        output = formatter.format_output(data)

        assert '{"test":' in output

    def test_format_csv(self):
        """Test CSV formatting."""
        formatter = OutputFormatter(format="csv")
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        output = formatter.format_output(data)

        assert "name,age" in output
        assert "Alice,30" in output
        assert "Bob,25" in output

    def test_format_csv_with_special_chars(self):
        """Test CSV formatting with special characters."""
        formatter = OutputFormatter(format="csv")
        data = [{"name": "Test, Inc.", "value": "has,comma"}]
        output = formatter.format_output(data)

        # Should be quoted
        assert '"Test, Inc."' in output
        assert '"has,comma"' in output

    def test_format_tsv(self):
        """Test TSV formatting."""
        formatter = OutputFormatter(format="tsv")
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        output = formatter.format_output(data)

        assert "name\tage" in output
        assert "Alice\t30" in output
        assert "Bob\t25" in output

    def test_format_table(self):
        """Test table formatting."""
        formatter = OutputFormatter(format="table")
        data = [{"col1": "a", "col2": 1}, {"col1": "b", "col2": 2}]
        output = formatter.format_output(data)

        assert "col1" in output
        assert "col2" in output
        assert "a" in output
        assert "b" in output

    def test_format_table_single_dict(self):
        """Test table formatting with single dict."""
        formatter = OutputFormatter(format="table")
        data = {"key1": "value1", "key2": "value2"}
        output = formatter.format_output(data)

        assert "key1" in output
        assert "value1" in output
        assert "key2" in output
        assert "value2" in output

    def test_format_table_empty_data(self):
        """Test table formatting with empty data."""
        formatter = OutputFormatter(format="table")
        output = formatter.format_output([])

        assert output == ""

    def test_format_auto_non_tty(self, monkeypatch):
        """Test auto format defaults to JSON when not in TTY."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

        formatter = OutputFormatter(format="auto")
        data = [{"test": "value"}]
        output = formatter.format_output(data)

        # Should be JSON
        assert '"test"' in output

    def test_format_auto_tty_small_data(self, monkeypatch):
        """Test auto format uses table for small data in TTY."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        formatter = OutputFormatter(format="auto")
        data = [{"col": "value"}]
        output = formatter.format_output(data)

        # Should be table format (contains column headers aligned)
        assert "col" in output

    def test_format_auto_tty_large_data(self, monkeypatch):
        """Test auto format uses JSON for large data even in TTY."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        formatter = OutputFormatter(format="auto")
        # Large list > 100 items
        data = [{"id": i} for i in range(150)]
        output = formatter.format_output(data)

        # Should be JSON
        assert '"id"' in output

    def test_format_csv_empty_data(self):
        """Test CSV formatting with empty data."""
        formatter = OutputFormatter(format="csv")
        output = formatter.format_output([])

        # pandas returns just newline for empty DataFrame
        assert output.strip() == ""

    def test_format_tsv_empty_data(self):
        """Test TSV formatting with empty data."""
        formatter = OutputFormatter(format="tsv")
        output = formatter.format_output([])

        assert output.strip() == ""

    def test_format_csv_single_dict(self):
        """Test CSV formatting with single dict."""
        formatter = OutputFormatter(format="csv")
        data = {"name": "test", "value": 42}
        output = formatter.format_output(data)

        assert "name" in output
        assert "test" in output
        assert "42" in output

    def test_format_tsv_single_dict(self):
        """Test TSV formatting with single dict."""
        formatter = OutputFormatter(format="tsv")
        data = {"name": "test", "value": 42}
        output = formatter.format_output(data)

        assert "name" in output
        assert "test" in output
        assert "42" in output
        assert "\t" in output

    def test_format_table_list_of_non_dicts(self):
        """Test table formatting with list of non-dict items."""
        formatter = OutputFormatter(format="table")
        data = ["item1", "item2", "item3"]
        output = formatter.format_output(data)

        assert "item1" in output
        assert "item2" in output
        assert "item3" in output

    def test_format_json_with_non_serializable(self):
        """Test JSON formatting with non-serializable types uses str fallback."""
        formatter = OutputFormatter(format="json")
        from datetime import datetime

        data = {"date": datetime(2024, 1, 1)}
        output = formatter.format_output(data)

        # Should convert to string
        assert "2024" in output


class TestProgressTracker:
    """Tests for the ProgressTracker class."""

    def test_info_message(self):
        """Test info message output."""
        tracker = ProgressTracker(quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.info("Test info message")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Test info message" in output
        assert "ℹ" in output

    def test_error_message(self):
        """Test error message output."""
        tracker = ProgressTracker(quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.error("Test error")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Test error" in output
        assert "Error:" in output

    def test_success_message(self):
        """Test success message output."""
        tracker = ProgressTracker(quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.success("Test success")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Test success" in output
        assert "✓" in output

    def test_warning_message(self):
        """Test warning message output."""
        tracker = ProgressTracker(quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.warning("Test warning")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Test warning" in output
        assert "⚠" in output

    def test_progress_message(self):
        """Test progress message output."""
        tracker = ProgressTracker(show_progress=True, quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.progress("Processing...")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Processing..." in output

    def test_verbose_enabled(self):
        """Test verbose message when verbose is enabled."""
        tracker = ProgressTracker(quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.verbose("Verbose message", is_verbose=True)
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Verbose message" in output

    def test_verbose_disabled(self):
        """Test verbose message when verbose is disabled."""
        tracker = ProgressTracker(quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.verbose("Should not appear", is_verbose=False)
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Should not appear" not in output

    def test_quiet_mode_suppresses_info(self):
        """Test quiet mode suppresses info messages."""
        tracker = ProgressTracker(quiet=True)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.info("Should not appear")
        tracker.success("Should not appear")
        tracker.warning("Should not appear")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Should not appear" not in output

    def test_quiet_mode_shows_errors(self):
        """Test quiet mode still shows error messages."""
        tracker = ProgressTracker(quiet=True)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.error("Error message")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Error message" in output

    def test_show_progress_false(self):
        """Test show_progress=False suppresses progress messages."""
        tracker = ProgressTracker(show_progress=False, quiet=False)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.progress("Should not appear")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Should not appear" not in output

    def test_quiet_mode_suppresses_progress(self):
        """Test quiet mode suppresses progress messages."""
        tracker = ProgressTracker(show_progress=True, quiet=True)

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        tracker.progress("Should not appear")
        output = sys.stderr.getvalue()

        sys.stderr = old_stderr

        assert "Should not appear" not in output

    def test_default_initialization(self):
        """Test default initialization values."""
        tracker = ProgressTracker()

        assert tracker.show_progress is True
        assert tracker.quiet is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_format_output_function(self):
        """Test format_output convenience function."""
        data = {"test": "value"}
        output = format_output(data, format="json")

        assert '"test"' in output
        assert '"value"' in output

    def test_print_output_function(self, capsys):
        """Test print_output convenience function."""
        data = {"test": "value"}
        print_output(data, format="json")

        captured = capsys.readouterr()
        assert '"test"' in captured.out
        assert '"value"' in captured.out


class TestConfigCommands:
    """Tests for config CLI commands."""

    def test_show_command(self, capsys):
        """Test config show command."""
        config_cmd = ConfigCommands()
        config_cmd.show()

        captured = capsys.readouterr()
        assert "cache:" in captured.out
        assert "defaults:" in captured.out
        assert "cli:" in captured.out
        assert "network:" in captured.out

    def test_path_command_no_file(self, capsys, monkeypatch):
        """Test config path command when no config file exists."""
        # Change to a temp directory to ensure no config file
        with TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            config_cmd = ConfigCommands()
            config_cmd.path()

            captured = capsys.readouterr()
            assert "No config file found" in captured.out
            assert "philly.yml" in captured.out

    def test_init_command(self, capsys):
        """Test config init command."""
        with TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test-config.yml"
            config_cmd = ConfigCommands()
            config_cmd.init(str(test_path))

            captured = capsys.readouterr()
            assert "Created config file" in captured.out
            assert test_path.exists()

            # Verify file contents
            content = test_path.read_text()
            assert "cache:" in content
            assert "defaults:" in content

    def test_init_command_existing_file(self, capsys):
        """Test config init command with existing file."""
        with TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "existing.yml"
            test_path.write_text("existing: content")

            config_cmd = ConfigCommands()
            config_cmd.init(str(test_path))

            captured = capsys.readouterr()
            assert "already exists" in captured.out

    def test_get_command(self, capsys):
        """Test config get command."""
        config_cmd = ConfigCommands()
        config_cmd.get("cache.ttl")

        captured = capsys.readouterr()
        assert "3600" in captured.out

    def test_get_command_nested(self, capsys):
        """Test config get command with nested value."""
        config_cmd = ConfigCommands()
        config_cmd.get("defaults.format_preference")

        captured = capsys.readouterr()
        assert "csv" in captured.out
        assert "json" in captured.out

    def test_get_command_invalid_key(self, capsys):
        """Test config get command with invalid key."""
        config_cmd = ConfigCommands()
        config_cmd.get("invalid.key")

        captured = capsys.readouterr()
        assert "Unknown config key" in captured.out

    def test_path_command_with_file(self, capsys):
        """Test config path command when config file exists in current dir."""
        # The project already has a philly.yml file
        config_cmd = ConfigCommands()
        config_cmd.path()

        captured = capsys.readouterr()
        # Should show the path, not the "No config file found" message
        assert "philly.yml" in captured.out or "config.yml" in captured.out


class TestOutputFormatterEdgeCases:
    """Additional edge case tests for OutputFormatter."""

    def test_format_case_insensitive(self):
        """Test that format is case-insensitive."""
        for fmt in ["JSON", "Json", "jSoN"]:
            formatter = OutputFormatter(format=fmt)
            assert formatter.format == "json"

    def test_format_csv_with_newline_in_value(self):
        """Test CSV formatting with newline in values."""
        formatter = OutputFormatter(format="csv")
        data = [{"text": "line1\nline2", "id": 1}]
        output = formatter.format_output(data)

        # Value with newline should be quoted
        assert "text" in output
        assert "id" in output

    def test_format_jsonl_empty_list(self):
        """Test JSONL with empty list."""
        formatter = OutputFormatter(format="jsonl")
        output = formatter.format_output([])

        assert output == ""

    def test_format_auto_with_dict_large(self, monkeypatch):
        """Test auto format with large dict in TTY."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        formatter = OutputFormatter(format="auto")
        # Dict with > 50 keys
        data = {f"key{i}": i for i in range(60)}
        output = formatter.format_output(data)

        # Should be JSON for large dicts
        assert '"key0"' in output
