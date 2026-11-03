"""Copied command previews preserve arguments instead of expanding shell text."""

import json
import shlex
import shutil
import subprocess
import sys

import pytest

from cellstine.cli.interactive import prompts


def test_posix_preview_roundtrips_special_arguments(monkeypatch):
    monkeypatch.setattr(prompts.sys, "platform", "linux")
    arguments = ["view", "$(1+2).vasp", "`n", 'a"b', "O'Brien", "[111]", "a;b", "", "a\\b c", "--%"]
    assert shlex.split(prompts._format_command(arguments)) == ["cellstine", *arguments]


def test_powershell_preview_uses_literal_quotes(monkeypatch):
    monkeypatch.setattr(prompts.sys, "platform", "win32")
    assert prompts._format_command(["view", "$(1+2).vasp", 'a"b', "O'Brien", "`n", ""]) == (
        "cellstine view '$(1+2).vasp' 'a\"b' 'O''Brien' '`n' ''"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell host argument roundtrip")
def test_powershell_preview_passes_literal_arguments_to_python(tmp_path):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell not installed")
    script = tmp_path / "print_args.py"
    script.write_text("import json,sys;print(json.dumps(sys.argv[1:]))", encoding="utf-8")
    arguments = ["$(1+2).vasp", "`n", 'a"b', "O'Brien", "[111]", "a;b", "a\\b c", "@args"]
    command = prompts._format_command([sys.executable, str(script), *arguments])
    # Call the quoted executable explicitly. 'cellstine ' is only the launcher
    # prefix; every supplied argument uses the same production quoting.
    command = "& " + command.removeprefix("cellstine ")
    completed = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                               check=True, capture_output=True, text=True, encoding="utf-8")
    assert json.loads(completed.stdout) == arguments
