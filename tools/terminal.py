"""Controlled terminal execution.

The agent needs to run real programs (`python -m pytest`, `npm test`,
`git status`) and read real output, but this assistant drives a real
computer, so execution is risk-classified rather than unrestricted:

- Every command is parsed and its executable checked against
  `SAFE_EXECUTABLES`; anything else is `blocked_executable`.
- Patterns that destroy data or state (`rm -rf`, `format`, `shutdown`,
  `git reset --hard`, ...) are `destructive` and refused unless the
  caller explicitly passes `approved=True`, which only a human-confirmed
  path ever does.
- Shell metacharacters (`|`, `&&`, `;`, backticks, redirects) are
  rejected in the STRING form, because this module has to split that
  itself and they would let a safe-looking executable smuggle an
  unchecked second command past the allowlist. The list form goes
  straight to `subprocess.run` with no shell, so the same characters are
  ordinary argument data there and are not rejected.
- Every run has a timeout; a hanging process is killed, not waited on.
- stdout/stderr are captured, secret-redacted, and length-bounded.

`JARVIS_TERMINAL_UNRESTRICTED=1` lifts the executable allowlist for a
machine where the user genuinely wants that. It does NOT lift the
destructive-pattern check or the shell-metacharacter check -- a
destructive command still requires explicit approval.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence

from config import get_config
from memory.memory_manager import redact
from tools.windows_process import hidden_process_kwargs

log = logging.getLogger("jarvis.terminal")

# Executables an agent may run without human approval. Interpreters and
# build/test tooling: they read and run project code, which is the point,
# but none of them is itself a system-modifying command.
SAFE_EXECUTABLES = {
    "python", "python3", "py", "pip", "pytest", "tox", "mypy", "ruff", "flake8", "black",
    "node", "npm", "npx", "yarn", "pnpm", "deno", "bun",
    "git", "cargo", "go", "dotnet", "java", "javac", "mvn", "gradle", "make",
    "echo", "type", "cat", "ls", "dir", "where", "which", "findstr", "grep", "head", "tail", "wc",
}

# Refused unless explicitly approved, even for an allowlisted executable.
DESTRUCTIVE_PATTERN = re.compile(
    r"(?:\brm\s+-[a-z]*[rf]|\bdel\s+/[sq]|\brmdir\s+/s|\bformat\b|\bdiskpart\b|"
    r"\bshutdown\b|\brestart-computer\b|\breboot\b|\bmkfs\b|"
    r"\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[a-z]*f|\bgit\s+push\b|"
    r"\breg(?:\.exe)?\s+(?:add|delete)\b|\btaskkill\b|\bremove-item\b|"
    r"\bpip\s+uninstall\b|\bnpm\s+uninstall\b|:\s*\(\s*\)\s*\{)",
    re.IGNORECASE,
)

# A safe executable plus one of these becomes an arbitrary second command.
SHELL_METACHARACTERS = re.compile(r"[|&;<>`$]|\|\||&&")


def _executable_name(argument: str) -> str:
    name = Path(argument).name.casefold()
    return name[:-4] if name.endswith(".exe") else name


def classify_command(command: str | Sequence[str]) -> dict:
    """Decide whether a command may run, WITHOUT running it.

    Returns `{"allowed": bool, "risk": "safe"|"destructive"|"blocked",
    "reason": str|None, "argv": [...]}`. Exposed separately from
    `run_command` so a planner/skill can check a command before proposing
    it, and so the decision is unit-testable on its own.
    """
    raw = command if isinstance(command, str) else " ".join(command)
    if isinstance(command, str):
        if SHELL_METACHARACTERS.search(command):
            return {"allowed": False, "risk": "blocked", "reason": "shell_metacharacters", "argv": [], "command": raw}
        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError:
            return {"allowed": False, "risk": "blocked", "reason": "unparsable_command", "argv": [], "command": raw}
    else:
        # A list is passed straight to `subprocess.run` with no shell, so a
        # metacharacter inside one argument is ordinary data (e.g.
        # `python -c "a;b"`) and is NOT a smuggled second command. Only the
        # string form, which this module has to split itself, needs that check.
        argv = [str(part) for part in command]

    argv = [part.strip('"') for part in argv if part != ""]
    if not argv:
        return {"allowed": False, "risk": "blocked", "reason": "empty_command", "argv": [], "command": raw}

    if DESTRUCTIVE_PATTERN.search(raw):
        return {"allowed": False, "risk": "destructive", "reason": "destructive_command", "argv": argv, "command": raw}

    executable = _executable_name(argv[0])
    if not get_config().terminal_allow_unrestricted and executable not in SAFE_EXECUTABLES:
        return {"allowed": False, "risk": "blocked", "reason": f"blocked_executable:{executable}", "argv": argv, "command": raw}

    return {"allowed": True, "risk": "safe", "reason": None, "argv": argv, "command": raw}


def run_command(
    command: str | Sequence[str],
    working_directory: str | None = None,
    timeout: float | None = None,
    approved: bool = False,
    env: dict[str, str] | None = None,
) -> dict:
    """Run one command and return a structured result.

    Never raises for an ordinary failure: a non-zero exit code, a
    timeout, and a missing executable all come back as a result dict, so
    the agent can read the error and react instead of the whole task
    dying.
    """
    config = get_config()
    timeout = float(timeout or config.terminal_timeout)
    decision = classify_command(command)
    if not decision["allowed"] and not (approved and decision["risk"] == "destructive"):
        return {
            "success": False,
            "verified": True,
            "message": f"I did not run that command: {decision['reason']}.",
            "error": decision["reason"],
            "risk": decision["risk"],
            "requires_approval": decision["risk"] == "destructive",
            "command": redact(str(decision["command"])),
            "exit_code": None,
            "timed_out": False,
        }

    argv = decision["argv"]
    cwd = Path(working_directory).expanduser().resolve() if working_directory else Path.cwd()
    if not cwd.is_dir():
        return {
            "success": False,
            "verified": True,
            "message": "The working directory does not exist.",
            "error": "working_directory_not_found",
            "working_directory": str(cwd),
            "exit_code": None,
            "timed_out": False,
        }

    environment = {**os.environ, **(env or {})}
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=environment,
            **hidden_process_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "verified": True,
            "message": f"The command timed out after {timeout:.0f}s and was terminated.",
            "error": "timeout",
            "timed_out": True,
            "exit_code": None,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "command": redact(" ".join(argv)),
            "working_directory": str(cwd),
            "stdout": _bound(_decode(exc.stdout)),
            "stderr": _bound(_decode(exc.stderr)),
        }
    except FileNotFoundError:
        return {
            "success": False,
            "verified": True,
            "message": f"The executable {argv[0]!r} was not found.",
            "error": "executable_not_found",
            "exit_code": None,
            "timed_out": False,
            "command": redact(" ".join(argv)),
            "working_directory": str(cwd),
        }
    except OSError as exc:
        return {
            "success": False,
            "message": "The command could not be started.",
            "error": f"{type(exc).__name__}: {exc}",
            "exit_code": None,
            "timed_out": False,
            "command": redact(" ".join(argv)),
        }

    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    stdout = _bound(redact(completed.stdout or ""))
    stderr = _bound(redact(completed.stderr or ""))
    succeeded = completed.returncode == 0
    summary = (stdout or stderr).strip().splitlines()
    return {
        "success": succeeded,
        # The exit code IS independent evidence of what happened, so a
        # completed run is genuinely verified either way.
        "verified": True,
        "message": (
            f"Command finished with exit code {completed.returncode}."
            + (f" {summary[-1][:200]}" if summary else "")
        ),
        "error": None if succeeded else f"exit_code_{completed.returncode}",
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
        "timed_out": False,
        "command": redact(" ".join(argv)),
        "working_directory": str(cwd),
    }


def _decode(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _bound(text: str) -> str:
    limit = get_config().terminal_max_output_chars
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n... [{omitted} characters omitted] ...\n{text[-half:]}"
