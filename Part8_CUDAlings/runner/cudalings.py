#!/usr/bin/env python3
"""
CUDAlings -- a rustlings-style exercise runner for the CUDA course.

Goal:
    Walk you through ~100+ tiny exercises that build CUDA + Python-binding
    muscle memory. You edit one file, save, and the runner tells you instantly
    whether your kernel compiles, runs, and produces the expected output.

How an exercise is identified:
    Each exercise lives in `exercises/<chapter>/<NN>_name.cu` (or .py / .cpp).
    The first time you look at it you'll see a marker line near the top:

        // I AM NOT DONE

    (or `# I AM NOT DONE` for Python). The runner SKIPS exercises that still
    have this marker, treating them as "not yet attempted". When you've taken
    a stab at the problem, you delete that line. The runner will then try to
    compile and validate your work.

Validation strategy:
    Each exercise has a sibling `expected.txt` describing the validator:

        mode: stdout_exact      # compare program stdout literally to a string
        mode: stdout_contains   # check stdout contains all the listed lines
        mode: pytest            # run pytest against a sibling test file
        mode: numeric           # parse a single float and compare with tolerance

    See `_validators` below for the full list. This keeps the framework
    extensible -- a future exercise that tests a Triton kernel only needs a
    new mode entry.

Commands:
    cudalings list                  # show all exercises and status
    cudalings run <NAME-or-NUMBER>  # validate one exercise
    cudalings next                  # validate the first unfinished exercise
    cudalings watch                 # rerun `next` whenever you save a file
    cudalings hint <NAME>           # print the hint for that exercise
    cudalings solution <NAME>       # print the reference solution (use sparingly!)
    cudalings reset <NAME>          # restore the original "I AM NOT DONE" stub

Why we don't ship a Cargo-style binary like rustlings does:
    Rustlings depends on Cargo's incremental build cache. We're orchestrating
    nvcc + (later) pybind11 + pytorch extensions, which already do their own
    caching. A 200-line Python runner is plenty -- and it stays hackable so
    you can tweak the validation rules as you build new chapters.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
RUNNER_DIR = Path(__file__).resolve().parent
ROOT = RUNNER_DIR.parent                          # Part8_CUDAlings/
EXERCISES_DIR = ROOT / "exercises"
SOLUTIONS_DIR = ROOT / "solutions"

# Markers that indicate "user hasn't started this yet". We support a few
# language-specific spellings so the marker reads naturally in each file.
NOT_DONE_MARKERS = (
    "// I AM NOT DONE",
    "# I AM NOT DONE",
    "/* I AM NOT DONE */",
)

# nvcc target -- matches the rest of the course (Pascal, Quadro P4200).
# If you're on a newer GPU you can override via CUDALINGS_ARCH=sm_86 etc.
DEFAULT_NVCC_ARCH = os.environ.get("CUDALINGS_ARCH", "sm_61")
NVCC = os.environ.get("NVCC", "nvcc")
PYTHON = sys.executable

# Many distros ship gcc-13 by default, which CUDA 11.x's nvcc doesn't accept.
# If `g++-11` (or older) exists we point nvcc at it; otherwise we let nvcc
# pick the system default. We prefer g++ over gcc because nvcc-with-gcc
# won't auto-link libstdc++ at the final link step, breaking any exercise
# that uses `new`/`delete`/std::vector. Override with CUDALINGS_HOST_CC.
def _detect_host_cc() -> str | None:
    if "CUDALINGS_HOST_CC" in os.environ:
        return os.environ["CUDALINGS_HOST_CC"]
    from shutil import which
    for cand in ("g++-11", "g++-10", "g++-9"):
        if which(cand):
            return which(cand)
    return None
NVCC_HOST_CC = _detect_host_cc()

# ANSI colors -- terminal-only, disabled if NO_COLOR is set or stdout isn't a tty.
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(s: str) -> str:  return _c("32", s)
def red(s: str) -> str:    return _c("31", s)
def yellow(s: str) -> str: return _c("33", s)
def blue(s: str) -> str:   return _c("34", s)
def gray(s: str) -> str:   return _c("90", s)
def bold(s: str) -> str:   return _c("1",  s)


# ---------------------------------------------------------------------------
# Exercise discovery + metadata
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Exercise:
    """One exercise on disk.

    `path` is the source file the student edits. `expected` is the sibling
    file describing how to validate it. `chapter` is the parent directory
    name (e.g. `01_hello_gpu`) and `name` is the source stem.
    """
    path: Path
    expected: Path
    chapter: str
    name: str
    order: int          # global ordering -- (chapter index, file index)

    @property
    def display(self) -> str:
        return f"{self.chapter}/{self.name}"

    @property
    def language(self) -> str:
        return {".cu": "cuda", ".cpp": "cpp", ".py": "python"}.get(
            self.path.suffix, "unknown"
        )

    def has_not_done_marker(self) -> bool:
        """Return True if the file still has the `I AM NOT DONE` line.

        We read the whole file (these are tiny) and search line by line.
        Comparing per-line is more robust than `in text`, since a stray
        match inside a docstring shouldn't count.
        """
        if not self.path.exists():
            return False
        for line in self.path.read_text().splitlines():
            if line.strip() in NOT_DONE_MARKERS:
                return True
        return False


def discover() -> list[Exercise]:
    """Walk `exercises/` and build the ordered exercise list.

    Order matters for `next` and `watch`. We sort first by chapter directory
    name (which starts with a number prefix like `04_`) and then by file name
    (also number-prefixed). Globbing alone wouldn't guarantee that order.
    """
    if not EXERCISES_DIR.exists():
        return []

    items: list[Exercise] = []
    for chapter_dir in sorted(EXERCISES_DIR.iterdir()):
        if not chapter_dir.is_dir():
            continue
        # Pick up .cu, .cpp, .py -- the runner is language-agnostic.
        sources = sorted(
            p for p in chapter_dir.iterdir()
            if p.suffix in {".cu", ".cpp", ".py"}
            and not p.name.startswith("_")
            and p.stem != "test"      # `test_*.py` is a sibling validator, not the exercise
            and not p.stem.startswith("test_")
        )
        for src in sources:
            expected = src.with_suffix(".expected.txt")
            items.append(Exercise(
                path=src,
                expected=expected,
                chapter=chapter_dir.name,
                name=src.stem,
                order=len(items),
            ))
    return items


def find(items: Iterable[Exercise], query: str) -> Exercise | None:
    """Resolve a `run <query>` argument.

    Accepts: full path stem, just the file stem, the global index, or any
    unique substring. Falling back to substring is the convenience that
    rustlings users rely on (`rustlings run varia` matches `variables1`).
    """
    items = list(items)

    # 1. global index ("run 14")
    if query.isdigit():
        idx = int(query)
        for e in items:
            if e.order == idx:
                return e

    # 2. exact display ("run 03_thread_hierarchy/02_index_2d")
    for e in items:
        if e.display == query or e.name == query:
            return e

    # 3. unique substring
    matches = [e for e in items if query in e.display or query in e.name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(red(f"Ambiguous query '{query}'. Candidates:"))
        for m in matches:
            print(f"  {m.display}")
        sys.exit(2)
    return None


# ---------------------------------------------------------------------------
# Building + running
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    rc: int
    duration_s: float


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> RunResult:
    """Run a subprocess and capture its result.

    We use a hard timeout because a buggy CUDA kernel can hang the process
    (infinite loop on the device, missed cudaDeviceSynchronize). 60s is
    generous for these small exercises.
    """
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=cwd,
            capture_output=True, text=True,
            timeout=timeout,
        )
        return RunResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            rc=proc.returncode,
            duration_s=time.monotonic() - t0,
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(
            ok=False,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + f"\n[cudalings] timed out after {timeout}s",
            rc=124,
            duration_s=time.monotonic() - t0,
        )
    except FileNotFoundError as e:
        return RunResult(False, "", f"[cudalings] command not found: {e}", 127, 0.0)


def build(ex: Exercise) -> tuple[bool, str, Path | None]:
    """Compile (or otherwise prepare) the exercise.

    For CUDA/C++ we shell out to nvcc. For Python we just check syntax with
    `python -m py_compile` -- there's no compile step proper. The caller gets
    back a path to the produced binary (or None for Python).
    """
    if ex.language == "python":
        r = _run([PYTHON, "-m", "py_compile", str(ex.path)], ex.path.parent)
        return r.ok, (r.stderr if not r.ok else ""), None

    binary = ex.path.with_suffix("")
    cmd = [
        NVCC,
        f"-arch={DEFAULT_NVCC_ARCH}",
        "-O2",
        "-std=c++17",
    ]
    if NVCC_HOST_CC is not None:
        cmd += ["-ccbin", NVCC_HOST_CC]
    cmd += ["-o", str(binary), str(ex.path)]
    r = _run(cmd, ex.path.parent)
    return r.ok, r.stderr, (binary if r.ok else None)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def _parse_expected(ex: Exercise) -> dict:
    """Parse the validator spec for one exercise.

    The format is intentionally flat -- a tiny key/value block with an
    optional payload after a `---` separator. Anything we'd want to express
    in YAML works fine here without pulling in a YAML dependency.

    Example:
        mode: stdout_contains
        ---
        Hello from thread 0
        Hello from thread 1
    """
    if not ex.expected.exists():
        # No spec means "compile + run; rc==0 is success". This is the right
        # default for early exercises where we just want the kernel to launch.
        return {"mode": "rc_zero", "payload": ""}

    text = ex.expected.read_text()
    head, _, payload = text.partition("\n---\n")
    spec: dict[str, str] = {}
    for line in head.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            spec[k.strip()] = v.strip()
    spec["payload"] = payload.rstrip("\n")
    return spec


def _diff(expected: str, got: str) -> str:
    return "\n".join(difflib.unified_diff(
        expected.splitlines(), got.splitlines(),
        lineterm="", fromfile="expected", tofile="got",
    ))


def _validate_stdout_exact(spec: dict, run: RunResult) -> tuple[bool, str]:
    if run.stdout.rstrip() == spec["payload"].rstrip():
        return True, ""
    return False, "stdout doesn't match expected output:\n" + _diff(
        spec["payload"], run.stdout
    )


def _validate_stdout_contains(spec: dict, run: RunResult) -> tuple[bool, str]:
    """Every non-empty line in the payload must appear somewhere in stdout."""
    missing = [
        line for line in spec["payload"].splitlines()
        if line.strip() and line not in run.stdout
    ]
    if not missing:
        return True, ""
    return False, "stdout is missing these required lines:\n  " + "\n  ".join(missing)


def _validate_stdout_regex(spec: dict, run: RunResult) -> tuple[bool, str]:
    pat = re.compile(spec["payload"], re.DOTALL | re.MULTILINE)
    if pat.search(run.stdout):
        return True, ""
    return False, f"stdout doesn't match regex /{spec['payload']}/"


def _validate_numeric(spec: dict, run: RunResult) -> tuple[bool, str]:
    """Parse a single float from stdout (last numeric token) and compare.

    spec keys:
        target: <float>     -- the expected value
        tol: <float>        -- absolute tolerance (default 1e-3)

    Useful for kernels that print a checksum: e.g. dot product result.
    """
    target = float(spec["target"])
    tol = float(spec.get("tol", "1e-3"))
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", run.stdout)
    if not nums:
        return False, "no numeric value found in stdout"
    got = float(nums[-1])
    if abs(got - target) <= tol:
        return True, ""
    return False, f"got {got}, expected {target} (tol {tol})"


def _validate_pytest(spec: dict, run: RunResult, ex: Exercise) -> tuple[bool, str]:
    """Run pytest on a sibling test file. Used for Python exercises."""
    test_file = ex.path.parent / spec.get("file", f"test_{ex.name}.py")
    if not test_file.exists():
        return False, f"missing test file {test_file.name}"
    r = _run([PYTHON, "-m", "pytest", "-q", str(test_file)], ex.path.parent)
    return r.ok, (r.stdout + r.stderr) if not r.ok else ""


_VALIDATORS = {
    "rc_zero":          lambda s, r, e: (True, "") if r.ok else (False, "program exited non-zero"),
    "stdout_exact":     lambda s, r, e: _validate_stdout_exact(s, r),
    "stdout_contains":  lambda s, r, e: _validate_stdout_contains(s, r),
    "stdout_regex":     lambda s, r, e: _validate_stdout_regex(s, r),
    "numeric":          lambda s, r, e: _validate_numeric(s, r),
    "pytest":           lambda s, r, e: _validate_pytest(s, r, e),
}


# ---------------------------------------------------------------------------
# The big one: validate a single exercise end to end
# ---------------------------------------------------------------------------
def check(ex: Exercise) -> bool:
    print(f"{bold('•')} {ex.display}  {gray(f'[{ex.language}]')}")

    if ex.has_not_done_marker():
        print(f"  {yellow('⏸  not started')} -- remove the `I AM NOT DONE` line "
              f"to attempt this exercise.")
        return False

    ok, err, binary = build(ex)
    if not ok:
        print(red("  ✗ build failed"))
        # Compiler errors are gold for learning -- print them in full.
        for line in err.splitlines():
            print(f"    {gray(line)}")
        return False

    spec = _parse_expected(ex)

    # Run the produced artifact (or the .py file directly).
    if ex.language == "python":
        run = _run([PYTHON, str(ex.path)], ex.path.parent)
    else:
        run = _run([str(binary)], ex.path.parent)

    validator = _VALIDATORS.get(spec["mode"])
    if validator is None:
        print(red(f"  ✗ unknown validator mode '{spec['mode']}' "
                  f"in {ex.expected.name}"))
        return False

    allow_nonzero = spec.get("allow_nonzero_rc", "false").lower() == "true"
    if spec["mode"] != "rc_zero" and not run.ok and not allow_nonzero:
        # If the program crashed, surface that BEFORE checking outputs --
        # otherwise students see "stdout doesn't match" when actually their
        # kernel segfaulted.
        print(red(f"  ✗ runtime error (rc={run.rc})"))
        for line in (run.stderr or run.stdout).splitlines()[-10:]:
            print(f"    {gray(line)}")
        return False

    passed, msg = validator(spec, run, ex)
    if passed:
        print(green(f"  ✓ passed  ({run.duration_s*1000:.0f} ms)"))
        return True

    print(red("  ✗ validation failed"))
    for line in msg.splitlines():
        print(f"    {gray(line)}")
    return False


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------
def cmd_list(items: list[Exercise]) -> int:
    print(bold(f"\n{len(items)} exercises across {len({e.chapter for e in items})} chapters\n"))
    current_chapter = None
    for e in items:
        if e.chapter != current_chapter:
            print(f"\n{bold(blue(e.chapter))}")
            current_chapter = e.chapter
        if e.has_not_done_marker():
            tag = yellow("⏸ todo  ")
        else:
            tag = gray("· started")
        print(f"  {tag}  {e.order:3d}  {e.name}")
    print()
    return 0


def cmd_run(items: list[Exercise], query: str) -> int:
    ex = find(items, query)
    if ex is None:
        print(red(f"No exercise matches '{query}'"))
        return 2
    return 0 if check(ex) else 1


def cmd_next(items: list[Exercise]) -> int:
    """Find and validate the first exercise that is either still marked
    'I AM NOT DONE' or that fails to validate. This is what `watch` calls."""
    for e in items:
        if e.has_not_done_marker():
            print(yellow(f"\n→ Next exercise: {e.display}\n"))
            check(e)
            return 1
        if not check(e):
            return 1
    print(green("\n🎉 All exercises pass! You've worked through the course.\n"))
    return 0


def cmd_watch(items: list[Exercise]) -> int:
    """Crude file-watcher: poll mtimes every 0.5s and rerun `next` on change.

    A real watcher would use inotify/fsevents. Polling keeps zero deps and
    is plenty fast for ~200 small files.
    """
    print(blue("CUDAlings watch -- save any exercise file to re-run.  Ctrl-C to exit.\n"))
    last_mtimes: dict[Path, float] = {}
    try:
        while True:
            changed = False
            for e in items:
                m = e.path.stat().st_mtime if e.path.exists() else 0
                if last_mtimes.get(e.path) != m:
                    last_mtimes[e.path] = m
                    changed = True
            if changed:
                print(gray("─" * 60))
                cmd_next(items)
                print(gray("─" * 60) + "\n")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nbye!")
        return 0


def cmd_hint(items: list[Exercise], query: str) -> int:
    ex = find(items, query)
    if ex is None:
        print(red(f"No exercise matches '{query}'"))
        return 2
    hint = ex.path.parent / f"{ex.name}.hint.md"
    if not hint.exists():
        print(yellow("No hint available for this exercise -- try the README!"))
        return 1
    print(hint.read_text())
    return 0


def cmd_solution(items: list[Exercise], query: str) -> int:
    ex = find(items, query)
    if ex is None:
        print(red(f"No exercise matches '{query}'"))
        return 2
    sol = SOLUTIONS_DIR / ex.chapter / ex.path.name
    if not sol.exists():
        print(yellow(f"No reference solution at {sol}"))
        return 1
    print(gray(f"--- {sol} ---"))
    print(sol.read_text())
    return 0


def cmd_reset(items: list[Exercise], query: str) -> int:
    """Restore the original stub by copying from `solutions/.../<name>.stub.<ext>`.

    The stubs are checked in next to the solutions, so a reset is a plain copy.
    """
    ex = find(items, query)
    if ex is None:
        print(red(f"No exercise matches '{query}'"))
        return 2
    stub = SOLUTIONS_DIR / ex.chapter / f"{ex.name}.stub{ex.path.suffix}"
    if not stub.exists():
        print(yellow(f"No stub recorded for {ex.display} (path: {stub})"))
        return 1
    ex.path.write_text(stub.read_text())
    print(green(f"Reset {ex.display} from {stub.name}"))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cudalings",
        description="Progressive CUDA + Python exercises (rustlings-style).")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list",  help="show all exercises and their status")
    sub.add_parser("next",  help="run the first unfinished exercise")
    sub.add_parser("watch", help="re-run `next` whenever a file changes")
    for name in ("run", "hint", "solution", "reset"):
        sp = sub.add_parser(name)
        sp.add_argument("query", help="exercise name, number, or substring")

    args = p.parse_args(argv)
    items = discover()

    if not items:
        print(yellow(f"No exercises found under {EXERCISES_DIR}.\n"
                     "Add exercises and rerun."))
        return 1

    if args.cmd in (None, "next"):
        return cmd_next(items)
    if args.cmd == "list":
        return cmd_list(items)
    if args.cmd == "watch":
        return cmd_watch(items)
    if args.cmd == "run":
        return cmd_run(items, args.query)
    if args.cmd == "hint":
        return cmd_hint(items, args.query)
    if args.cmd == "solution":
        return cmd_solution(items, args.query)
    if args.cmd == "reset":
        return cmd_reset(items, args.query)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
