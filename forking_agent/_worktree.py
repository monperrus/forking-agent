"""Git worktree creation that carries over the uncommitted state."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], check=False, cwd=cwd, capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {r.stderr.strip()}")
    return r.stdout


def create_fork_worktree(source: Path, dest: Path, branch: str) -> Path:
    """Create worktree *dest* on new *branch* mirroring *source*'s current state.

    The worktree starts at *source*'s HEAD, then modified, deleted and untracked
    (non-ignored) files are copied over, so the fork sees exactly what the
    parent sees.  Nothing is committed in the fork.
    """
    source, dest = Path(source).resolve(), Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(source, "worktree", "add", "-b", branch, str(dest), "HEAD")
    # -m modified, -d deleted, -o untracked; -z for safe filenames
    listing = _git(source, "ls-files", "-z", "-m", "-d", "-o", "--exclude-standard")
    for rel in filter(None, listing.split("\0")):
        src, dst = source / rel, dest / rel
        if src.is_symlink() or src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            shutil.copy2(src, dst, follow_symlinks=False)
        elif not src.exists() and (dst.exists() or dst.is_symlink()):
            dst.unlink()
    return dest
