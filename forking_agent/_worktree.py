"""Git worktree creation that carries over the uncommitted state."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def git_identity(agent_id: str) -> dict[str, str]:
    """Environment that makes git author and commit as agent *agent_id*.

    Every commit made in a fork, by the agent itself or by :func:`commit_fork`,
    then names the fork that produced it.
    """
    name, email = f"forking-agent {agent_id}", f"{agent_id}@forking-agent.invalid"
    return {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    r = subprocess.run(["git", *args], check=False, cwd=cwd, capture_output=True,
                       text=True, env={**os.environ, **env} if env else None)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {r.stderr.strip()}")
    return r.stdout


def create_fork_worktree(source: Path, dest: Path, branch: str,
                         parent_id: str = "operator") -> Path:
    """Create worktree *dest* on new *branch* mirroring *source*'s current state.

    The worktree starts at *source*'s HEAD, then modified, deleted and untracked
    (non-ignored) files are copied over, so the fork sees exactly what the
    parent sees.  That inherited state is committed as one snapshot commit
    authored by *parent_id*, so ``HEAD`` of the new worktree is the base of
    the fork's own work (``base..branch``).
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
    commit_fork(dest, parent_id, f"snapshot of {parent_id}'s working state for {branch}")
    return dest


def head(worktree: Path) -> str:
    return _git(worktree, "rev-parse", "HEAD").strip()


def commit_fork(worktree: Path, agent_id: str, message: str) -> str | None:
    """Commit everything (non-ignored) in *worktree*; return the sha, None if no change."""
    _git(worktree, "add", "-A")
    if not _git(worktree, "status", "--porcelain").strip():
        return None
    _git(worktree, "commit", "-q", "--no-verify", "-m", message, env=git_identity(agent_id))
    return head(worktree)


def diff(worktree: Path, base: str, tip: str, *, stat: bool = False) -> str:
    """``git diff base tip`` (``--stat`` if *stat*); branches are shared by all worktrees."""
    return _git(worktree, "diff", *(["--stat"] if stat else []), base, tip)


def merge_in_progress(worktree: Path) -> bool:
    r = subprocess.run(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd=worktree,
                       capture_output=True, check=False)
    return r.returncode == 0


def merge_fork(worktree: Path, branch: str, agent_id: str, message: str) -> list[str]:
    """Merge *branch* into *worktree* as *agent_id*; return conflicted paths ([] if clean).

    Pending changes are first committed as a checkpoint, so the merge never
    touches uncommitted work.  On conflict the merge is left in progress for
    the agent to resolve and commit.
    """
    commit_fork(worktree, agent_id, f"checkpoint before merging {branch}")
    r = subprocess.run(["git", "merge", "--no-ff", "--no-edit", "-m", message, branch],
                       cwd=worktree, capture_output=True, text=True, check=False,
                       env={**os.environ, **git_identity(agent_id)})
    if r.returncode == 0:
        return []
    conflicted = _git(worktree, "diff", "--name-only", "--diff-filter=U").split()
    if not conflicted:
        raise RuntimeError(f"git merge {branch} failed in {worktree}: {r.stderr.strip()}")
    return conflicted
