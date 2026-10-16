"""forking-agent: explore a problem's solution tree by forking into git worktrees."""

from __future__ import annotations

from ._forks import ForkManager, ForkResult
from ._worktree import create_fork_worktree

__all__ = ["ForkManager", "ForkResult", "create_fork_worktree"]
