"""Offline tests: worktree snapshot semantics and the fork/wait tool flow."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forking_agent import ForkManager, create_fork_worktree


def git(cwd: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("a\n")
    (r / "gone.txt").write_text("x\n")
    (r / ".gitignore").write_text("ignored.txt\n")
    git(r, "add", ".")
    git(r, "commit", "-qm", "init")
    return r


def test_worktree_copies_uncommitted_state(repo: Path, tmp_path: Path) -> None:
    (repo / "a.txt").write_text("modified\n")        # tracked, modified
    (repo / "new.txt").write_text("untracked\n")     # untracked
    (repo / "ignored.txt").write_text("secret\n")    # ignored -> not copied
    (repo / "gone.txt").unlink()                     # tracked, deleted
    dest = create_fork_worktree(repo, tmp_path / "wt", "fork/x")
    assert (dest / "a.txt").read_text() == "modified\n"
    assert (dest / "new.txt").read_text() == "untracked\n"
    assert not (dest / "ignored.txt").exists()
    assert not (dest / "gone.txt").exists()
    assert git(dest, "rev-parse", "--abbrev-ref", "HEAD").strip() == "fork/x"


def test_fork_runs_children_in_parallel_and_wait_collects(repo: Path) -> None:
    (repo / "wip.txt").write_text("uncommitted\n")
    seen: dict[str, str] = {}

    def runner(task, tools, system):
        t = {x.name: x.fn for x in tools}
        if task == "root":
            t["fork"]("A", "do A")
            t["fork"]("B", "do B")
            return t["wait_forks"]()[0]
        seen[task] = t["read_file"]("wip.txt")[0]       # child sees uncommitted file
        t["write_file"]("out.txt", task)
        return f"done {task.splitlines()[-1]}"

    mgr = ForkManager(repo, runner)
    report = mgr.start("root")
    assert "done do A" in report and "done do B" in report
    assert set(seen.values()) == {"uncommitted\n"}
    assert not (repo / "out.txt").exists()               # parent isolated from forks
    assert len(mgr.results()) == 2


def test_depth_limit_and_crash_isolation(repo: Path) -> None:
    def runner(task, tools, system):
        t = {x.name: x.fn for x in tools}
        if task == "root":
            t["fork"]("boom", "x")
            return t["wait_forks"]()[0]
        raise RuntimeError("bad branch")

    mgr = ForkManager(repo, runner, max_depth=1)
    assert "ERROR: fork crashed" in mgr.start("root")
    assert "max fork depth" in mgr.tools_for("z", repo, 1)[3].fn("i", "p")[0]
