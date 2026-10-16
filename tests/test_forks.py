"""Offline tests: worktree snapshots, the fork/wait flow, and merging forks back."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from forking_agent import ForkManager, create_fork_worktree
from forking_agent._worktree import commit_fork


def git(cwd: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


def ident(agent_id: str) -> str:
    return f"forking-agent {agent_id}"


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


def tools(ts) -> dict:
    return {x.name: x.fn for x in ts}


def test_worktree_copies_uncommitted_state_into_snapshot(repo: Path, tmp_path: Path) -> None:
    (repo / "a.txt").write_text("modified\n")        # tracked, modified
    (repo / "new.txt").write_text("untracked\n")     # untracked
    (repo / "ignored.txt").write_text("secret\n")    # ignored -> not copied
    (repo / "gone.txt").unlink()                     # tracked, deleted
    dest = create_fork_worktree(repo, tmp_path / "wt", "fork/x", "p")
    assert (dest / "a.txt").read_text() == "modified\n"
    assert (dest / "new.txt").read_text() == "untracked\n"
    assert not (dest / "ignored.txt").exists()
    assert not (dest / "gone.txt").exists()
    assert git(dest, "rev-parse", "--abbrev-ref", "HEAD").strip() == "fork/x"
    # inherited state is one snapshot commit by the parent; the worktree is clean
    assert git(dest, "log", "-1", "--format=%an").strip() == ident("p")
    assert set(git(dest, "show", "--name-only", "--format=").split()) == {"a.txt", "new.txt", "gone.txt"}
    assert git(dest, "status", "--porcelain") == ""


def test_clean_source_gets_no_snapshot_commit(repo: Path, tmp_path: Path) -> None:
    dest = create_fork_worktree(repo, tmp_path / "wt", "fork/clean")
    assert git(dest, "rev-parse", "HEAD") == git(repo, "rev-parse", "HEAD")
    assert commit_fork(dest, "clean", "nothing") is None


def test_forks_run_in_parallel_with_own_work_and_provenance(repo: Path) -> None:
    (repo / "wip.txt").write_text("uncommitted\n")
    status_before, head_before = git(repo, "status", "--porcelain"), git(repo, "rev-parse", "HEAD")
    seen: dict[str, str] = {}

    def runner(task, ts, system):
        t = tools(ts)
        if task == "root":
            t["fork"]("A", "do A")
            t["fork"]("B", "do B")
            return t["wait_forks"]()[0]
        seen[task] = t["read_file"]("wip.txt")[0]       # child sees uncommitted file
        t["write_file"]("own.txt", task)
        t["bash"]("git add own.txt && git commit -qm own")  # agent commits itself
        t["write_file"]("out.txt", task)
        return f"done {task.splitlines()[-1]}"

    mgr = ForkManager(repo, runner)
    report = mgr.start("root")
    assert "done do A" in report and "done do B" in report
    assert "2 files changed" in report                   # diff stat of the fork's own work
    assert set(seen.values()) == {"uncommitted\n"}
    # the operator's checkout is never written to
    assert git(repo, "status", "--porcelain") == status_before
    assert git(repo, "rev-parse", "HEAD") == head_before
    assert mgr.root is not None and mgr.root.branch.endswith("/root")
    results = mgr.results()
    assert len(results) == 2
    for r in results:
        assert r.commit and git(repo, "rev-parse", r.branch).strip() == r.commit
        # base..branch is exactly the fork's own work, all commits by the fork
        assert set(git(repo, "diff", "--name-only", r.base, r.branch).split()) == {"own.txt", "out.txt"}
        authors = git(repo, "log", "--format=%an", f"{r.base}..{r.branch}").splitlines()
        committers = git(repo, "log", "--format=%cn", f"{r.base}..{r.branch}").splitlines()
        assert authors == committers == [ident(r.fork_id)] * 2
        # the root changed nothing before forking, so the base is the root's own
        # snapshot of the operator's uncommitted wip.txt
        assert git(repo, "log", "-1", "--format=%an", r.base).strip() == ident("operator")


def test_adopt_two_forks_merges_with_provenance(repo: Path) -> None:
    def runner(task, ts, system):
        t = tools(ts)
        if task == "root":
            t["fork"]("A", "A")
            t["fork"]("B", "B")
            t["wait_forks"]()
            assert "+A" in t["fork_diff"]("f1")[0]
            assert t["adopt_fork"]("f1")[1]["result"] == "ok"
            assert t["adopt_fork"]("f2")[1]["result"] == "ok"
            return "adopted"
        name = task.splitlines()[-1]
        t["write_file"](f"{name}.txt", f"{name}\n")
        return name

    mgr = ForkManager(repo, runner)
    assert mgr.start("root") == "adopted"
    root = mgr.root
    assert root is not None
    assert (root.path / "A.txt").exists() and (root.path / "B.txt").exists()
    assert git(root.path, "blame", "--porcelain", "A.txt").count(f"author {ident('f1')}") == 1
    merges = git(root.path, "log", "--merges", "--format=%an|%s").splitlines()
    assert merges == [f"{ident('root')}|adopt f2: B", f"{ident('root')}|adopt f1: A"]
    assert not (repo / "A.txt").exists()


def test_adopt_conflict_is_left_to_the_agent(repo: Path) -> None:
    out: dict[str, tuple] = {}

    def runner(task, ts, system):
        t = tools(ts)
        if task == "root":
            t["fork"]("X", "X")
            t["fork"]("Y", "Y")
            t["wait_forks"]()
            out["first"] = t["adopt_fork"]("f1")
            out["second"] = t["adopt_fork"]("f2")
            out["third"] = t["adopt_fork"]("f2")
            return "done"
        t["write_file"]("a.txt", task.splitlines()[-1] + "\n")
        return "edited"

    mgr = ForkManager(repo, runner)
    mgr.start("root")
    assert out["first"][1]["result"] == "ok"
    text, meta = out["second"]
    assert meta["result"] == "error" and "conflicts in: a.txt" in text
    assert "merge is in progress" in out["third"][0]


def test_fork_diff_and_adopt_refuse_foreign_or_running_forks(repo: Path) -> None:
    release = threading.Event()
    out: dict[str, str] = {}

    def runner(task, ts, system):
        t = tools(ts)
        if task == "root":
            t["fork"]("slow", "slow")
            out["running"] = t["fork_diff"]("f1")[0]
            out["foreign"] = t["adopt_fork"]("f9")[0]
            release.set()
            t["wait_forks"]()
            out["empty"] = t["fork_diff"]("f1")[0]
            return "done"
        release.wait(10)
        return "nothing changed"

    ForkManager(repo, runner).start("root")
    assert "still running" in out["running"]
    assert "not one of your forks" in out["foreign"]
    assert out["empty"] == "no changes"


def test_nested_adoption_folds_the_tree(repo: Path) -> None:
    def runner(task, ts, system):
        t = tools(ts)
        if task == "root":
            t["fork"]("mid", "mid")
            t["wait_forks"]()
            t["adopt_fork"]("f1")
            return "root done"
        if task.endswith("mid"):
            t["fork"]("leaf", "leaf")
            t["wait_forks"]()
            assert t["adopt_fork"]("f1.2")[1]["result"] == "ok"
            return "mid done"
        t["write_file"]("leaf.txt", "leaf\n")
        return "leaf done"

    mgr = ForkManager(repo, runner)
    mgr.start("root")
    assert mgr.root is not None
    assert git(mgr.root.path, "blame", "--porcelain", "leaf.txt").count(f"author {ident('f1.2')}") == 1


def test_two_runs_on_one_repo_do_not_collide(repo: Path) -> None:
    def runner(task, ts, system):
        t = tools(ts)
        t["fork"]("x", "x")
        return t["wait_forks"]()[0]

    a = ForkManager(repo, runner, run_id="r1")
    b = ForkManager(repo, runner, run_id="r2")
    a.start("t")
    b.start("t")
    assert a.root and b.root and a.root.branch != b.root.branch


def test_cli_json_stdout_is_pure_json(repo: Path, monkeypatch, capsys) -> None:
    from forking_agent import _cli

    def runner(task, ts, system):
        print("agentknit progress noise")
        (Path(ts[0].fn("pwd")[0].split("\n", 1)[1].strip()) / "r.txt").write_text("r")
        return "final"

    monkeypatch.setattr(_cli, "agentknit_runner", lambda model, endpoint: runner)
    assert _cli.main(["task", "--repo", str(repo), "--json"]) == 0
    out = capsys.readouterr()
    data = json.loads(out.out)
    assert data["reply"] == "final" and data["forks"] == []
    assert data["root"]["commit"] and data["root"]["branch"].endswith("/root")
    assert "noise" in out.err


def test_depth_limit_and_crash_isolation(repo: Path) -> None:
    def runner(task, ts, system):
        t = tools(ts)
        if task == "root":
            t["fork"]("boom", "x")
            return t["wait_forks"]()[0]
        raise RuntimeError("bad branch")

    mgr = ForkManager(repo, runner, max_depth=1)
    assert "ERROR: fork crashed" in mgr.start("root")
    assert "max fork depth" in tools(mgr.tools_for("z", repo, 1))["fork"]("i", "p")[0]
