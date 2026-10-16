"""Fork bookkeeping and the agent tools (bash, files, fork, wait_forks, fork_diff, adopt_fork)."""

from __future__ import annotations

import itertools
import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agentknit import Tool

from ._worktree import (commit_fork, create_fork_worktree, diff, git_identity, head,
                        merge_fork, merge_in_progress)

# runner(task, tools, system_prompt_supplement) -> final reply text
Runner = Callable[[str, "list[Tool]", str], str]

SYSTEM = (
    "You research a problem that has several possible solutions. Work inside your "
    "own git worktree. When you face real alternatives, call `fork` once per "
    "alternative (idea + prompt): each fork gets a copy of your current worktree "
    "(uncommitted files included) and works in parallel. Then call `wait_forks`, "
    "compare the results (`fork_diff` shows a fork's exact changes) and call "
    "`adopt_fork` on the fork(s) you recommend to merge their work into your "
    "worktree; adopt only what you recommend. Do not fork for trivial choices. "
    "Finish with a concise report: what you tried, what worked, what you adopted."
)

DIFF_LIMIT = 30000


@dataclass
class ForkResult:
    fork_id: str
    idea: str
    path: Path
    branch: str
    reply: str
    commit: str | None  # tip of `branch` when the fork ended, None if it made no commit past `base`
    base: str = ""      # snapshot of the inherited state; the fork's own work is base..branch


def _ok(text: str) -> tuple[str, dict]:
    return text, {"result": "ok"}


def _err(text: str) -> tuple[str, dict]:
    return f"ERROR: {text}", {"result": "error"}


class ForkManager:
    """Owns the fork tree of one research run.

    The root agent itself runs in a worktree (``fork/<run>/root``): the
    operator's checkout is never written to.
    """

    def __init__(self, repo: Path, runner: Runner, *, max_depth: int = 2,
                 max_parallel: int = 4, run_id: str | None = None) -> None:
        self.repo = Path(repo).resolve()
        self.runner = runner
        self.max_depth = max_depth
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.root: ForkResult | None = None
        self._pool = ThreadPoolExecutor(max_workers=max_parallel * (max_depth + 1))
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._futures: dict[str, Future[ForkResult]] = {}
        self._children: dict[str, list[str]] = {}  # parent id -> child ids

    # -- tool factory ------------------------------------------------------
    def tools_for(self, agent_id: str, workdir: Path, depth: int) -> list[Tool]:
        """Tools bound to one agent's worktree."""
        env = {**os.environ, **git_identity(agent_id)}

        def bash(command: str) -> tuple[str, dict]:
            r = subprocess.run(command, shell=True, cwd=workdir, capture_output=True,
                               text=True, timeout=600, env=env)
            out = (r.stdout + r.stderr)[-20000:]
            return f"exit={r.returncode}\n{out}", {"result": "ok" if r.returncode == 0 else "error"}

        def read_file(path: str) -> tuple[str, dict]:
            return _ok((workdir / path).read_text())

        def write_file(path: str, content: str) -> tuple[str, dict]:
            p = workdir / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return _ok(f"wrote {len(content)} chars to {path}")

        def fork(idea: str, prompt: str) -> tuple[str, dict]:
            if depth >= self.max_depth:
                return _err(f"max fork depth {self.max_depth} reached")
            fid = self.fork(agent_id, workdir, idea, prompt, depth + 1)
            return _ok(f"forked {fid} ({idea}); running in parallel")

        def wait_forks() -> tuple[str, dict]:
            return _ok(self.wait(agent_id))

        def fork_diff(fork_id: str) -> tuple[str, dict]:
            r = self._finished_child(agent_id, fork_id)
            if isinstance(r, str):
                return _err(r)
            text = diff(workdir, r.base, r.branch)
            if len(text) > DIFF_LIMIT:
                text = text[:DIFF_LIMIT] + f"\n[... truncated; full diff: git diff {r.base} {r.branch}]"
            return _ok(text or "no changes")

        def adopt_fork(fork_id: str) -> tuple[str, dict]:
            r = self._finished_child(agent_id, fork_id)
            if isinstance(r, str):
                return _err(r)
            if merge_in_progress(workdir):
                return _err("a merge is in progress: resolve it and commit first")
            conflicts = merge_fork(workdir, r.branch, agent_id,
                                   f"adopt {r.fork_id}: {r.idea}\n\n{r.reply}")
            if conflicts:
                return _err(f"merge of {r.branch} conflicts in: {', '.join(conflicts)}. "
                            "Resolve the markers, `git add` them and `git commit --no-edit`.")
            return _ok(f"merged {r.branch} into your worktree")

        specs: list[tuple[Callable[..., Any], str, dict[str, str]]] = [
            (bash, "Run a shell command in your worktree.", {"command": "Shell command"}),
            (read_file, "Read a file (path relative to your worktree).", {"path": "File path"}),
            (write_file, "Write a file (path relative to your worktree).",
             {"path": "File path", "content": "File content"}),
            (fork, "Explore an alternative in parallel: copies your worktree "
                   "(incl. uncommitted files) and runs a child agent on `prompt`.",
             {"idea": "Short name of the alternative", "prompt": "Full task for the child"}),
            (wait_forks, "Block until all forks you started finish; returns their "
                         "reports and diff stats.", {}),
            (fork_diff, "Show the exact changes a finished fork of yours made.",
             {"fork_id": "Fork id, e.g. f1"}),
            (adopt_fork, "Merge a finished fork of yours into your worktree (git merge, "
                         "keeps the fork's commits). Your pending changes are committed first.",
             {"fork_id": "Fork id, e.g. f1"}),
        ]
        tools = []
        for fn, desc, props in specs:
            params: dict[str, object] = {"type": "object",
                      "properties": {k: {"type": "string", "description": v}
                                     for k, v in props.items()},
                      "required": list(props)}
            tools.append(Tool(fn.__name__, desc, fn, parameters=params))
        return tools

    # -- forking -----------------------------------------------------------
    def _worktree_for(self, fid: str, parent_id: str, parent_dir: Path) -> tuple[Path, str, str]:
        branch = f"fork/{self.run_id}/{fid}"
        dest = self.repo.parent / f"{self.repo.name}-forks" / self.run_id / fid
        create_fork_worktree(parent_dir, dest, branch, parent_id)
        return dest, branch, head(dest)

    def start(self, task: str) -> str:
        """Run the root agent in its own worktree and return its final reply."""
        dest, branch, base = self._worktree_for("root", "operator", self.repo)
        reply = self.runner(task, self.tools_for("root", dest, 0), SYSTEM)
        commit_fork(dest, "root", f"root: final state\n\n{reply}")
        tip = head(dest)
        self.root = ForkResult("root", task.splitlines()[0] if task else "", dest, branch,
                               reply, tip if tip != base else None, base)
        return reply

    def fork(self, parent_id: str, parent_dir: Path, idea: str, prompt: str,
             depth: int) -> str:
        with self._lock:
            fid = f"{parent_id}.{next(self._ids)}" if parent_id != "root" else f"f{next(self._ids)}"
            self._children.setdefault(parent_id, []).append(fid)
        dest, branch, base = self._worktree_for(fid, parent_id, parent_dir)  # snapshot now
        self._futures[fid] = self._pool.submit(self._run_child, fid, idea, prompt, dest,
                                               branch, base, depth)
        return fid

    def _run_child(self, fid: str, idea: str, prompt: str, dest: Path, branch: str,
                   base: str, depth: int) -> ForkResult:
        tools = self.tools_for(fid, dest, depth)
        task = f"You are fork {fid}, exploring: {idea}\n\n{prompt}"
        try:
            reply = self.runner(task, tools, SYSTEM)
        except Exception as e:  # noqa: BLE001 - a failed branch is a result, not a crash
            reply = f"ERROR: fork crashed: {e!r}"
        commit_fork(dest, fid, f"fork {fid}: {idea}\n\n{reply}")
        tip = head(dest)  # the agent may have committed everything itself
        return ForkResult(fid, idea, dest, branch, reply, tip if tip != base else None, base)

    def _finished_child(self, parent_id: str, fork_id: str) -> ForkResult | str:
        """The result of *parent_id*'s own fork *fork_id*, or an error message."""
        if fork_id not in self._children.get(parent_id, []):
            return f"{fork_id!r} is not one of your forks ({', '.join(self._children.get(parent_id, [])) or 'none'})"
        fut = self._futures[fork_id]
        if not fut.done():
            return f"{fork_id} is still running; call wait_forks first"
        return fut.result()

    def wait(self, parent_id: str) -> str:
        ids = list(self._children.get(parent_id, []))
        if not ids:
            return "no forks to wait for"
        parts = []
        for fid in ids:
            r = self._futures[fid].result()
            stat = diff(r.path, r.base, r.branch, stat=True).strip() or "no changes"
            parts.append(f"## {r.fork_id}: {r.idea}\nbranch: {r.branch}\n{stat}\n\n{r.reply}")
        return "\n\n".join(parts)

    def results(self) -> list[ForkResult]:
        return [f.result() for f in self._futures.values()]


def agentknit_runner(model: str, endpoint: str, **kwargs: Any) -> Runner:
    """Runner backed by agentknit.run_agent."""
    from agentknit import run_agent

    def run(task: str, tools: list[Tool], system: str) -> str:
        res = run_agent(task=task, model=model, endpoint=endpoint, tools=tools,
                        system_prompt_supplement=system, non_interactive=True, **kwargs)
        return res.final_reply or ""

    return run
