"""Fork bookkeeping and the agent tools (bash, files, fork, wait_forks)."""

from __future__ import annotations

import itertools
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agentknit import Tool

from ._worktree import create_fork_worktree

# runner(task, tools, system_prompt_supplement) -> final reply text
Runner = Callable[[str, "list[Tool]", str], str]

SYSTEM = (
    "You research a problem that has several possible solutions. Work inside your "
    "own git worktree. When you face real alternatives, call `fork` once per "
    "alternative (idea + prompt): each fork gets a copy of your current worktree "
    "(uncommitted files included) and works in parallel. Then call `wait_forks` and "
    "compare the results. Do not fork for trivial choices. Finish with a concise "
    "report: what you tried, what worked, what you recommend."
)


@dataclass
class ForkResult:
    fork_id: str
    idea: str
    path: Path
    branch: str
    reply: str


class ForkManager:
    """Owns the fork tree of one research run."""

    def __init__(self, repo: Path, runner: Runner, *, max_depth: int = 2,
                 max_parallel: int = 4) -> None:
        self.repo = Path(repo).resolve()
        self.runner = runner
        self.max_depth = max_depth
        self._pool = ThreadPoolExecutor(max_workers=max_parallel * (max_depth + 1))
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._futures: dict[str, Future[ForkResult]] = {}
        self._children: dict[str, list[str]] = {}  # parent id -> child ids

    # -- tool factory ------------------------------------------------------
    def tools_for(self, agent_id: str, workdir: Path, depth: int) -> list[Tool]:
        """Tools bound to one agent's worktree.

        agentknit keys its global tool library by function name, so each
        callable is renamed uniquely per agent to avoid cross-fork clashes.
        """
        tag = agent_id.replace("-", "_").replace(".", "_")

        def bash(command: str) -> tuple[str, dict]:
            r = subprocess.run(command, shell=True, cwd=workdir, capture_output=True,
                               text=True, timeout=600)
            out = (r.stdout + r.stderr)[-20000:]
            return f"exit={r.returncode}\n{out}", {"result": "ok" if r.returncode == 0 else "error"}

        def read_file(path: str) -> tuple[str, dict]:
            return (workdir / path).read_text(), {"result": "ok"}

        def write_file(path: str, content: str) -> tuple[str, dict]:
            p = workdir / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"wrote {len(content)} chars to {path}", {"result": "ok"}

        def fork(idea: str, prompt: str) -> tuple[str, dict]:
            if depth >= self.max_depth:
                return f"ERROR: max fork depth {self.max_depth} reached", {"result": "error"}
            fid = self.fork(agent_id, workdir, idea, prompt, depth + 1)
            return f"forked {fid} ({idea}); running in parallel", {"result": "ok"}

        def wait_forks() -> tuple[str, dict]:
            return self.wait(agent_id), {"result": "ok"}

        specs: list[tuple[Callable[..., Any], str, dict[str, str]]] = [
            (bash, "Run a shell command in your worktree.", {"command": "Shell command"}),
            (read_file, "Read a file (path relative to your worktree).", {"path": "File path"}),
            (write_file, "Write a file (path relative to your worktree).",
             {"path": "File path", "content": "File content"}),
            (fork, "Explore an alternative in parallel: copies your worktree "
                   "(incl. uncommitted files) and runs a child agent on `prompt`.",
             {"idea": "Short name of the alternative", "prompt": "Full task for the child"}),
            (wait_forks, "Block until all forks you started finish; returns their reports.", {}),
        ]
        tools = []
        for fn, desc, props in specs:
            name = fn.__name__
            fn.__name__ = f"{name}_{tag}"
            params: dict[str, object] = {"type": "object",
                      "properties": {k: {"type": "string", "description": v}
                                     for k, v in props.items()},
                      "required": list(props)}
            tools.append(Tool(name, desc, fn, parameters=params))
        return tools

    # -- forking -----------------------------------------------------------
    def start(self, task: str) -> str:
        """Run the root agent in the repo itself and return its final reply."""
        tools = self.tools_for("root", self.repo, 0)
        return self.runner(task, tools, SYSTEM)

    def fork(self, parent_id: str, parent_dir: Path, idea: str, prompt: str,
             depth: int) -> str:
        with self._lock:
            fid = f"{parent_id}.{next(self._ids)}" if parent_id != "root" else f"f{next(self._ids)}"
            self._children.setdefault(parent_id, []).append(fid)
        branch = f"fork/{fid}"
        dest = self.repo.parent / f"{self.repo.name}-forks" / fid
        create_fork_worktree(parent_dir, dest, branch)  # synchronous: snapshot now
        self._futures[fid] = self._pool.submit(self._run_child, fid, idea, prompt, dest,
                                               branch, depth)
        return fid

    def _run_child(self, fid: str, idea: str, prompt: str, dest: Path, branch: str,
                   depth: int) -> ForkResult:
        tools = self.tools_for(fid, dest, depth)
        task = f"You are fork {fid}, exploring: {idea}\n\n{prompt}"
        try:
            reply = self.runner(task, tools, SYSTEM)
        except Exception as e:  # noqa: BLE001 - a failed branch is a result, not a crash
            reply = f"ERROR: fork crashed: {e!r}"
        return ForkResult(fid, idea, dest, branch, reply)

    def wait(self, parent_id: str) -> str:
        ids = list(self._children.get(parent_id, []))
        if not ids:
            return "no forks to wait for"
        parts = []
        for fid in ids:
            r = self._futures[fid].result()
            parts.append(f"## {r.fork_id}: {r.idea}\nworktree: {r.path}  branch: {r.branch}\n{r.reply}")
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
