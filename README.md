# forking-agent

Research agent that explores a problem with many solutions as a tree. Built on [agentknit](../agentknit).

When the agent faces real alternatives it calls the `fork(idea, prompt)` tool. Each fork:

- gets its own git worktree (`<repo>-forks/<id>`, branch `fork/<id>`) holding the parent's current state, uncommitted and untracked files included (ignored files are not copied);
- runs a child agent in parallel on `prompt`;
- can fork again, up to `--max-depth`;
- when done, commits everything in its worktree on `fork/<id>`, message = idea + final report. The inherited uncommitted state is part of that commit, so the branch alone reproduces the fork.

Provenance: every agent's `bash` runs with git author and committer `forking-agent <id> <<id>@forking-agent.invalid>` (`root` for the top agent), so each commit, whether the agent's own or the final one, names the fork that made it.

Merging results and branches back is not done: each alternative stays on its branch.

`wait_forks()` blocks until the agent's own forks finish and returns their reports. A crashed fork is reported as a result, not a crash.

## Usage

```
pip install -e .[dev]
forking-agent "Find the fastest way to dedupe lines in a 10GB file" --repo . [--json]
```

Default model: `glm-5.3-flash` via `run:///home/martin/bin/zai-glm-5.3-flash-completions.py` (`--model`, `--endpoint` to change).

Tools given to every agent: `bash`, `read_file`, `write_file` (all rooted in its worktree), `fork`, `wait_forks`.

## Programmatic

```python
from forking_agent import ForkManager
from forking_agent._forks import agentknit_runner

mgr = ForkManager(Path("."), agentknit_runner("glm-5.3-flash", "run:///home/me/bin/zai-glm-5.3-flash-completions.py"))
print(mgr.start("task"))
```

`ForkManager(repo, runner)` accepts any `runner(task, tools, system) -> str`, which keeps tests offline.

## Tests

```
pytest
```
