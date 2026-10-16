# forking-agent

Research agent that explores a problem with many solutions as a tree. Built on [agentknit](../agentknit).

Your checkout is never written to: the top agent (`root`) itself runs in a worktree on branch `fork/<run>/root`, `<run>` being a timestamp per invocation. When it ends, merge its branch yourself (`git merge fork/<run>/root`).

When an agent faces real alternatives it calls the `fork(idea, prompt)` tool. Each fork:

- gets its own git worktree (`<repo>-forks/<run>/<id>`, branch `fork/<run>/<id>`) holding the parent's current state, uncommitted and untracked files included (ignored files are not copied). That inherited state becomes one *snapshot* commit authored by the parent; its sha is the fork's `base`, so `base..fork/<run>/<id>` is exactly the fork's own work;
- runs a child agent in parallel on `prompt`;
- can fork again, up to `--max-depth`;
- when done, commits what is left in its worktree, message = idea + final report.

`wait_forks()` blocks until the agent's own forks finish and returns, per fork, its diff stat and report. A crashed fork is reported as a result, not a crash. `fork_diff(id)` shows a fork's exact changes.

`adopt_fork(id)` merges a finished fork of the caller into the caller's worktree: pending changes are committed as a checkpoint, then `git merge --no-ff`. Adopting several forks one after another combines them; on conflict, the merge is left in progress and the agent resolves it. Any agent can adopt its own forks, so good results fold up the tree before the parent sees them.

Provenance: every agent's `bash` runs with git author and committer `forking-agent <id> <<id>@forking-agent.invalid>` (`operator` for the snapshot of your checkout), so every commit, merges included, names the agent that made it, and `git blame` on the final result says which fork wrote each line.

## Usage

```
pip install -e .[dev]
forking-agent "Find the fastest way to dedupe lines in a 10GB file" --repo . [--json]
```

Default model: `glm-5.3-flash` via `run:///home/martin/bin/zai-glm-5.3-flash-completions.py` (`--model`, `--endpoint` to change).

Tools given to every agent: `bash`, `read_file`, `write_file` (all rooted in its worktree), `fork`, `wait_forks`, `fork_diff`, `adopt_fork`.

`--json` prints `{"reply", "root", "forks"}`, each fork with `id`, `idea`, `path`, `branch`, `base`, `commit`, `reply`.

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
