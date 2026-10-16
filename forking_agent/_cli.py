"""Command line: forking-agent "task" [--repo DIR]."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from ._forks import ForkManager, ForkResult, agentknit_runner

DEFAULT_MODEL = "glm-5.3-flash"
DEFAULT_ENDPOINT = "run:///home/martin/bin/zai-glm-5.3-flash-completions.py"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forking-agent", description=__doc__)
    ap.add_argument("task", help="problem to research")
    ap.add_argument("--repo", default=".", help="git repo to work in (default: .)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--json", action="store_true", help="emit the fork tree as JSON")
    a = ap.parse_args(argv)

    mgr = ForkManager(Path(a.repo), agentknit_runner(a.model, a.endpoint),
                      max_depth=a.max_depth)
    # agentknit prints progress on stdout; keep stdout for the result only
    with contextlib.redirect_stdout(sys.stderr):
        reply = mgr.start(a.task)
    def as_dict(r: ForkResult) -> dict[str, object]:
        return {"id": r.fork_id, "idea": r.idea, "path": str(r.path), "branch": r.branch,
                "base": r.base, "commit": r.commit, "reply": r.reply}

    if a.json:
        assert mgr.root is not None
        json.dump({"reply": reply, "root": as_dict(mgr.root),
                   "forks": [as_dict(r) for r in mgr.results()]}, sys.stdout, indent=2)
        print()
    else:
        print(reply)
        if mgr.root is not None:
            print(f"\nresult branch: {mgr.root.branch} (merge it yourself: git merge {mgr.root.branch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
