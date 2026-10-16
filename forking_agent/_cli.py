"""Command line: forking-agent "task" [--repo DIR]."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from ._forks import ForkManager, agentknit_runner

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
    if a.json:
        json.dump({"reply": reply,
                   "forks": [{"id": r.fork_id, "idea": r.idea, "path": str(r.path),
                              "branch": r.branch, "commit": r.commit, "reply": r.reply}
                             for r in mgr.results()]}, sys.stdout, indent=2)
        print()
    else:
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
