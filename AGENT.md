## Goal

Write a prototype of a forking agent.

The goal of this agent is to research a problem, which has many solutions and alternatives. The entire exploration could be seen as a tree.

So when the agent as multiple alternatives, it acn explore all of them by forking.

Forking means:
- a dedicated tool call, which receives the fork, mainly an idea and a "prompt".
- the fork tool automatically creates a git worktree with the current state (incl. copied not commited files) and proceed in parallel.

We implement it on top of agentknit. If we find missing APIs or bugs in agentknit, we add/fix directly.
