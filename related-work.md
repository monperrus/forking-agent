# Related work

forking-agent: an LLM agent calls `fork(idea, prompt)`. A child agent then runs in parallel in a git worktree that holds the parent's state, uncommitted files included. Children can fork further, and a parent merges a chosen child's work back with `git merge`. Each agent commits under its own git identity.

Seven questions separate the related work:

1. **Who decides to fork**: the model itself, an orchestrator or search policy, or a human.
2. **What is forked**: tokens or context only, or the filesystem and repository state.
3. **What the child starts from**: the parent's full history, or a message the parent writes.
4. **Nesting**: can a child fork again?
5. **Join**: what comes back to the parent: tokens, a summary, or code merged into the parent's state.
6. **Selection**: who judges the branches: the model, tests, a benchmark, or a value model.
7. **Provenance**: can you tell afterwards which branch produced what?

## 1. Learned fork/join inside a model's reasoning (the model decides; tokens only)

### Adaptive Parallel Reasoning (APR)
Pan, Li, Lian, Snell, Zhou, Yala, Darrell, Keutzer, Suhr (UC Berkeley, UCSF). [arXiv 2504.15466](https://arxiv.org/abs/2504.15466), v2 Aug 2025. Code: [Parallel-Reasoning/APR](https://github.com/Parallel-Reasoning/APR). The search results showed an [ICML 2025 listing](https://icml.cc/virtual/2025/51796), but the arXiv page names no venue, so the venue is unconfirmed.

- **Primitives.** `spawn(msgs)` takes a list of strings: each string starts one child and is the *only* context that child gets. `join(msg)` ends a child and returns `msg` to the parent. The parent continues from its own context before the spawn plus the joined messages; the children's intermediate traces are thrown away. On their task, failed children return nothing, and successful ones return only the solution path.
- **Execution.** Children are concurrent decoding threads in SGLang, with continuous batching and prefix caching (RadixAttention); the shared parent/child prefix is cached.
- **Training.** A 228M-parameter Llama-2-style model trained from scratch, with a 4k context window. Supervised fine-tuning on 500k Countdown traces from a symbolic solver: a hybrid BFS/DFS search that hands chosen nodes to child threads. Then end-to-end reinforcement learning (GRPO), rewarded for a correct final answer.
- **Results (Countdown).** 83.4% vs 60.0% for the serialized baseline (SoS+) at a 4k context. 80.1% vs 66.6% at 20k total tokens. 75.2% vs 57.3% at about 5 s latency. Reinforcement learning raises APR from 75.5% to 83.4%. Below about 4k tokens APR loses to the baseline, which the authors put down to the overhead of running threads in parallel. From an automatically extracted summary, not checked against the paper: a pretrained Qwen2.5-1.5B reaches 80.2% vs 57.5%; only the main thread spawns, so there is no nesting.
- **Stated limits.** A single toy task (Countdown); models trained from scratch, with pretrained models left as future work; needs supervised traces to bootstrap.
- **Compared with ours.**
  - Same shape: the model decides, a fork is a call with a message, the parent receives a condensed result, and context stays small because children's traces don't come back.
  - Differences:
    - APR forks *tokens*; we fork a *repository state*.
    - APR's join returns text; ours returns text through `wait_forks` *and* code through `adopt_fork` (git merge).
    - Our children are full tool-using agents with their own filesystem.
    - Ours is pure prompting, no training.
  - APR shows that the *decision when to fork* can be learned with RL. In our case it is left to the model's judgment, guided only by the system prompt.

### THREAD: Thinking Deeper with Recursive Spawning
Schroeder et al., [arXiv 2405.17402](https://arxiv.org/abs/2405.17402), 2024. A generation is a thread that can spawn child threads *recursively*. A child's context comes from tokens the parent writes; the child returns only the tokens the parent needs. In one variant the parent waits until the child finishes. Prompted, no training. **Closest in spirit to our nesting**, but it covers only text: no shared filesystem state, no merge.

### Multiverse
Yang, An, Liu, Chen, Chen. [arXiv 2506.09991](https://arxiv.org/abs/2506.09991), NeurIPS 2025. Generation runs in three stages, Map (split the task), Process (subtasks in parallel) and Reduce (combine), and the model itself decides where to parallelize. Multiverse-32B: 54% on AIME24 and 46% on AIME25 after a 3-hour fine-tune on 1K examples, with up to a 2× speedup. Data, weights and inference engine are open source. Their Reduce corresponds to our merge, but it combines text, not code.

### ThreadWeaver, Parallel-R1
[ThreadWeaver](https://arxiv.org/abs/2512.07843) (arXiv 2512.07843) uses control tokens (`<Parallel>`, `<Thread>`) and RL that rewards both correctness and a shorter critical path. Its 8B model reports 79.9% on AIME24, against up to 53.8% for Multiverse-32B and up to 19.4% for Parallel-R1. Parallel-R1 is another RL-trained parallel-thinking model; I haven't read it directly.

### AsyncThink: The Era of Agentic Organization
Microsoft Research, [arXiv 2510.26658](https://arxiv.org/abs/2510.26658). An organizer model emits `Fork` tags (a sub-query handed to a worker) and `Join` tags (wait for that worker). Trained with format fine-tuning, then RL. Listed as future work: "recursive agentic organization", where any worker can become a sub-organizer with its own workers. **That is the nesting we already have.**

### Hogwild! Inference
Rodionov et al., [arXiv 2504.06261](https://arxiv.org/abs/2504.06261), NeurIPS 2025. Parallel copies of the same model share one attention (KV) cache and see each other's tokens as they are generated; works without training. Branches here coordinate *while running*, the opposite of our isolated forks that are compared at the end.

## 2. Agent-level forks with a separate context (the model decides; context only)

### Context-Folding
Sun, Lu et al. (ByteDance Seed, CMU, Stanford), [arXiv 2510.11967](https://arxiv.org/abs/2510.11967), Oct 2025. An agent calls `branch(description, prompt)` to open a sub-trajectory with its own context, and `return(message)` to fold it back into a summary. Trained with FoldGRPO (RL with process rewards). On Deep Research and SWE tasks it matches or beats ReAct with a 10× smaller active context.
- **Almost the same signature as our `fork(idea, prompt)`**, but sequential: one branch at a time, folding context rather than exploring alternatives. The branch also shares the parent's filesystem, so nothing is isolated and nothing needs merging.

### Quine: LLM agents as native POSIX processes
[arXiv 2603.18030](https://arxiv.org/abs/2603.18030). Agents spawn or fork children as OS processes, each with its own context window. The OS-process analogy; I haven't read it in detail.

### Harness subagents: Claude Code, LangChain deepagents
- **Claude Code** ([subagents docs](https://code.claude.com/docs/en/sub-agents)): the model decides when to call its `Agent` tool.
  - `subagent_type: "fork"` inherits the parent's conversation.
  - `isolation: "worktree"` gives the child a git worktree under `.claude/worktrees/agent-<id>` on branch `worktree-agent-<id>`, removed automatically if nothing changed.
  - **Tested 2026-10-16: the worktree starts from `HEAD`. Neither an untracked file nor an uncommitted edit in the parent's checkout was visible inside.**
  - Forks are told not to delegate further, so there is no nesting, and there is no merge tool: the parent gets a text report.
- **LangChain deepagents** ([docs](https://docs.langchain.com/oss/python/deepagents/subagents)): the model calls a `task` tool; "forked" subagents inherit the supervisor's state, which is pitched as a prompt-cache saving. No filesystem branching.
- **Second Thought** ([arXiv 2608.13667](https://arxiv.org/abs/2608.13667)): forks a running agent's full history into parallel reasoning branches that reuse the prompt cache; aimed at reflection rather than exploring alternative implementations.

## 3. Search over code with branches (an orchestrator decides; filesystem)

- **SWE-Search** ([arXiv 2410.20285](https://arxiv.org/abs/2410.20285), ICLR 2025; [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)): Monte Carlo tree search over a software agent's actions, plus a value agent and a discriminator agent that debate. +23% relative on SWE-bench across five models.
- **AB-MCTS / TreeQuest** ([arXiv 2503.04412](https://arxiv.org/abs/2503.04412), NeurIPS 2025; [SakanaAI/treequest](https://github.com/SakanaAI/treequest)): at each node, adaptively choose wider (new attempts) or deeper (refine an existing one), guided by external feedback. A search library, without repository state.
- **AIDE** ([arXiv 2502.13138](https://arxiv.org/abs/2502.13138)): a tree of whole scripts for machine-learning engineering, with draft, debug and improve operators, pruned by a metric, and earlier attempts summarized into hints. A follow-up study ([arXiv 2507.02554](https://arxiv.org/abs/2507.02554)) finds the operators, not the search algorithm, are the bottleneck.
- **agent-mcts** ([natsu0529/mcts-llm-agent](https://github.com/natsu0529/mcts-llm-agent), MIT, v0.1): wraps Claude Code and others in Monte Carlo tree search.
  - A node is a git worktree plus a forked session (Claude Code's `--fork-session`); branches are `agent-mcts/<run>/<node>`.
  - Branches are scored by your tests; `apply` squash-merges the best node.
  - Closest to us in mechanics, but the search policy forks, not the agent, and squashing loses per-branch provenance.
- **evo** ([evo-hq/evo](https://github.com/evo-hq/evo), Apache-2.0, 1.5k★): discovers a benchmark, then runs tree search with parallel subagents, one worktree per experiment. The next branch is chosen by score strategy (argmax, top-k, softmax, …), with pass/fail gates. Merge-back is not described.
- **Vesper** ([arXiv 2605.15221](https://arxiv.org/abs/2605.15221)): a git worktree per agent for algorithm discovery.

## 4. Parallel sessions a human orchestrates (filesystem, no tree)

[claude-squad](https://github.com/smtg-ai/claude-squad), [Crystal/Nimbalyst](https://github.com/stravu/crystal), [uzi](https://github.com/devflowinc/uzi) and [container-use](https://github.com/dagger/container-use) (containers instead of worktrees): one worktree, branch or container per session. A human starts the sessions, compares them and merges. No agent-initiated forks, no nesting.

## Positioning

|                          | who forks | what is forked | child starts from | nested | join / merge | selection | provenance |
|--------------------------|-----------|----------------|-------------------|--------|--------------|-----------|------------|
| APR, Multiverse, ThreadWeaver | model (learned) | tokens | parent's message | APR: main thread only (from summary) | text | model | — |
| THREAD                   | model (prompted) | tokens | parent's message | yes | text | model | — |
| AsyncThink               | model (learned) | tokens | sub-query | future work | text | model | — |
| Context-Folding          | model (learned) | context | parent's message | no | summary | model | — |
| Claude Code fork + worktree | model | context + `HEAD` worktree | full conversation | no | text report | model | branch only |
| agent-mcts, evo          | search policy | worktree (+ session) | (forked) session | yes (tree) | squash apply / none | tests / benchmark | branch per node |
| claude-squad & co.       | human | worktree / container | fresh | no | manual | human | branch |
| **forking-agent**        | model (prompted) | worktree **incl. uncommitted state** | parent's message | **yes, any depth** | text **+ `git merge --no-ff` at any level** | model | **per-agent git identity; `git blame` names the fork** |

**What's ours:**
1. The model decides to fork the *repository state*, uncommitted work included.
2. Forks nest.
3. Results fold back up the tree with real git merges, at any level.
4. Every commit and merge is attributed to the agent that made it.

**Not new:**
- Model-initiated forking (APR, THREAD, Context-Folding, Claude Code).
- A worktree per branch (agent-mcts, evo, claude-squad).
- Tree search over code (SWE-Search, AIDE).

**Ideas from this literature we don't use:**
- Learning when to fork with RL (APR, AsyncThink, Context-Folding).
- Scoring branches with tests or benchmarks (agent-mcts, evo, SWE-Search).
- Children inheriting the parent's context and prompt cache (Claude Code forks, Second Thought).
- Branches communicating while they run (Hogwild!).
