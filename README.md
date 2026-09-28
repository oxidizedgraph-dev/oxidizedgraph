# oxidizedgraph

Commercial Rust engine for state-checkpointed agent workflows.

**Site:** https://oxidizedgraph.dev  
**Contact:** [hello@oxidizedgraph.dev](mailto:hello@oxidizedgraph.dev)

This GitHub organization is the **public release surface** only. It is not the development forge.

This tree is a **library evaluation cut**: graph definition, compilation, in-process execution, and local JSON checkpoints. The production host (`oxidizedgraph-server`), fleet overlays, and private registry are not published here.

This software is **not open source**. Evaluation rights are in [`LICENSE`](LICENSE). Production use requires a written agreement.

## Evaluate locally

Requires a recent stable Rust (`rustc` 1.85+).

```bash
git clone https://github.com/oxidizedgraph-dev/oxidizedgraph.git
cd oxidizedgraph
cargo test
cargo run --example simple_workflow
```

Other examples: `react_agent`, `multi_agent`, `streaming_events`, `hitl_workflow`, `memory_workflow`.

Default local persistence is `EmbeddedCheckpointer` writing JSON at `OG_CHECKPOINT_PATH` or `.og/state.json`. Tests use an in-memory checkpointer.

## What this crate is

- `GraphBuilder` / `NodeExecutor` / `GraphRunner` — compile an IO-free graph, then run it
- `MemoryCheckpointer` — ephemeral state for tests
- `EmbeddedCheckpointer` — local JSON checkpoints
- Typed nodes, conditional edges, cycles, HITL pause/resume, in-process subgraphs

## What this crate is not

- Not the A2A HTTP host
- Not a Kubernetes worker spawner
- Not a production database adapter
- Not a dump of the internal product repository

## License and security

- License: [`LICENSE`](LICENSE)
- Vulnerability reports: [hello@oxidizedgraph.dev](mailto:hello@oxidizedgraph.dev) with subject `SECURITY` — see [`SECURITY.md`](SECURITY.md)
