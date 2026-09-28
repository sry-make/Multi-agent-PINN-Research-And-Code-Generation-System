# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent PINN (Physics-Informed Neural Networks) research assistant. Uses LangGraph to orchestrate a 3-agent SOP workflow: Researcher (literature search) -> Coder (code generation + Docker sandbox execution) -> Examiner (quality gate with retry loop). Written in Chinese-first style (comments, prompts, UI text).

## Commands

```bash
# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt   # adds pytest, pytest-asyncio, pytest-cov, pytest-xdist

# Run
python main.py                        # TUI mode (default)
python main.py --cli                  # CLI interactive mode
python main.py --query "..."          # single-shot query

# Test
python -m pytest tests/unit tests/integration -q
python -m pytest tests/unit tests/integration -q --cov=memory --cov=orchestrator --cov=agents --cov=tools --cov=sandbox --cov=tui --cov-report=term-missing
python tests/smoke_test_phase2.py     # environment health check (Ollama, Docker, ChromaDB)

# Eval
python -m eval.runner --mode mock --judge-mode heuristic    # offline eval
python -m eval.runner --mode live --judge-mode llm          # live eval with real LLM

# Infrastructure
docker build -t pinn_agent_sandbox:latest -f sandbox/Dockerfile.sandbox .
python -m rag.build_memory            # build ChromaDB vector store from papers/
```

## Configuration

All config lives in `config.py` — the single source of truth. No hardcoded paths/params elsewhere. Environment variables use `PINN_AGENT_` prefix, loaded from `.env` / `.env.local` via python-dotenv.

Two LLM backends: `ollama` (local, default) and `qwen` (DashScope API). Switch via `PINN_AGENT_LLM_PROVIDER`. Each agent role (researcher, coder, examiner, router) has its own model setting.

## Architecture

### SOP Workflow (orchestrator/graph.py)

```
parse_intent → memory_read → [researcher | coder] → examiner → synthesize → memory_writeback → END
```

- `AgentState` (orchestrator/state.py): TypedDict shared blackboard — all inter-agent data flows through this
- Router (orchestrator/router.py): LLM intent detection with regex fallback. Intents: `qa`, `survey`, `code`, `full_pipeline`
- Examiner verdict `FAIL` triggers coder retry (up to `EXAMINER_MAX_RETRIES=3`)

### Agent Pattern

All three agents (agents/researcher.py, agents/coder.py, agents/examiner.py) follow the same ReAct loop pattern:
1. Build system prompt with memory context
2. Call LLM with tool bindings
3. Parse tool calls, execute, feed results back
4. Loop until max iterations or done
5. Return partial state dict to update AgentState

### Tool Layer

Tools are LangChain `@tool` decorated functions in `tools/`:
- `code_tools.py`: execute_python (Docker sandbox), read_file, write_file, run_shell (whitelist-gated)
- `rag_tools.py`: search_local_papers (ChromaDB + bge-reranker)
- `search_tools.py`: search_arxiv, web_search (DuckDuckGo)
- `formula_tools.py`: simplify_formula, latex_to_sympy

### Three-Tier Memory (memory/)

- Session memory (`sessions/<id>.json`): compressed message history, recent queries/errors/artifacts
- Project memory (`project_memory.json`): long-term facts, architecture rules, tech stack
- Experience memory (`experience_db.jsonl`): success/failure cases, deduplicated by fingerprint

Memory read/writeback are explicit graph nodes, not hidden inside agents.

### Sandbox (sandbox/docker_runner.py)

`DockerSandbox.run_python()` executes code in `pinn_agent_sandbox:latest` with `--network none`, CPU/mem limits, timeout. Artifacts exported to `outputs/sandbox_runs/<run_id>/`.

### Observability (observability/)

- `tracer.py`: logs all LLM calls and tool invocations to `logs/trace_*.jsonl`
- `cost_tracker.py`: token accounting per session/agent/step

## Key Conventions

- Agent entry functions: `run_researcher(state)`, `run_coder(state)`, `run_examiner(state)` — each returns a partial dict to merge into AgentState
- Graph nodes are thin wrappers that import and call agent functions (lazy imports to avoid circular deps)
- Conditional edges use plain functions returning next node name strings
- `build_graph(checkpointer=None)` returns a compiled LangGraph; MemorySaver used by default for multi-turn
- All graph invocations require `config={"configurable": {"thread_id": session_id}}`
