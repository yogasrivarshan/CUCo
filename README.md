# CUCo: An Agentic Framework for Compute and Communication Co-design

CUCo is a training-free, agent-driven framework that automatically generates high-performance CUDA kernels jointly orchestrating computation and communication. By co-optimizing these traditionally disjoint components, CUCo unlocks optimization opportunities unavailable to existing approaches, reducing end-to-end latency by up to 1.57x over host-driven baselines.

## Overview

CUCo consists of three inter-twined components:

1. **Design Space Specification** — A structured, declarative set of communication primitives (backend, placement, sync scope, issuer granularity, chunk size) that grounds agent reasoning in valid collective semantics.

2. **Fast-Path Agent** — A correctness-first pipeline that converts host-driven NCCL code into device-initiated (GIN/LSA) equivalents through a three-step process: CUDA code analysis, host-to-device transformation via an LLM-judge loop, and evolve-block annotation.

3. **Slow-Path Agent** — An LLM-driven evolutionary search that optimizes the fast-path baseline through island-based populations, phase-dependent explore/exploit mutation, cascaded evaluation, and a shared candidate database with meta-summarization.

### How It Works

Given a host-driven CUDA+NCCL kernel, CUCo's fast-path agent first analyzes the communication pattern, converts host-side collectives to device-initiated GIN/LSA primitives, and annotates mutable regions with EVOLVE-BLOCK markers. The slow-path agent then treats the annotated kernel as generation 0 and runs an evolutionary search: each generation, an LLM mutates the code within the evolve blocks, the candidate is compiled, run, and scored, and the result feeds back into the next iteration. Over 10–20 generations, this loop discovers optimizations like compute-communication overlap, kernel fusion, and pipelined transfers that are difficult to find manually.

## Repository Layout

```
cuco/                   Core framework
├── core/               Evolution runner, sampler, novelty judge, summarizer
├── database/           Candidate database, complexity analysis, island management
├── edit/               Diff/full-rewrite application, async editing
├── llm/                LLM client, model backends (Anthropic, OpenAI, Gemini, DeepSeek)
├── prompts/            Mutation prompt templates (base, diff, full, cross, novelty, meta)
├── transform/          Fast-path agent: CUDA analyzer, host-to-device transformer
├── plots/              Visualization utilities (lineage trees, pareto fronts, improvement plots)
├── webui/              Interactive evolution visualization UI
├── launch/             Local and Slurm launch backends
├── cuco_launch         Entry point for launching evolution runs
└── cuco_visualize      Entry point for the visualization UI
workloads/
└── ds_v3_moe/          DeepSeek-V3 MoE dispatch-compute-combine workload
    ├── ds_v3_moe.cu    Seed CUDA kernel (host-driven baseline)
    ├── evaluate.py     Build, run, and fitness evaluation logic
    ├── run_evo.py      Launch slow-path evolutionary search
    ├── run_transform.py Launch fast-path host-to-device transformation
    └── nccl_api_docs.py NCCL device API documentation for agent context
setup.py               Package configuration and dependencies
uv.lock                Locked dependency versions
```

## Setup

### Prerequisites

- Python >= 3.10
- CUDA 13.1+ with NCCL 2.28.9+ (for device-initiated communication)
- NVIDIA GPUs with NVLink (intra-node) or RoCE/InfiniBand (inter-node)
- LLM API credentials (Anthropic Bedrock, OpenAI, etc.)

### Installation

```bash
# Clone the repository
git clone <this-repo-url>
cd CUCo

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Configuration

Create a `.env` file in the repository root with your LLM API credentials:

```bash
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

## Usage

### Fast-Path Agent (Host-to-Device Transformation)

The fast-path agent converts a host-driven NCCL program into a device-initiated equivalent:

```bash
cd workloads/ds_v3_moe
python run_transform.py
```

This runs the three-step pipeline (CUDA analysis, host-to-device transformation, evolve-block annotation) and outputs the transformed kernel to `_transform_host_output/`.

### Slow-Path Agent (Evolutionary Search)

The slow-path agent optimizes the transformed kernel through LLM-driven evolution:

```bash
cd workloads/ds_v3_moe
python run_evo.py --num_generations=18
```

Evolution results (candidate programs, scores, logs) are saved to `results_ds_v3_moe/`.

### Visualization

Launch the interactive web UI to explore the evolution tree:

```bash
cuco_visualize --db workloads/ds_v3_moe/results_ds_v3_moe/evolution_db.sqlite
```

## Extensibility

CUCo's core framework is workload-agnostic. The evaluation script (`evaluate.py`), prompt customization (`run_evo.py`), and API documentation file are all user-defined — you can adapt CUCo for any kernel, library, or optimization target where an LLM can generate code and a script can score it.

## License

Apache 2.0
