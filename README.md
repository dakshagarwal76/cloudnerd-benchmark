# ☁️ CloudNerd-Benchmark: Open Deep Research


This repository contains the evaluation and benchmarking codebase for **CloudNerd**, the specialized deep-research extension of the CustomNerd framework. It is designed to evaluate the factual accuracy, retrieval performance, and citation faithfulness of multi-agent Retrieval-Augmented Generation (RAG) systems against a closed-domain local dataset (`cloud.jsonl`).

## 🔬 System Overview

The CloudNerd benchmarking system adapts a LangGraph-based multi-agent architecture (Supervisor-Researcher pattern). Unlike traditional web-connected research agents, CloudNerd benchmark is strictly confined to a local Stack Overflow database, ensuring that all generated insights and citations are verifiably grounded in the provided dataset without external pollution.

### Key Features
* **Multi-Agent Research Workflow:** A Supervisor agent dynamically delegates parallel research tasks to Sub-Researchers based on the complexity of the user's query.
* **Strict Local Grounding (JSONL):** Native web search APIs are disabled. Agents retrieve data exclusively from a local `cloud.jsonl` database using a custom-built search tool.
* **Automated Hallucination Detection:** Incorporates an `AnswerValidator` pipeline that intercepts the final AI-generated report. It cross-references the report with the retrieved dataset to guarantee that every cited "Post ID" corresponds to a valid Stack Overflow post actually retrieved during the run.
* **Resilient Benchmarking Suite:** Built-in automated checkpointing (`test_checkpoint.json`) allows long-running evaluations over hundreds of QA pairs to pause and resume seamlessly.

---

## 📂 Repository Structure

* `test_researcher.py`: The primary evaluation script. Iterates through test queries, invokes the deep research graph, captures performance metrics (duration, success rates), and generates a detailed hallucination/citation summary.
* `deep_researcher.py`: The core LangGraph state machine orchestrating the Supervisor, Researcher, and Final Report Generation nodes.
* `utils.py`: Contains the `jsonl_search` tool logic, calculating keyword relevance to return the most pertinent Stack Overflow posts to the agents.
* `prompts.py`: Specialized system prompts instructing the AI to synthesize findings and properly track Post IDs for the final Sources section without forced inline citations.
* `configuration.py`: Pydantic configurations enforcing the JSONL search API protocol and managing agent iteration limits.
* `state.py`: Defines the operational states and message memory structures for the LangGraph agents.
* `fix_hallucinations.py`: Houses the `AnswerValidator` logic to detect and scrub fabricated Post ID citations.

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/dakshagarwal76/CloudNerd-Benchmarking.git

cd cloudnerd-benchmark
