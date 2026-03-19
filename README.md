# Config-Driven RL & Game Theory in JAX

A high-performance, modular framework for simulating multi-agent environments and game theory experiments. Built natively in **JAX** and **Equinox**, this project uses a strict **config-driven architecture** to separate experiment definitions from mathematical logic.

## 🏗️ The Config-Driven Architecture

This project solves the classic Reinforcement Learning problem of "hyperparameter soup" by combining the hierarchical composition of **Hydra** with the strict type-safety of **Dacite**.

The configuration pipeline flows in three distinct stages:

### 1. Composition (Hydra)
Instead of a single massive configuration file, settings are broken down into logical, interchangeable YAML components (e.g., `ucb.yaml`, `prisoners_dilemma.yaml`). Hydra dynamically stitches these together at runtime based on the `defaults` specified in the root `config.yaml` or via command-line overrides.

### 2. Type Safety & Parsing (Dacite)
Raw dictionaries are dangerous and lack IDE autocomplete. Before any simulation begins, the resolved Hydra configuration (`DictConfig`) is passed through **Dacite** to instantiate heavily nested, `frozen=True` Python `dataclasses`. 

By utilizing a `type` literal string in the dataclasses, Dacite intelligently resolves `Union` types (e.g., picking `UCBConfig` vs `ThomsonSamplingConfig`), ensuring the application only boots if the provided YAML precisely matches the expected Python types.

### 3. Instantiation (Factories)
The strongly-typed `AppConfig` object is routed to standard Factory patterns (`create_agent`, `create_environment`). These factories unpack the dataclass fields directly into Equinox modules (`eqx.Module`), seamlessly bridging the gap between static configuration and dynamic XLA computation graphs.

## 🚀 Key Features

* **Strict Separation of Concerns:** Environments define the *rules* (pure math), Agents define the *policy*, and the Experiment config defines the *time* (step limits and episode counts).
* **Infinite MDPs via Masking:** Environments like Matrix Games are treated as infinite, stateless Markov Decision Processes. Episode termination (`done=True`) is handled via reward masking within the compiled execution loop.
* **Blazing Fast Vectorization:** The entire interaction loop is written using `jax.lax.scan` (to compile the time dimension) and wrapped in `jax.vmap` (to parallelize the episode/batch dimension). This allows 10,000+ episodes to simulate concurrently on a GPU/TPU in milliseconds.

## 📂 Project Structure

```text
.
├── configs/
│   ├── config.yaml                  # Root config & experiment settings
│   ├── agent/                       # Agent hyperparameters (UCB, Thomson)
│   └── environment/                 # Env matrices (Prisoner's Dilemma)
├── src/config_driven_design/
│   ├── config.py                    # Root AppConfig dataclass definitions
│   ├── run.py                       # Compiled JAX execution loops (scan/vmap)
│   ├── agents/                      # Equinox agent logic & interfaces
│   └── environments/                # Equinox environment logic & interfaces
├── main.py                          # Hydra entry point
└── pyproject.toml                   # Project dependencies (Hatchling)
```

## 🛠️ Quickstart

**1. Install the environment:**
```bash
pip install -e .
```

**2. Run a default experiment:**
```bash
python main.py
```

**3. Run an experiment with overrides (The Hydra advantage):**
```bash
python main.py agent=thomson_sampling experiment.num_episodes=500
```