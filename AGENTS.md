# AGENTS.md — Coding Guidelines

Coding conventions and design patterns for this codebase.

---

## Table of Contents

1. [Imports](#imports)
2. [Typing & Tensor Shapes](#typing--tensor-shapes)
3. [Naming Conventions](#naming-conventions)
4. [Design Patterns](#design-patterns)
5. [Configuration System](#configuration-system)
6. [Documentation](#documentation)
7. [Error Handling](#error-handling)
8. [Python Features](#python-features)
9. [Enums](#enums)
10. [Static Analysis](#static-analysis)
11. [Testing](#testing)
12. [Agent Skills](#agent-skills)
13. [Regression Testing Workflow](#regression-testing-workflow)
14. [Quick Reference](#quick-reference)

---

## Imports

**Order:** stdlib → third-party → local (blank lines between groups).

- Explicit imports only — never `from module import *`
- Use `import typing as T`, `import jaxtyping as Jt`

```python
import os
from pathlib import Path

import torch
import typing as T
import jaxtyping as Jt

from ..Utility.Config import load_config
```

---

## Typing & Tensor Shapes

**Dataclasses:** Use `jaxtyping` + `beartype` for runtime shape validation:

```python
@Jt.jaxtyped(typechecker=beartype)
@dataclass(kw_only=True)
class StereoData:
    T_BS: Jt.Float64[pp.LieTensor, "B 7"]
    imageL: Jt.Float32[torch.Tensor, "B 3 H W"]
```

**Functions:** Always annotate shapes in signatures using jaxtyping. Type variables use `T_` prefix.

---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | `PascalCase` | `FlowFormerMatcher` |
| Interfaces | `I` + `PascalCase` | `IMatcher`, `IOptimizer` |
| Functions/methods | `snake_case` | `load_config` |
| Constants | `SCREAMING_SNAKE_CASE` | `DATA_ROOT_CONFIG` |
| Private | Leading `_` | `_update`, `_HIERARCHY` |
| Type variables | `T_` prefix | `T_Input`, `T_Data` |

**Transforms:** `T_AB` = A ← B (from B to A). Document frames in comments.

---

## Design Patterns

**Interfaces:** `ABC` + `@abstractmethod`, prefixed with `I`.

**SubclassRegistry:** Config-driven instantiation — `IMatcher.instantiate(config.type, config.args)`.

**ConfigTestable:** Implement `is_valid_config` with `_enforce_config_spec`:

```python
cls._enforce_config_spec(config, {"weight": lambda s: isinstance(s, str), "device": lambda s: s in {"cuda", "cpu"}})
```

**ConfigTestableSubclass** = SubclassRegistry + ConfigTestable.

**Factory:** `from_config`, `load_from_mapv2`, etc.

**Dataclasses:** `@dataclass(kw_only=True)`. Use `slots=True` for performance-critical inner classes.

**Decorators:** `@SparseBA.register`, `@Rerun_Visualizer.register` for feature/visualization registration.

---

## Configuration System

- Load via `load_config(Path(...))` → `SimpleNamespace`
- Validate with `_enforce_config_spec` — **never silent defaults**
- Missing fields → raise `KeyError`, not `getattr(..., default)`

---

## Documentation

- Module docstring at top of every file
- Section separators: `# ==== Section Title ====`
- Docstrings: `Args` / `Returns` / `Note`
- Document coordinate frames explicitly

---

## Error Handling

- **Fail fast** — raise clear exceptions when requirements unmet
- **Assert** for internal invariants
- **No silent fallbacks** — never `except Exception: result = default`

---

## Python Features

- **match/case** — exhaustive handling; `case _: raise ValueError(...)` for unknown
- **Union types** — `X | None`, `Literal["a", "b"]`
- **Type hints** — required on all function signatures

---

## Enums

Centralize in `Src/Utility/Enum.py`:

```python
class ConstraintStatus(Enum):
    Unoptimized = 0x00
    Optimized   = 0x02
```

---

## Static Analysis

Pyright: `typeCheckingMode = "standard"`. Must pass `pyright`.

- Exhaustive match, no duplicate imports, respect ABC contracts

---

## Testing

- Tests live in: `Src/Scripts/UnitTest/`, `Src/Scripts/AutoTest/`, and `Src/CUExt/WarpKernels/Test/`.
- Test placement rule:
  - For files under `Src/CUExt/WarpKernels/`, write corresponding parity tests in `Src/CUExt/WarpKernels/Test/` (keep one source file mapped to one test file).
  - For all other new TDD tests, place them in `Src/Scripts/AutoTest/`.
- Warp speed benchmarks live in: `Src/CUExt/WarpKernels/Bench/`.
  - Execute all benchmarks with: `python -m Src.CUExt.WarpKernels.Bench`.
  - Keep one source file mapped to one benchmark module (`bench_*.py`) to avoid monolithic suites.
  - Benchmark config must be schema-driven in `Src/CUExt/WarpKernels/Bench/Config.py`:
    - `BENCHMARK_CONFIG`: benchmark values keyed by benchmark function name.
    - `BENCHMARK_SPEC`: required keys and validator functions for each benchmark entry.
    - `validate()` must fail fast on missing keys, invalid values, excessive keys, and unexpected benchmark names.
- Markers: `@pytest.mark.local`, `@pytest.mark.trt`, `@pytest.mark.vio`
- Validate configs: parametrize over YAML files, call `is_valid_config`

---

## Agent Skills

Project skills in `.cursor/skills/` extend agent capabilities for this codebase:

| Skill | Purpose |
|-------|---------|
| **pyright-type-checking** | Enforce type annotations, jaxtyping, fail-fast; run `pyright Src/` before commit |
| **result-evaluation** | EvalSeq metrics (ATE, RTE, ROE); read `eval.json` in sandboxes for quantitative results |
| **result-visualization** | PlotSeq trajectory plots; read PNGs in sandboxes (Trajectory.png, Translation.png, etc.) |
| **test-driven-development** | TDD workflow: write failing test first, minimal code to pass, refactor; read before regression testing |

Use these skills when running `main_VO`, `main_VIO`, `main_SLAM`, or when the user asks for evaluation/visualization.

---

## Regression Testing Workflow

**Any plan involving changes to odometry/SLAM behavior must always include regression testing steps.** Do not skip this even if the change appears behavioral-neutral.

When making changes that may affect odometry/SLAM behavior, follow this workflow to regression-test:

0. **Read the test-driven-development skill** — Before proceeding, read `.cursor/skills/test-driven-development/SKILL.md` and follow its TDD workflow throughout the remaining steps. Write failing tests before any implementation code.

1. **Select config and entry point** — Create or use an existing config in `Config/Experiment/` that covers the changes. Choose the appropriate entry point: `main_VIO.py`, `main_VO.py`, or `main_SLAM.py`.

2. **Run before-change evaluation** — Execute a baseline run with an appropriate sequence (e.g. `Config/Sequence/Example/EuRoC_MH01_IMU.yaml`). Use `--clip 0:500` to limit to 500 frames and control runtime. **Avoid setting the result root** — the sandbox folder (with timestamp) differentiates runs. If desired, set the `name` field in the odometry YAML config to change the saved path automatically generated within the Results directory.

3. **Record before-change metrics** — Use the **result-evaluation** skill: read `eval.json` in the sandbox for quantitative baseline (RTE, ROE, ATE).

4. **Conduct changes** — Apply the intended modifications.

5. **Run pyright** — Use the **pyright-type-checking** skill. Iterate until `pyright Src/` passes, or the remaining error is clearly due to insufficient type support from an external package or limited deduction capability of pyright.

6. **Run after-change evaluation** — Execute under **exactly the same configuration** as step 2. Compare `eval.json` metrics to the baseline. Report any regression. **When regression demonstrates reduced performance**, use the **result-visualization** skill: run PlotSeq on the sandboxes and read the generated PNGs (Trajectory.png, Translation.png, Rotation.png, RTE/ROE CDFs) to inspect the reason for the regression.

---

## Quick Reference

1. Minimal, explicit code — no magic
2. Type everything; validate all configs
3. Fail fast with clear messages
4. Document tensor shapes and coordinate frames
5. `T_AB` = A ← B; factory-based construction
6. Pyright must pass before commit
7. Regression-test behavior changes: baseline run → changes → pyright → after-change run (same config)

**Checklist:** Imports ordered | `T`/`Jt` aliases | `I` prefix for interfaces | `kw_only=True` dataclasses | `is_valid_config` | no silent defaults
