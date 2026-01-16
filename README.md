# fast-autotuning-testsuite

This repository contains a small test-suite and tools for extracting and
working with tuning parameters from OpenCL kernel templates used by
an autotuning flow.

**Workflow**
- **Inspect templates:** kernel templates live in the `kernel-template/` directory. Each file is an OpenCL kernel that uses preprocessor macros for tunable parameters (work-group sizes, cache-block sizes, vector widths, etc.).
- **Extract parameters:** use the parser CLI `python3 -m scripts.get_tuning_parameters [path]` to extract candidate tuning parameters. `path` can be a single `.cl` file or a directory — by default it scans `kernel-template/`.
- **Save / reuse:** the CLI can write JSON output with `--json out.json` for downstream tools.
- **Automated tests:** parsing logic is covered by pytest tests in `tests/` — run `pytest` to validate parser compatibility with all templates.

**Directory layout (key folders)**
- `kernel-template/`: OpenCL kernel template sources to analyze.
- `kernel-instances/`: example kernel instances generated from templates.
- `tuning-parameters/`: (intended) storage for extracted parameter sets.
- `scripts/`: tooling and the parser. Main script: `scripts/get_tuning_parameters.py` exposing a Python API (`parse_tuning_parameters`, `parse_directory`) and a CLI.
- `examples/`: small examples and tuned output used as references.
- `pyatf-artifacts/`: archived artifacts and evaluation assets.
- `tests/`: pytest tests for the parser and related tooling.

**Parser notes**
- The parser uses a conservative heuristic: it collects UPPER_SNAKE identifiers used in a file, subtracts macros defined inside the file (`#define`), and presents the remainder as candidate tuning parameters.
- The heuristic intentionally errs on the side of discovery; downstream filtering (by name patterns or manual review) is expected.

**Quick commands**
```bash
# run parser on the whole templates directory (pretty-printed)
python3 -m scripts.get_tuning_parameters --pretty

# run parser on a single template and save JSON
python3 -m scripts.get_tuning_parameters kernel-template/gemm_1.cl --json gemm1_params.json

# run tests
pytest -q
```

If you want, I can extend the parser to group parameters by intent (work-item vs cache sizes), emit a `requirements.txt`, or add an example pipeline that converts extracted parameters into a tuning configuration.

