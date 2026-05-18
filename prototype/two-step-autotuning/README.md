# Two-Step Autotuning Prototype

This directory contains two kinds of tuners:

- live end-to-end tuners that generate configs, compile kernels, run kernels,
  and store new results in a run-local SQLite database;
- database replay tuners that search only within the existing 5000-config
  dataset for fast controlled experiments.

## Live Tuners

The live tuners are the real application prototype.

- `two_step_autotuning.live.parameter_tuner` is the baseline. OpenTuner searches
  generated tuning-parameter configs and each valid config is executed directly.
- `two_step_autotuning.live.two_step_tuner` is the proposed method. OpenTuner
  searches an instruction-map encoding. A resolver maps each requested
  instruction map to the closest known live-profiled config, then that config is
  executed and recorded.

Both live tuners write `trace.csv`, `summary.json`, `final_config.json`,
OpenTuner's database, generated config JSON files, dumped OpenCL binaries,
runtime LLVM IR, and `live_experiments.db` under the selected `--output-dir`.

Example baseline run:

```bash
PYTHONPATH=prototype/two-step-autotuning python3 -m two_step_autotuning.live.parameter_tuner \
  --kernel gaussian \
  --input-size 512x512 \
  --output-dir prototype/two-step-autotuning/runs/live_parameter_gaussian_512 \
  --database sqlite:///prototype/two-step-autotuning/runs/live_parameter_gaussian_512/opentuner.db \
  --valid-config-limit 200 \
  --test-limit 20000
```

Example two-step run:

```bash
PYTHONPATH=prototype/two-step-autotuning python3 -m two_step_autotuning.live.two_step_tuner \
  --kernel gaussian \
  --input-size 512x512 \
  --output-dir prototype/two-step-autotuning/runs/live_two_step_gaussian_512 \
  --database sqlite:///prototype/two-step-autotuning/runs/live_two_step_gaussian_512/opentuner.db \
  --bootstrap-profile-count 64 \
  --resolver-candidate-limit 64 \
  --valid-config-limit 200 \
  --test-limit 20000
```

`--resolver-candidate-limit` is the inner resolver autotuning iteration limit
per requested instruction map. It no longer means "pre-profile this many fresh
resolver candidates and choose the nearest one".

The two-step tuner requires `symb-viewer` and its shared-library dependencies to
be available, because it extracts static instruction counts from dumped runtime
LLVM IR before OpenTuner can search the instruction-map space.

## Shared Code Layout

The DB replay and live tuners share the common implementation pieces:

- `core/instruction_space.py`: instruction-map OpenTuner encoding for dataset rows
  and live profiles.
- `core/resolver.py`: approximate instruction-map distance index plus DB and online
  resolver frontends.
- `core/types.py`: shared record/profile/result data classes.
- `core/common_args.py`: shared CLI argument helpers.
- `core/result_recorder.py`: trace and summary output.

Live-only files are kept to the application boundary:

- `live/config_space.py`: online generation of valid host configs.
- `live/executor.py`: compile/profile/run and run-local database storage.
- `live/parameter_tuner.py`: live baseline tuner entrypoint.
- `live/two_step_tuner.py`: live two-step tuner entrypoint.
- `live/diagnose_two_step.py`: concrete single-resolution diagnostic.

## Database Replay Tuners

The previous within-database experiment code remains available with explicit
names:

- `two_step_autotuning.db.parameter_tuner`
- `two_step_autotuning.db.instruction_map_tuner`

The database replay tuners only evaluate completed rows already present in the
configured SQLite experiment database.
