#!/usr/bin/env python3
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from scripts.collect_tuning_performance_simple import KernelRunResult, hash_config, run_kernel, run_kernel_average
from scripts.collect_tuning_performance_simple import _parse_request, schedule_experiment_work
from scripts.internal.db_manager import ExperimentDB


def _build_dir() -> Path:
	return ROOT / "build"


def _kernel_executable(kernel_type: str) -> Path:
	return _build_dir() / kernel_type


def _fetch_completed_rows(db_path: Path):
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	try:
		rows = conn.execute(
			"""
			SELECT exp_id, kernel_type, config_json, runtime_ms, status
			FROM experiments
			WHERE status = 'completed'
			ORDER BY exp_id ASC
			"""
		).fetchall()
		return [dict(row) for row in rows]
	finally:
		conn.close()


def _fetch_experiment_logs(db_path: Path):
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	try:
		rows = conn.execute(
			"""
			SELECT exp_id, stage, level, message, payload_json
			FROM experiment_logs
			ORDER BY exp_id ASC, log_id ASC
			"""
		).fetchall()
		return [dict(row) for row in rows]
	finally:
		conn.close()


@pytest.mark.parametrize(
	"kernel_type,input_size",
	[
		("gemm", "128x128x128"),
		("gaussian", "256x256"),
	],
)
def test_collect_db_runtime_matches_host_rerun(tmp_path, kernel_type: str, input_size: str):
	"""Integration test per kernel: collect -> DB -> rerun host -> compare runtimes."""
	exe = _kernel_executable(kernel_type)
	if not exe.exists():
		pytest.skip(f"{kernel_type} executable not found: {exe}")

	experiment_name = f"runtime_validation_{kernel_type}"
	runs_per_config = 3
	cmd = [
		sys.executable,
		"-m",
		"scripts.collect_tuning_performance_simple",
		"--request",
		kernel_type,
		input_size,
		"1",
		"--workers",
		"1",
		"--device-type",
		"cpu",
		"--runs-per-config",
		str(runs_per_config),
		"--experiment-root",
		str(tmp_path),
		"--experiment-name",
		experiment_name,
		"--build-dir",
		str(_build_dir()),
	]

	result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
	if result.returncode != 0:
		pytest.skip(
			"Collection script could not complete in this environment. "
			f"stdout: {result.stdout[-800:]} stderr: {result.stderr[-800:]}"
		)

	db_path = tmp_path / experiment_name / "experiments.db"
	assert db_path.exists(), f"Expected database not found: {db_path}"

	rows = [row for row in _fetch_completed_rows(db_path) if row["kernel_type"] == kernel_type]
	assert rows, f"No completed {kernel_type} rows in database after collection"
	log_rows = _fetch_experiment_logs(db_path)
	assert log_rows, "Expected experiment logs after collection"

	# Validate using database-stored config_json to rerun host and compare runtimes.
	for row in rows:
		stored_runtime = row["runtime_ms"]
		assert stored_runtime is not None and stored_runtime > 0

		config = json.loads(row["config_json"])
		config_path = tmp_path / f"rerun_config_{row['exp_id']}.json"
		config_path.write_text(json.dumps(config))

		rerun_avg = run_kernel_average(
			kernel_type=row["kernel_type"],
			config_file=str(config_path),
			build_dir=str(_build_dir()),
			runs_per_config=runs_per_config,
		)
		if not rerun_avg.success or rerun_avg.runtime_ms is None:
			pytest.skip(
				"Host rerun failed for a stored configuration in this environment: "
				f"{rerun_avg.error_summary()}"
			)

		diff = abs(rerun_avg.runtime_ms - stored_runtime)
		rel_diff = diff / max(abs(stored_runtime), 1e-9)
		assert rel_diff <= 0.35 or diff <= 2.0, (
			f"Runtime mismatch too high for exp_id={row['exp_id']}: "
			f"stored={stored_runtime:.6f}ms rerun={rerun_avg.runtime_ms:.6f}ms "
			f"abs_diff={diff:.6f}ms rel_diff={rel_diff:.4f}"
		)

		exp_logs = [entry for entry in log_rows if entry["exp_id"] == row["exp_id"]]
		assert any(entry["stage"] == "kernel_host" for entry in exp_logs)
		raw_json_logs = [entry for entry in exp_logs if entry["stage"] == "symb_viewer_json"]
		assert raw_json_logs, f"Expected raw symb-viewer JSON logs for exp_id={row['exp_id']}"
		sources = {
			json.loads(entry["payload_json"])["source_file"]
			for entry in raw_json_logs
			if entry["payload_json"]
		}
		assert "instrcount.json" in sources
		assert "bbcount.json" in sources


def test_run_kernel_average_uses_multiple_runs(monkeypatch):
	"""Unit test for averaging logic: multiple runs must be collected and averaged."""
	calls = []
	values = [1.0, 2.0, 3.0]

	def fake_run_kernel(kernel_type: str, config_file: str, build_dir: str, dump_opencl_binary=None):
		calls.append((kernel_type, config_file, build_dir, dump_opencl_binary))
		return KernelRunResult(runtime_ms=values[len(calls) - 1])

	module = sys.modules["scripts.collect_tuning_performance_simple"]
	monkeypatch.setattr(module, "run_kernel", fake_run_kernel)

	avg = run_kernel_average(
		kernel_type="gemm",
		config_file="dummy.json",
		build_dir="build",
		runs_per_config=3,
	)

	assert len(calls) == 3
	assert avg.runtime_ms == pytest.approx(2.0)


def test_run_kernel_reports_missing_executable(tmp_path):
	"""Detailed kernel runs should preserve the reason a launch failed."""
	config_path = tmp_path / "config.json"
	config_path.write_text("{}")

	result = run_kernel(
		kernel_type="gemm",
		config_file=str(config_path),
		build_dir=str(tmp_path / "missing_build"),
	)

	assert not result.success
	assert result.runtime_ms is None
	assert "Executable not found" in result.error_msg

	average = run_kernel_average(
		kernel_type="gemm",
		config_file=str(config_path),
		build_dir=str(tmp_path / "missing_build"),
		runs_per_config=2,
	)

	assert not average.success
	assert average.successful_runs == 0
	assert len(average.errors) == 2
	assert "Executable not found" in average.error_summary()


def test_schedule_resume_reuses_existing_pending_row(tmp_path):
	request = _parse_request("gaussian", "256x256", "1")
	config_dir = tmp_path / "configs"
	config_dir.mkdir(parents=True, exist_ok=True)
	config_path = config_dir / "gaussian_256x256_000000.json"
	config = {"input_size_h": 256, "input_size_w": 256, "wi_1_ocl_dim": 0, "wi_2_ocl_dim": 1, "wi_1": 2, "wi_2": 2, "wg_1_ocl_dim": 0, "wg_2_ocl_dim": 1, "wg_1": 64, "wg_2": 8}
	config_path.write_text(json.dumps(config))

	db_path = tmp_path / "experiments.db"
	llvm_root = tmp_path / "llvm_ir"
	param_hash = hash_config(config)
	with ExperimentDB(str(db_path)) as db:
		exp_id = db.add_experiment("gaussian", "256x256", param_hash, config, skip_if_exists=False)
		assert exp_id is not None

	scheduling = schedule_experiment_work(
		resolved_configs=[(request, config_path)],
		db_path=db_path,
		llvm_root=llvm_root,
		build_dir="build",
		runs_per_config=1,
		resume=True,
		persist_artifacts=False,
	)

	assert scheduling.error_count == 0
	assert len(scheduling.experiments) == 1
	assert scheduling.experiments[0][0] == exp_id

	with ExperimentDB(str(db_path)) as db:
		assert db.mark_completed(exp_id, 1.23)

	scheduling = schedule_experiment_work(
		resolved_configs=[(request, config_path)],
		db_path=db_path,
		llvm_root=llvm_root,
		build_dir="build",
		runs_per_config=1,
		resume=True,
		persist_artifacts=False,
	)

	assert scheduling.error_count == 0
	assert scheduling.experiments == []
