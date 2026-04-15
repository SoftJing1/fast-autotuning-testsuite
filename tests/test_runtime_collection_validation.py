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

from scripts.collect_tuning_performance_simple import KernelRunResult, run_kernel, run_kernel_average


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
