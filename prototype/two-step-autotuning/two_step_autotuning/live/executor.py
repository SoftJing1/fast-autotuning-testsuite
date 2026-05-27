from __future__ import annotations

import json
import statistics
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.collect_tuning_performance_simple import (
	_extract_runtime_ir_result,
	_serialize_llvm_paths,
	_store_symbolic_instruction_counts,
	run_kernel,
)
from scripts.internal.db_manager import ExperimentDB

from ..core.instruction_map import canonical_config_key, sum_counts
from ..core.types import LiveProfile
from .config_space import hash_config


@dataclass(frozen=True)
class LiveExecutionResult:
	status: str
	profile: LiveProfile | None
	runtime_ms: float | None
	error_msg: str = ""
	warmup_runtimes_ms: tuple[float, ...] = ()
	measured_runtimes_ms: tuple[float, ...] = ()

	@property
	def valid(self) -> bool:
		return self.profile is not None and self.status in {"profiled", "completed"}


def check_instruction_counter_available() -> None:
	"""Fail early when the static instruction counter executable is unusable."""
	if shutil.which("symb-viewer") is None:
		raise RuntimeError(
			"live two-step tuning requires symb-viewer on PATH for static instruction counts"
		)
	try:
		result = subprocess.run(
			["symb-viewer", "--help"],
			capture_output=True,
			text=True,
			timeout=10,
		)
	except subprocess.TimeoutExpired:
		return
	except OSError as exc:
		raise RuntimeError(f"failed to start symb-viewer: {exc}") from exc

	combined_output = "\n".join(part for part in (result.stderr, result.stdout) if part)
	if "error while loading shared libraries" in combined_output:
		raise RuntimeError(
			"symb-viewer is present but cannot start; static instruction counting failed: "
			+ combined_output.strip()
		)


class LiveKernelExecutor:
	def __init__(
		self,
		output_dir: str | Path,
		kernel_type: str,
		input_size: str,
		build_dir: str = "build",
		warmup_runs: int = 3,
		runs_per_config: int = 5,
	):
		self.output_dir = Path(output_dir)
		self.kernel_type = kernel_type
		self.input_size = input_size
		self.build_dir = build_dir
		self.warmup_runs = warmup_runs
		self.runs_per_config = runs_per_config
		self.config_dir = self.output_dir / "configs"
		self.llvm_dir = self.output_dir / "_transient_llvm_ir"
		self.binary_dir = self.output_dir / "_transient_opencl_binaries"
		self.db_path = self.output_dir / "live_experiments.db"
		self.config_dir.mkdir(parents=True, exist_ok=True)
		self.llvm_dir.mkdir(parents=True, exist_ok=True)
		self.binary_dir.mkdir(parents=True, exist_ok=True)
		self._profiles_by_key: dict[str, LiveProfile] = {}
		self._runtime_by_key: dict[str, float] = {}

	def profile_config(self, config: dict[str, Any]) -> LiveExecutionResult:
		key = canonical_config_key(config)
		if key in self._profiles_by_key:
			profile = self._profiles_by_key[key]
			return LiveExecutionResult(
				"profiled",
				profile,
				profile.runtime_ms,
				warmup_runtimes_ms=profile.warmup_runtimes_ms,
				measured_runtimes_ms=profile.measured_runtimes_ms,
			)

		param_hash = hash_config(config)
		config_path = self._write_config(config, param_hash)
		binary_path = self.binary_dir / f"{self.kernel_type}_{self.input_size}_{param_hash}.opencl.bin"
		with ExperimentDB(str(self.db_path)) as db:
			exp_id = self._ensure_experiment_row(db, config, param_hash)
			result = self._dump_opencl_binary_only(config_path, binary_path)
			if result.returncode != 0:
				error = self._subprocess_error("dump-only OpenCL binary extraction failed", result)
				db.mark_failed(exp_id, error)
				return LiveExecutionResult("failed", None, None, error)

			ir_result = _extract_runtime_ir_result(
				self.kernel_type,
				config,
				param_hash,
				self.llvm_dir,
				binary_path,
				persist_artifacts=False,
			)
			if not ir_result.success:
				error = ir_result.error_msg or "runtime LLVM IR extraction failed"
				db.mark_failed(exp_id, error)
				return LiveExecutionResult("failed", None, None, error)

			llvm_paths = [artifact["llvm_ir_path"] for artifact in ir_result.artifacts]
			db.mark_profiled(exp_id, None)
			symbolic_errors = _store_symbolic_instruction_counts(db, exp_id, ir_result.artifacts)
			if symbolic_errors:
				error = "instruction-count extraction failed: " + " | ".join(symbolic_errors)
				db.mark_failed(exp_id, error)
				self._cleanup_artifacts(llvm_paths, binary_path)
				return LiveExecutionResult("failed", None, None, error)

			count_rows = db.get_llvm_instruction_counts(exp_id)
			raw_counts = sum_counts(row["total_instruction_counts"] for row in count_rows)
			self._cleanup_artifacts(llvm_paths, binary_path)

		profile = LiveProfile(
			exp_id=exp_id,
			kernel_type=self.kernel_type,
			input_size=self.input_size,
			param_hash=param_hash,
			config=dict(config),
			raw_counts=raw_counts,
		)
		self._profiles_by_key[key] = profile
		return LiveExecutionResult("profiled", profile, None)

	def execute_config(self, config: dict[str, Any]) -> LiveExecutionResult:
		key = canonical_config_key(config)
		if key in self._profiles_by_key and key in self._runtime_by_key:
			profile = self._profiles_by_key[key]
			return LiveExecutionResult(
				"completed",
				profile,
				self._runtime_by_key[key],
				warmup_runtimes_ms=profile.warmup_runtimes_ms,
				measured_runtimes_ms=profile.measured_runtimes_ms,
			)

		param_hash = hash_config(config)
		config_path = self._write_config(config, param_hash)
		binary_path = self.binary_dir / f"{self.kernel_type}_{self.input_size}_{param_hash}.opencl.bin"
		with ExperimentDB(str(self.db_path)) as db:
			exp_id = self._ensure_experiment_row(db, config, param_hash)
			runtime_result = self._run_kernel_with_warmups(config_path, binary_path)
			if runtime_result["runtime_ms"] is None:
				error = runtime_result["error_msg"] or "execution failed"
				db.mark_failed(exp_id, error)
				self._record_runtime_samples(
					db,
					exp_id,
					param_hash,
					config,
					runtime_result["warmup_results"],
					runtime_result["measured_results"],
					success=False,
					average_runtime_ms=None,
				)
				self._cleanup_artifacts([], binary_path)
				return LiveExecutionResult("failed", None, None, error)

			ir_result = _extract_runtime_ir_result(
				self.kernel_type,
				config,
				param_hash,
				self.llvm_dir,
				binary_path,
				persist_artifacts=False,
			)
			if not ir_result.success:
				error = ir_result.error_msg or "runtime LLVM IR extraction failed"
				db.mark_failed(exp_id, error)
				self._cleanup_artifacts([], binary_path)
				return LiveExecutionResult("failed", None, None, error)

			llvm_paths = [artifact["llvm_ir_path"] for artifact in ir_result.artifacts]
			db.mark_completed(exp_id, runtime_result["runtime_ms"], None)
			self._record_runtime_samples(
				db,
				exp_id,
				param_hash,
				config,
				runtime_result["warmup_results"],
				runtime_result["measured_results"],
				success=True,
				average_runtime_ms=runtime_result["runtime_ms"],
			)
			symbolic_errors = _store_symbolic_instruction_counts(db, exp_id, ir_result.artifacts)
			if symbolic_errors:
				db.record_error_message(
					exp_id,
					"instruction-count extraction failed: " + " | ".join(symbolic_errors),
				)
			count_rows = db.get_llvm_instruction_counts(exp_id)
			raw_counts = sum_counts(row["total_instruction_counts"] for row in count_rows)
			self._cleanup_artifacts(llvm_paths, binary_path)

		profile = LiveProfile(
			exp_id=exp_id,
			kernel_type=self.kernel_type,
			input_size=self.input_size,
			param_hash=param_hash,
			config=dict(config),
			raw_counts=raw_counts,
			runtime_ms=runtime_result["runtime_ms"],
			warmup_runtimes_ms=tuple(runtime_result["warmup_runtimes"]),
			measured_runtimes_ms=tuple(runtime_result["measured_runtimes"]),
		)
		self._profiles_by_key[key] = profile
		self._runtime_by_key[key] = runtime_result["runtime_ms"]
		return LiveExecutionResult(
			"completed",
			profile,
			runtime_result["runtime_ms"],
			warmup_runtimes_ms=tuple(runtime_result["warmup_runtimes"]),
			measured_runtimes_ms=tuple(runtime_result["measured_runtimes"]),
		)

	def _write_config(self, config: dict[str, Any], param_hash: str) -> Path:
		path = self.config_dir / f"{self.kernel_type}_{self.input_size}_{param_hash}.json"
		if not path.exists():
			path.write_text(json.dumps(config, indent=2, sort_keys=True))
		return path

	def _ensure_experiment_row(self, db: ExperimentDB, config: dict[str, Any], param_hash: str) -> int:
		existing = db.get_experiment_by_key(self.kernel_type, self.input_size, param_hash)
		if existing is not None:
			return existing.exp_id
		exp_id = db.add_experiment(
			kernel_type=self.kernel_type,
			input_size=self.input_size,
			param_hash=param_hash,
			config_dict=config,
			skip_if_exists=True,
		)
		if exp_id is None:
			existing = db.get_experiment_by_key(self.kernel_type, self.input_size, param_hash)
			if existing is not None:
				return existing.exp_id
			raise RuntimeError("failed to create or find experiment row")
		return exp_id

	def _dump_opencl_binary_only(self, config_path: Path, binary_path: Path) -> subprocess.CompletedProcess[str]:
		exe_path = Path(self.build_dir) / self.kernel_type
		cmd = [
			str(exe_path),
			str(config_path),
			"--dump-opencl-binary-only",
			str(binary_path),
		]
		return subprocess.run(cmd, capture_output=True, text=True)

	def _run_kernel_with_warmups(self, config_path: Path, binary_path: Path) -> dict[str, Any]:
		warmup_results = []
		measured_results = []
		warmup_runtimes = []
		measured_runtimes = []

		for run_index in range(self.warmup_runs):
			result = run_kernel(
				self.kernel_type,
				str(config_path),
				build_dir=self.build_dir,
				dump_opencl_binary=str(binary_path) if run_index == 0 else None,
			)
			warmup_results.append(result)
			if result.success and result.runtime_ms is not None:
				warmup_runtimes.append(result.runtime_ms)

		for _ in range(self.runs_per_config):
			result = run_kernel(
				self.kernel_type,
				str(config_path),
				build_dir=self.build_dir,
			)
			measured_results.append(result)
			if result.success and result.runtime_ms is not None:
				measured_runtimes.append(result.runtime_ms)

		if not measured_runtimes:
			return {
				"runtime_ms": None,
				"error_msg": self._runtime_failure_message(warmup_results, measured_results),
				"warmup_results": tuple(warmup_results),
				"measured_results": tuple(measured_results),
				"warmup_runtimes": tuple(warmup_runtimes),
				"measured_runtimes": tuple(measured_runtimes),
			}

		return {
			"runtime_ms": statistics.median(measured_runtimes),
			"error_msg": "",
			"warmup_results": tuple(warmup_results),
			"measured_results": tuple(measured_results),
			"warmup_runtimes": tuple(warmup_runtimes),
			"measured_runtimes": tuple(measured_runtimes),
		}

	def _record_runtime_samples(
		self,
		db: ExperimentDB,
		exp_id: int,
		param_hash: str,
		config: dict[str, Any],
		warmup_results,
		measured_results,
		success: bool,
		average_runtime_ms: float | None,
	) -> None:
		def _payload(results):
			payload = []
			for index, result in enumerate(results, start=1):
				payload.append(
					{
						"run_index": index,
						"runtime_ms": result.runtime_ms,
						"success": result.success,
						"error_msg": result.error_msg,
						"returncode": result.returncode,
						"stdout_tail": result.stdout_tail,
						"stderr_tail": result.stderr_tail,
					}
				)
			return payload

		warmup_payload = _payload(warmup_results)
		measured_payload = _payload(measured_results)
		db.add_experiment_log(
			exp_id,
			stage="runtime_samples",
			level="info" if success else "warning",
			message=(
				f"Recorded runtime samples for {self.kernel_type}: "
				f"{sum(1 for row in measured_payload if row['success'])}/{len(measured_payload)} measured runs succeeded."
			),
			payload={
				"kernel_type": self.kernel_type,
				"input_size": self.input_size,
				"param_hash": param_hash,
				"config": config,
				"warmup_runs": self.warmup_runs,
				"measured_runs": self.runs_per_config,
				"runtime_statistic": "median",
				"average_runtime_ms": average_runtime_ms,
				"reported_runtime_ms": average_runtime_ms,
				"warmup_samples": warmup_payload,
				"measured_samples": measured_payload,
			},
		)
		self._append_runtime_record(
			exp_id=exp_id,
			param_hash=param_hash,
			config=config,
			average_runtime_ms=average_runtime_ms,
			warmup_samples=warmup_payload,
			measured_samples=measured_payload,
		)

	def _append_runtime_record(
		self,
		exp_id: int,
		param_hash: str,
		config: dict[str, Any],
		average_runtime_ms: float | None,
		warmup_samples: list[dict[str, Any]],
		measured_samples: list[dict[str, Any]],
	) -> None:
		path = self.output_dir / "runtime_samples.jsonl"
		record = {
			"exp_id": exp_id,
			"kernel_type": self.kernel_type,
			"input_size": self.input_size,
			"param_hash": param_hash,
			"config": config,
			"warmup_runs": self.warmup_runs,
			"measured_runs": self.runs_per_config,
			"runtime_statistic": "median",
			"average_runtime_ms": average_runtime_ms,
			"reported_runtime_ms": average_runtime_ms,
			"warmup_samples": warmup_samples,
			"measured_samples": measured_samples,
		}
		with path.open("a", encoding="utf-8") as f:
			f.write(json.dumps(record, sort_keys=True) + "\n")

	@staticmethod
	def _runtime_failure_message(warmup_results, measured_results) -> str:
		all_results = list(warmup_results) + list(measured_results)
		errors = []
		for result in all_results:
			if result.success:
				continue
			detail = result.error_msg or "unknown kernel failure"
			if result.stderr_tail:
				detail = f"{detail}; stderr: {result.stderr_tail}"
			elif result.stdout_tail:
				detail = f"{detail}; stdout: {result.stdout_tail}"
			errors.append(detail)
		if not errors:
			return "execution failed: no successful measured runs"
		return "execution failed: " + "; ".join(dict.fromkeys(errors))

	@staticmethod
	def _subprocess_error(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
		stderr = (result.stderr or "").strip()
		stdout = (result.stdout or "").strip()
		detail = stderr or stdout or "no output"
		return f"{prefix} (exit {result.returncode}): {detail[-2000:]}"

	@staticmethod
	def _cleanup_artifacts(llvm_paths: list[str], binary_path: Path) -> None:
		for llvm_path in llvm_paths:
			try:
				Path(llvm_path).unlink(missing_ok=True)
			except OSError:
				pass
		try:
			binary_path.unlink(missing_ok=True)
		except OSError:
			pass
