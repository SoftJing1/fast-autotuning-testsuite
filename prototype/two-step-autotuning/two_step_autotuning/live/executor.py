from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.collect_tuning_performance_simple import (
	_extract_runtime_ir_result,
	_serialize_llvm_paths,
	_store_symbolic_instruction_counts,
	run_kernel_average,
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
		runs_per_config: int = 3,
	):
		self.output_dir = Path(output_dir)
		self.kernel_type = kernel_type
		self.input_size = input_size
		self.build_dir = build_dir
		self.runs_per_config = runs_per_config
		self.config_dir = self.output_dir / "configs"
		self.llvm_dir = self.output_dir / "llvm_ir"
		self.binary_dir = self.output_dir / "opencl_binaries"
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
			return LiveExecutionResult("profiled", profile, profile.runtime_ms)

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
			)
			if not ir_result.success:
				error = ir_result.error_msg or "runtime LLVM IR extraction failed"
				db.mark_failed(exp_id, error)
				return LiveExecutionResult("failed", None, None, error)

			llvm_paths = [artifact["llvm_ir_path"] for artifact in ir_result.artifacts]
			db.mark_profiled(exp_id, _serialize_llvm_paths(llvm_paths))
			symbolic_errors = _store_symbolic_instruction_counts(db, exp_id, ir_result.artifacts)
			if symbolic_errors:
				error = "instruction-count extraction failed: " + " | ".join(symbolic_errors)
				db.mark_failed(exp_id, error)
				return LiveExecutionResult("failed", None, None, error)

			count_rows = db.get_llvm_instruction_counts(exp_id)
			raw_counts = sum_counts(row["total_instruction_counts"] for row in count_rows)

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
			return LiveExecutionResult("completed", profile, self._runtime_by_key[key])

		param_hash = hash_config(config)
		config_path = self._write_config(config, param_hash)
		binary_path = self.binary_dir / f"{self.kernel_type}_{self.input_size}_{param_hash}.opencl.bin"
		with ExperimentDB(str(self.db_path)) as db:
			exp_id = self._ensure_experiment_row(db, config, param_hash)
			runtime_result = run_kernel_average(
				self.kernel_type,
				str(config_path),
				build_dir=self.build_dir,
				runs_per_config=self.runs_per_config,
				dump_opencl_binary=str(binary_path),
			)
			if not runtime_result.success or runtime_result.runtime_ms is None:
				error = f"execution failed: {runtime_result.error_summary()}"
				db.mark_failed(exp_id, error)
				return LiveExecutionResult("failed", None, None, error)

			ir_result = _extract_runtime_ir_result(
				self.kernel_type,
				config,
				param_hash,
				self.llvm_dir,
				binary_path,
			)
			if not ir_result.success:
				error = ir_result.error_msg or "runtime LLVM IR extraction failed"
				db.mark_failed(exp_id, error)
				return LiveExecutionResult("failed", None, None, error)

			llvm_paths = [artifact["llvm_ir_path"] for artifact in ir_result.artifacts]
			db.mark_completed(exp_id, runtime_result.runtime_ms, _serialize_llvm_paths(llvm_paths))
			symbolic_errors = _store_symbolic_instruction_counts(db, exp_id, ir_result.artifacts)
			if symbolic_errors:
				db.record_error_message(
					exp_id,
					"instruction-count extraction failed: " + " | ".join(symbolic_errors),
				)
			count_rows = db.get_llvm_instruction_counts(exp_id)
			raw_counts = sum_counts(row["total_instruction_counts"] for row in count_rows)

		profile = LiveProfile(
			exp_id=exp_id,
			kernel_type=self.kernel_type,
			input_size=self.input_size,
			param_hash=param_hash,
			config=dict(config),
			raw_counts=raw_counts,
			runtime_ms=runtime_result.runtime_ms,
		)
		self._profiles_by_key[key] = profile
		self._runtime_by_key[key] = runtime_result.runtime_ms
		return LiveExecutionResult("completed", profile, runtime_result.runtime_ms)

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

	@staticmethod
	def _subprocess_error(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
		stderr = (result.stderr or "").strip()
		stdout = (result.stdout or "").strip()
		detail = stderr or stdout or "no output"
		return f"{prefix} (exit {result.returncode}): {detail[-2000:]}"
