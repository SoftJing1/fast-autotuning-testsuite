#!/usr/bin/env python3
"""Collect tuning performance with request-driven config generation.

Workflow:
1. User defines one or more experiment requests as:
   (kernel type, input size, total configs)
2. Script creates an experiment directory
3. Script calls internal config generation APIs to generate requested configs
4. Script runs kernels, collects runtime, and emits LLVM IR
5. Script stores DB + generated artifacts under experiment directory

Usage examples:
    python -m scripts.collect_tuning_performance_simple \
      --request gemm 256x256x256 100 \
      --request gaussian 1024x1024 80

    python -m scripts.collect_tuning_performance_simple \
      --request gemm 256x256x256 50 \
      --experiment-name my_run --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
	from scripts.internal.db_manager import ExperimentDB
except ModuleNotFoundError as exc:
	if exc.name not in {"scripts", "scripts.internal", "scripts.internal.db_manager"}:
		raise
	from internal.db_manager import ExperimentDB

try:
	from scripts.internal.llvm_to_inst_count import extract_instruction_counts
except ModuleNotFoundError as exc:
	if exc.name not in {"scripts", "scripts.internal", "scripts.internal.llvm_to_inst_count"}:
		raise
	from internal.llvm_to_inst_count import extract_instruction_counts

try:
	from scripts.internal.runtime_opencl_ir import extract_opencl_runtime_ir
except ModuleNotFoundError as exc:
	if exc.name not in {"scripts", "scripts.internal", "scripts.internal.runtime_opencl_ir"}:
		raise
	from internal.runtime_opencl_ir import extract_opencl_runtime_ir

DEFAULT_SEED = 42
DEFAULT_EXPERIMENT_ROOT = "experiments"
DEFAULT_DB_FILENAME = "experiments.db"
DEFAULT_BUILD_DIR = "build"
DEFAULT_RUNS_PER_CONFIG = 3
DEFAULT_PERSIST_ARTIFACTS = False
MAX_ERROR_MESSAGE_LEN = 4000

KERNEL_TEMPLATE_SPECS = {
	"gemm": [
		{
			"template_name": "gemm_1",
			"kernel_function": "gemm_1",
			"template_path": "kernels/kernel-template/gemm_1.cl",
		},
		{
			"template_name": "gemm_2",
			"kernel_function": "gemm_2",
			"template_path": "kernels/kernel-template/gemm_2.cl",
		},
	],
	"gaussian": [
		{
			"template_name": "gaussian_static_1",
			"kernel_function": "gaussian_1",
			"template_path": "kernels/kernel-template/gaussian_static_1.cl",
		},
	],
}


@dataclass(frozen=True)
class ExperimentRequest:
	"""A user request tuple: (kernel type, input size, total configs)."""
	kernel_type: str
	input_size: Tuple[int, ...]
	total_configs: int

	@property
	def size_string(self) -> str:
		return "x".join(str(v) for v in self.input_size)


@dataclass(frozen=True)
class KernelRunResult:
	"""Detailed outcome from one host-kernel invocation."""
	runtime_ms: Optional[float]
	error_msg: str = ""
	returncode: Optional[int] = None
	stdout_tail: str = ""
	stderr_tail: str = ""

	@property
	def success(self) -> bool:
		return self.runtime_ms is not None and not self.error_msg


@dataclass(frozen=True)
class KernelAverageResult:
	"""Detailed outcome from repeated host-kernel invocations."""
	runtime_ms: Optional[float]
	successful_runs: int
	attempted_runs: int
	errors: Tuple[str, ...] = ()
	run_results: Tuple[KernelRunResult, ...] = ()

	@property
	def success(self) -> bool:
		return self.runtime_ms is not None

	def error_summary(self) -> str:
		if not self.errors:
			return "no errors"
		deduped = list(dict.fromkeys(self.errors))
		return "; ".join(deduped)


@dataclass(frozen=True)
class SchedulingResult:
	"""Experiment work items plus count of configs that could not be scheduled."""
	experiments: List[Tuple[int, str, str, str, str, str, int, bool]]
	error_count: int


@dataclass(frozen=True)
class RuntimeIRResult:
	"""Runtime IR extraction result for one experiment."""
	artifacts: List[Dict[str, str]]
	error_msg: Optional[str] = None

	@property
	def success(self) -> bool:
		return self.error_msg is None and bool(self.artifacts)


def hash_config(config_dict: Dict) -> str:
	"""Generate deterministic hash of configuration."""
	config_str = json.dumps(config_dict, sort_keys=True)
	return hashlib.md5(config_str.encode()).hexdigest()[:16]


def _tail_text(value: object, limit: int = 1000) -> str:
	if value is None:
		return ""
	if isinstance(value, bytes):
		text = value.decode("utf-8", errors="replace")
	else:
		text = str(value)
	text = text.strip()
	if len(text) <= limit:
		return text
	return text[-limit:]


def _truncate_error(message: str, limit: int = MAX_ERROR_MESSAGE_LEN) -> str:
	message = message.strip()
	if len(message) <= limit:
		return message
	return message[: limit - 3] + "..."


def run_kernel(
	kernel_type: str,
	config_file: str,
	build_dir: str,
	dump_opencl_binary: Optional[str] = None,
) -> KernelRunResult:
	"""Run kernel with configuration and return a detailed outcome."""
	try:
		exe_path = os.path.join(build_dir, kernel_type)
		if not os.path.exists(exe_path):
			return KernelRunResult(
				runtime_ms=None,
				error_msg=f"Executable not found: {exe_path}",
			)

		if not os.path.exists(config_file):
			return KernelRunResult(
				runtime_ms=None,
				error_msg=f"Config file not found: {config_file}",
			)

		cmd = [exe_path, config_file]
		if dump_opencl_binary:
			Path(dump_opencl_binary).parent.mkdir(parents=True, exist_ok=True)
			cmd.extend(["--dump-opencl-binary", dump_opencl_binary])

		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
		)

		if result.returncode != 0:
			return KernelRunResult(
				runtime_ms=None,
				error_msg=_truncate_error(
					f"Kernel command failed with exit code {result.returncode}: {' '.join(cmd)}"
				),
				returncode=result.returncode,
				stdout_tail=_tail_text(result.stdout),
				stderr_tail=_tail_text(result.stderr),
			)

		match = re.search(r"Device kernel execution time:\s*(\d+\.?\d*)\s*ms", result.stdout, re.IGNORECASE)
		if match:
			return KernelRunResult(
				runtime_ms=float(match.group(1)),
				returncode=result.returncode,
				stdout_tail=_tail_text(result.stdout),
				stderr_tail=_tail_text(result.stderr),
			)

		return KernelRunResult(
			runtime_ms=None,
			error_msg="Runtime marker not found in kernel stdout",
			returncode=result.returncode,
			stdout_tail=_tail_text(result.stdout),
			stderr_tail=_tail_text(result.stderr),
		)

	except OSError as exc:
		return KernelRunResult(runtime_ms=None, error_msg=f"Kernel launch failed: {exc}")
	except Exception as exc:
		return KernelRunResult(runtime_ms=None, error_msg=f"Unexpected kernel error: {exc}")


def run_kernel_average(
	kernel_type: str,
	config_file: str,
	build_dir: str,
	runs_per_config: int,
	dump_opencl_binary: Optional[str] = None,
) -> KernelAverageResult:
	"""Run kernel multiple times and return a detailed average-runtime outcome."""
	if runs_per_config <= 0:
		raise ValueError("runs_per_config must be > 0")

	runtimes: List[float] = []
	errors: List[str] = []
	run_results: List[KernelRunResult] = []
	for run_index in range(1, runs_per_config + 1):
		result = run_kernel(
			kernel_type,
			config_file,
			build_dir=build_dir,
			dump_opencl_binary=dump_opencl_binary,
		)
		run_results.append(result)
		if result.success and result.runtime_ms is not None:
			runtimes.append(result.runtime_ms)
			continue

		details = result.error_msg or "unknown kernel failure"
		if result.stderr_tail:
			details = f"{details}; stderr: {result.stderr_tail}"
		elif result.stdout_tail:
			details = f"{details}; stdout: {result.stdout_tail}"
		errors.append(_truncate_error(f"run {run_index}: {details}"))

	if not runtimes:
		return KernelAverageResult(
			runtime_ms=None,
			successful_runs=0,
			attempted_runs=runs_per_config,
			errors=tuple(errors),
			run_results=tuple(run_results),
		)

	return KernelAverageResult(
		runtime_ms=sum(runtimes) / len(runtimes),
		successful_runs=len(runtimes),
		attempted_runs=runs_per_config,
		errors=tuple(errors),
		run_results=tuple(run_results),
	)


def _kernel_size_string(kernel_type: str, config_dict: Dict) -> str:
	if kernel_type == "gemm":
		return f"{config_dict.get('M', 0)}x{config_dict.get('N', 0)}x{config_dict.get('K', 0)}"
	return f"{config_dict.get('input_size_h', 0)}x{config_dict.get('input_size_w', 0)}"


def _runtime_kernel_function(llvm_ir_path: Path, kernel_function: str) -> str:
	"""Prefer Clang's generated implementation body over the tiny exported wrapper."""
	try:
		llvm_text = llvm_ir_path.read_text(errors="ignore")
	except OSError:
		return kernel_function
	impl_name = f"__clang_ocl_kern_imp_{kernel_function}"
	if re.search(rf"^define\s+.*@{re.escape(impl_name)}\(", llvm_text, flags=re.MULTILINE):
		return impl_name
	return kernel_function


def runtime_ir_artifacts_from_dump(
	kernel_type: str,
	config_dict: Dict,
	param_hash: str,
	llvm_output_dir: Path,
	opencl_binary_path: Path,
	persist_artifacts: bool,
) -> List[Dict[str, str]]:
	"""Convert a host-dumped OpenCL binary into runtime LLVM IR artifact records."""
	size_str = _kernel_size_string(kernel_type, config_dict)
	output_ll = llvm_output_dir / f"{kernel_type}_{size_str}_{param_hash}_runtime.ll"
	extract_opencl_runtime_ir(opencl_binary_path, output_ll)
	artifact_ref = str(output_ll) if persist_artifacts else f"transient://{output_ll.name}"

	artifacts: List[Dict[str, str]] = []
	for spec in KERNEL_TEMPLATE_SPECS.get(kernel_type, []):
		kernel_function = _runtime_kernel_function(output_ll, spec["kernel_function"])
		artifacts.append(
			{
				"template_name": spec["template_name"],
				"kernel_function": kernel_function,
				"llvm_ir_path": str(output_ll),
				"llvm_ir_ref": artifact_ref,
			}
		)
	return artifacts


def compile_to_llvm_ir(
	kernel_type: str,
	config_dict: Dict,
	param_hash: str,
	llvm_output_dir: Path,
) -> List[Dict[str, str]]:
	"""Compile kernel configuration to one or more LLVM IR files under experiment directory."""
	try:
		from scripts.internal.compile_to_llvm import compile_opencl_to_llvm
	except ModuleNotFoundError as exc:
		if exc.name not in {"scripts", "scripts.internal", "scripts.internal.compile_to_llvm"}:
			raise
		from internal.compile_to_llvm import compile_opencl_to_llvm

	try:
		import tempfile

		with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
			json.dump(config_dict, f)
			temp_config = f.name

		try:
			if kernel_type == "gemm":
				size_str = f"{config_dict.get('M', 0)}x{config_dict.get('N', 0)}x{config_dict.get('K', 0)}"
			else:
				size_str = f"{config_dict.get('input_size_h', 0)}x{config_dict.get('input_size_w', 0)}"

			template_specs = KERNEL_TEMPLATE_SPECS.get(kernel_type, [])
			llvm_output_dir.mkdir(parents=True, exist_ok=True)
			artifacts: List[Dict[str, str]] = []
			use_template_suffix = len(template_specs) > 1

			for spec in template_specs:
				template_path = spec["template_path"]
				if not os.path.exists(template_path):
					continue

				filename = f"{kernel_type}_{size_str}_{param_hash}.ll"
				if use_template_suffix:
					filename = f"{kernel_type}_{size_str}_{param_hash}_{spec['template_name']}.ll"
				output_ll = llvm_output_dir / filename

				success = compile_opencl_to_llvm(
					template_path,
					temp_config,
					str(output_ll),
				)
				if success and output_ll.exists():
					artifacts.append(
						{
							"template_name": spec["template_name"],
							"kernel_function": spec["kernel_function"],
							"llvm_ir_path": str(output_ll),
						}
					)

			return artifacts
		finally:
			if os.path.exists(temp_config):
				os.unlink(temp_config)
	except Exception:
		return []


def _serialize_llvm_paths(llvm_paths: Sequence[str]) -> Optional[str]:
	deduped = list(dict.fromkeys(llvm_paths))
	if not deduped:
		return None
	if len(deduped) == 1:
		return deduped[0]
	return json.dumps(deduped)


def _parse_request(kernel_type: str, input_size: str, total_configs: str) -> ExperimentRequest:
	kernel = kernel_type.strip().lower()
	if kernel not in {"gemm", "gaussian"}:
		raise ValueError(f"Unsupported kernel type: {kernel_type}")

	try:
		count = int(total_configs)
	except ValueError as exc:
		raise ValueError(f"Invalid total config count: {total_configs}") from exc
	if count <= 0:
		raise ValueError("total config count must be > 0")

	parts = input_size.lower().split("x")
	if kernel == "gemm":
		if len(parts) != 3:
			raise ValueError(f"GEMM input size must be MxNxK, got: {input_size}")
		dims = tuple(int(v) for v in parts)
	else:
		if len(parts) != 2:
			raise ValueError(f"Gaussian input size must be HxW, got: {input_size}")
		dims = tuple(int(v) for v in parts)

	if any(v <= 0 for v in dims):
		raise ValueError(f"Input dimensions must be > 0, got: {input_size}")

	return ExperimentRequest(kernel_type=kernel, input_size=dims, total_configs=count)


def _load_module_from_path(module_name: str, module_path: Path):
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Failed to load module spec from {module_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _load_config_generator_modules() -> Dict[str, object]:
	base = Path(__file__).resolve().parent / "internal"
	return {
		"gemm": _load_module_from_path("config_gen_gemm", base / "config-gen-gemm.py"),
		"gaussian": _load_module_from_path("config_gen_gaussian", base / "config-gen-gaussian.py"),
	}


def _device_limits(device_type: str) -> Tuple[Tuple[int, int, int], int]:
	if device_type == "cpu":
		return (8192, 8192, 8192), 8192
	return (1024, 1024, 64), 1024


def _request_output_dir(base_configs_dir: Path, request: ExperimentRequest) -> Path:
	return base_configs_dir / request.kernel_type / request.size_string


def _load_existing_configs(config_dir: Path, request: ExperimentRequest) -> List[Path]:
	prefix = f"{request.kernel_type}_{request.size_string}_"
	return sorted(config_dir.glob(f"{prefix}*.json"))


def _generate_request_configs(
	request: ExperimentRequest,
	config_dir: Path,
	generator_module,
	device_type: str,
	seed: int,
) -> List[Path]:
	config_dir.mkdir(parents=True, exist_ok=True)
	existing_files = _load_existing_configs(config_dir, request)

	if len(existing_files) >= request.total_configs:
		return existing_files[:request.total_configs]

	needed = request.total_configs - len(existing_files)
	max_wi_size, max_wg_size = _device_limits(device_type)

	seed_key = f"{request.kernel_type}:{request.size_string}".encode("utf-8")
	stable_offset = int(hashlib.md5(seed_key).hexdigest()[:8], 16) % 100000
	request_seed = seed + stable_offset

	if request.kernel_type == "gemm":
		m, n, k = request.input_size
		config_iter: Iterator[Dict] = generator_module.random_sample_configurations(
			m,
			n,
			k,
			needed,
			max_wi_size=max_wi_size,
			max_wg_size=max_wg_size,
			seed=request_seed,
		)
	else:
		h, w = request.input_size
		config_iter = generator_module.random_sample_configurations(
			h,
			w,
			needed,
			max_wi_size=max_wi_size,
			max_wg_size=max_wg_size,
			seed=request_seed,
		)

	generated = 0
	for offset, config in enumerate(config_iter, start=len(existing_files)):
		output_file = config_dir / f"{request.kernel_type}_{request.size_string}_{offset:06d}.json"
		if request.kernel_type == "gemm":
			m, n, k = request.input_size
			saved = generator_module.save_tuning_parameters_only(
				config,
				str(output_file),
				device_type=device_type,
				M=m,
				N=n,
				K=k,
			)
		else:
			h, w = request.input_size
			saved = generator_module.save_tuning_parameters_only(
				config,
				str(output_file),
				device_type=device_type,
				input_size_h=h,
				input_size_w=w,
			)
		if saved:
			generated += 1

	all_files = _load_existing_configs(config_dir, request)
	if len(all_files) < request.total_configs:
		print(
			f"Warning: requested {request.total_configs} configs for "
			f"{request.kernel_type}/{request.size_string}, got {len(all_files)}",
			file=sys.stderr,
		)

	if generated > 0:
		print(
			f"Generated {generated} new configs for "
			f"{request.kernel_type}/{request.size_string} in {config_dir}"
		)

	return all_files[:request.total_configs]


def prepare_experiment_configs(
	requests: Sequence[ExperimentRequest],
	configs_root: Path,
	device_type: str,
	seed: int,
) -> List[Tuple[ExperimentRequest, Path]]:
	"""Generate/load configs for all requests and return config file list."""
	modules = _load_config_generator_modules()
	resolved: List[Tuple[ExperimentRequest, Path]] = []

	for request in requests:
		config_dir = _request_output_dir(configs_root, request)
		try:
			request_files = _generate_request_configs(
				request=request,
				config_dir=config_dir,
				generator_module=modules[request.kernel_type],
				device_type=device_type,
				seed=seed,
			)
		except Exception as exc:
			print(
				f"Warning: failed to prepare configs for "
				f"{request.kernel_type}/{request.size_string}: {_format_exception(exc)}",
				file=sys.stderr,
			)
			continue
		resolved.extend((request, path) for path in request_files)

	return resolved


def _format_exception(exc: BaseException) -> str:
	return _truncate_error(f"{type(exc).__name__}: {exc}")


def _schedule_single_experiment(
	db: ExperimentDB,
	request: ExperimentRequest,
	config_file: Path,
	db_path: Path,
	llvm_root: Path,
	build_dir: str,
	runs_per_config: int,
	resume: bool,
	persist_artifacts: bool,
) -> Optional[Tuple[int, str, str, str, str, str, int, bool]]:
	with open(config_file) as f:
		config = json.load(f)

	param_hash = hash_config(config)
	existing = db.get_experiment_by_key(
		kernel_type=request.kernel_type,
		input_size=request.size_string,
		param_hash=param_hash,
	)
	exp_id = db.add_experiment(
		kernel_type=request.kernel_type,
		input_size=request.size_string,
		param_hash=param_hash,
		config_dict=config,
		skip_if_exists=resume,
	)

	if exp_id is None:
		if not resume or existing is None or existing.status != "pending":
			return None
		exp_id = existing.exp_id

	return (
		exp_id,
		request.kernel_type,
		str(config_file),
		str(db_path),
		str(llvm_root),
		build_dir,
		runs_per_config,
		persist_artifacts,
	)


def schedule_experiment_work(
	resolved_configs: Sequence[Tuple[ExperimentRequest, Path]],
	db_path: Path,
	llvm_root: Path,
	build_dir: str,
	runs_per_config: int,
	resume: bool,
	persist_artifacts: bool,
) -> SchedulingResult:
	"""Create DB rows and worker arguments for configs that are ready to run."""
	experiments: List[Tuple[int, str, str, str, str, str, int, bool]] = []
	error_count = 0

	with ExperimentDB(str(db_path)) as db:
		for request, config_file in resolved_configs:
			try:
				work_item = _schedule_single_experiment(
					db=db,
					request=request,
					config_file=config_file,
					db_path=db_path,
					llvm_root=llvm_root,
					build_dir=build_dir,
					runs_per_config=runs_per_config,
					resume=resume,
					persist_artifacts=persist_artifacts,
				)
			except Exception as exc:
				error_count += 1
				print(
					f"Warning: failed to schedule config {config_file}: {_format_exception(exc)}",
					file=sys.stderr,
				)
				continue

			if work_item is not None:
				experiments.append(work_item)

	return SchedulingResult(experiments=experiments, error_count=error_count)


def _record_db_log(
	db: ExperimentDB,
	exp_id: int,
	stage: str,
	level: str,
	message: str,
	payload: Optional[Dict[str, object]] = None,
) -> None:
	try:
		db.add_experiment_log(
			exp_id=exp_id,
			stage=stage,
			level=level,
			message=_truncate_error(message),
			payload=payload,
		)
	except Exception as exc:
		print(
			f"Warning: failed to store log for exp_id={exp_id}: {_format_exception(exc)}",
			file=sys.stderr,
		)


def _kernel_run_result_payload(run_index: int, result: KernelRunResult) -> Dict[str, object]:
	return {
		"run_index": run_index,
		"runtime_ms": result.runtime_ms,
		"success": result.success,
		"error_msg": result.error_msg,
		"returncode": result.returncode,
		"stdout_tail": result.stdout_tail,
		"stderr_tail": result.stderr_tail,
	}


def _record_runtime_execution_summary(
	db: ExperimentDB,
	exp_id: int,
	kernel_type: str,
	config_file: str,
	build_dir: str,
	runs_per_config: int,
	runtime_result: KernelAverageResult,
	persist_artifacts: bool,
) -> None:
	run_payload = [
		_kernel_run_result_payload(run_index, result)
		for run_index, result in enumerate(runtime_result.run_results, start=1)
	]
	message = (
		f"Host execution summary for {kernel_type}: "
		f"{runtime_result.successful_runs}/{runtime_result.attempted_runs} successful run(s)."
	)
	_record_db_log(
		db,
		exp_id,
		"kernel_host",
		"info" if runtime_result.success else "warning",
		message,
		payload={
			"kernel_type": kernel_type,
			"config_file": config_file,
			"build_dir": build_dir,
			"runs_per_config": runs_per_config,
			"successful_runs": runtime_result.successful_runs,
			"attempted_runs": runtime_result.attempted_runs,
			"average_runtime_ms": runtime_result.runtime_ms,
			"errors": list(runtime_result.errors),
			"persist_artifacts": persist_artifacts,
			"runs": run_payload,
		},
	)


def _worker_failure(
	db: ExperimentDB,
	exp_id: int,
	error_msg: str,
	stage: str,
	payload: Optional[Dict[str, object]] = None,
) -> Tuple[int, bool, str]:
	error_msg = _truncate_error(error_msg)
	_record_db_log(db, exp_id, stage, "error", error_msg, payload=payload)
	db.mark_failed(exp_id, error_msg)
	return (exp_id, False, error_msg)


def _mark_failed_from_exception(
	db_path: str,
	exp_id: int,
	error_msg: str,
	stage: str = "collector",
	payload: Optional[Dict[str, object]] = None,
) -> None:
	try:
		with ExperimentDB(db_path) as db:
			_record_db_log(db, exp_id, stage, "error", error_msg, payload=payload)
			db.mark_failed(exp_id, error_msg)
	except Exception:
		pass


def _extract_runtime_ir_result(
	kernel_type: str,
	config: Dict,
	param_hash: str,
	llvm_output_dir: Path,
	opencl_binary_path: Path,
	persist_artifacts: bool,
) -> RuntimeIRResult:
	try:
		artifacts = runtime_ir_artifacts_from_dump(
			kernel_type,
			config,
			param_hash,
			llvm_output_dir,
			opencl_binary_path,
			persist_artifacts=persist_artifacts,
		)
	except Exception as exc:
		return RuntimeIRResult(
			artifacts=[],
			error_msg=_truncate_error(f"Runtime LLVM IR extraction failed: {_format_exception(exc)}"),
		)

	if not artifacts:
		return RuntimeIRResult(
			artifacts=[],
			error_msg="Runtime LLVM IR extraction produced no artifacts",
		)

	return RuntimeIRResult(artifacts=artifacts)


def _store_symbolic_instruction_counts(
	db: ExperimentDB,
	exp_id: int,
	llvm_artifacts: Sequence[Dict[str, str]],
) -> List[str]:
	errors: List[str] = []
	for artifact in llvm_artifacts:
		try:
			inst_result = extract_instruction_counts(
				artifact["llvm_ir_path"],
				kernel_function=artifact["kernel_function"],
			)
			db.upsert_llvm_instruction_counts(
				exp_id=exp_id,
				template_name=artifact["template_name"],
				kernel_function=artifact["kernel_function"],
				llvm_ir_path=artifact["llvm_ir_ref"],
				bb_counts=inst_result.bb_counts,
				bb_instruction_counts=inst_result.bb_instruction_counts,
				total_instruction_counts=inst_result.total_instruction_counts,
			)
			for invocation in inst_result.symb_viewer_invocations:
				_record_db_log(
					db,
					exp_id,
					"symb_viewer",
					"info",
					f"symb-viewer completed for template={artifact['template_name']} command={' '.join(invocation.command)}",
					payload={
						"template_name": artifact["template_name"],
						"kernel_function": artifact["kernel_function"],
						"llvm_ir_ref": artifact["llvm_ir_ref"],
						"command": invocation.command,
						"returncode": invocation.returncode,
						"stdout": invocation.stdout,
						"stderr": invocation.stderr,
					},
				)
			_record_db_log(
				db,
				exp_id,
				"symb_viewer_json",
				"info",
				f"Stored raw symb-viewer inst-count JSON for template={artifact['template_name']}",
				payload={
					"template_name": artifact["template_name"],
					"kernel_function": artifact["kernel_function"],
					"llvm_ir_ref": artifact["llvm_ir_ref"],
					"source_file": "instrcount.json",
					"json_payload": inst_result.raw_instr_count_json,
				},
			)
			_record_db_log(
				db,
				exp_id,
				"symb_viewer_json",
				"info",
				f"Stored raw symb-viewer bb-count JSON for template={artifact['template_name']}",
				payload={
					"template_name": artifact["template_name"],
					"kernel_function": artifact["kernel_function"],
					"llvm_ir_ref": artifact["llvm_ir_ref"],
					"source_file": "bbcount.json",
					"json_payload": inst_result.raw_bb_count_json,
				},
			)
		except Exception as exc:
			errors.append(
				_truncate_error(
					f"template={artifact['template_name']}, "
					f"kernel_function={artifact['kernel_function']}: {_format_exception(exc)}"
				)
			)
	return errors


def _record_worker_diagnostics(
	db: ExperimentDB,
	exp_id: int,
	runtime_result: KernelAverageResult,
	symbolic_errors: Sequence[str],
	artifact_count: int,
	persist_artifacts: bool,
) -> None:
	diagnostics: List[str] = []

	if runtime_result.errors:
		message = (
			f"Kernel produced {runtime_result.successful_runs}/"
			f"{runtime_result.attempted_runs} successful run(s); ignored failed runs."
		)
		payload = {
			"successful_runs": runtime_result.successful_runs,
			"attempted_runs": runtime_result.attempted_runs,
			"errors": list(runtime_result.errors),
		}
		_record_db_log(db, exp_id, "runtime", "warning", message, payload=payload)
		diagnostics.append(f"{message} {runtime_result.error_summary()}")

	if symbolic_errors:
		message = (
			"Symbolic instruction-count extraction failed for "
			f"{len(symbolic_errors)}/{artifact_count} artifact(s)."
		)
		payload = {
			"artifact_count": artifact_count,
			"errors": list(symbolic_errors),
			"persist_artifacts": persist_artifacts,
		}
		_record_db_log(db, exp_id, "symbolic_counts", "warning", message, payload=payload)
		diagnostics.append(message + " " + " | ".join(symbolic_errors))

	if diagnostics:
		warning = _truncate_error(" ".join(diagnostics))
		try:
			db.record_error_message(exp_id, warning)
		except Exception as exc:
			print(
				f"Warning: failed to store diagnostic for exp_id={exp_id}: {_format_exception(exc)}",
				file=sys.stderr,
			)
		print(f"Warning: exp_id={exp_id}: {warning}", file=sys.stderr)


def process_single_experiment(args: Tuple[int, str, str, str, str, str, int, bool]) -> Tuple[int, bool, str]:
	"""Process single experiment in worker process."""
	(
		exp_id,
		kernel_type,
		config_file,
		db_path,
		llvm_output_dir,
		build_dir,
		runs_per_config,
		persist_artifacts,
	) = args

	artifact_tempdir: Optional[tempfile.TemporaryDirectory[str]] = None
	try:
		with ExperimentDB(db_path) as db:
			with open(config_file) as f:
				config = json.load(f)

			param_hash = hash_config(config)
			size_str = _kernel_size_string(kernel_type, config)
			if persist_artifacts:
				artifact_root = Path(llvm_output_dir)
			else:
				artifact_tempdir = tempfile.TemporaryDirectory(prefix="collect_artifacts_")
				artifact_root = Path(artifact_tempdir.name)
			opencl_binary_path = artifact_root / f"{kernel_type}_{size_str}_{param_hash}.opencl.bin"

			runtime_result = run_kernel_average(
				kernel_type,
				config_file,
				build_dir=build_dir,
				runs_per_config=runs_per_config,
				dump_opencl_binary=str(opencl_binary_path),
			)
			_record_runtime_execution_summary(
				db=db,
				exp_id=exp_id,
				kernel_type=kernel_type,
				config_file=config_file,
				build_dir=build_dir,
				runs_per_config=runs_per_config,
				runtime_result=runtime_result,
				persist_artifacts=persist_artifacts,
			)
			if not runtime_result.success or runtime_result.runtime_ms is None:
				return _worker_failure(
					db,
					exp_id,
					f"Execution failed across {runs_per_config} run(s): "
					f"{runtime_result.error_summary()}",
					stage="runtime",
					payload={
						"successful_runs": runtime_result.successful_runs,
						"attempted_runs": runtime_result.attempted_runs,
						"errors": list(runtime_result.errors),
					},
				)

			ir_result = _extract_runtime_ir_result(
				kernel_type,
				config,
				param_hash,
				artifact_root,
				opencl_binary_path,
				persist_artifacts=persist_artifacts,
			)
			if not ir_result.success:
				return _worker_failure(
					db,
					exp_id,
					ir_result.error_msg or "Runtime LLVM IR extraction failed",
					stage="runtime_ir",
					payload={"persist_artifacts": persist_artifacts},
				)
			_record_db_log(
				db,
				exp_id,
				"runtime_ir",
				"info",
				f"Runtime IR extraction produced {len(ir_result.artifacts)} artifact(s).",
				payload={
					"persist_artifacts": persist_artifacts,
					"artifacts": [
						{
							"template_name": artifact["template_name"],
							"kernel_function": artifact["kernel_function"],
							"llvm_ir_ref": artifact["llvm_ir_ref"],
						}
						for artifact in ir_result.artifacts
					],
				},
			)

			db.mark_completed(
				exp_id,
				runtime_result.runtime_ms,
				_serialize_llvm_paths([artifact["llvm_ir_ref"] for artifact in ir_result.artifacts])
				if persist_artifacts else None,
			)

			symbolic_errors = _store_symbolic_instruction_counts(db, exp_id, ir_result.artifacts)
			_record_worker_diagnostics(
				db=db,
				exp_id=exp_id,
				runtime_result=runtime_result,
				symbolic_errors=symbolic_errors,
				artifact_count=len(ir_result.artifacts),
				persist_artifacts=persist_artifacts,
			)
			_record_db_log(
				db,
				exp_id,
				"collector",
				"info",
				"Collected runtime and symbolic instruction counts.",
				payload={
					"runtime_ms": runtime_result.runtime_ms,
					"successful_runs": runtime_result.successful_runs,
					"attempted_runs": runtime_result.attempted_runs,
					"artifact_count": len(ir_result.artifacts),
					"persist_artifacts": persist_artifacts,
					"templates": [artifact["template_name"] for artifact in ir_result.artifacts],
				},
			)

		return (exp_id, True, "")
	except Exception as exc:
		error_msg = _format_exception(exc)
		_mark_failed_from_exception(
			db_path,
			exp_id,
			error_msg,
			stage="collector",
			payload={"persist_artifacts": persist_artifacts},
		)
		return (exp_id, False, error_msg)
	finally:
		if artifact_tempdir is not None:
			artifact_tempdir.cleanup()


def _build_experiment_directory(
	root_dir: str,
	experiment_name: Optional[str],
	resume: bool,
	persist_artifacts: bool,
) -> Path:
	root = Path(root_dir)
	root.mkdir(parents=True, exist_ok=True)
	name = experiment_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
	run_dir = root / name
	if run_dir.exists() and not resume:
		raise RuntimeError(
			f"Experiment directory already exists: {run_dir}. Use --resume or choose --experiment-name."
		)
	run_dir.mkdir(parents=True, exist_ok=True)
	(run_dir / "configs").mkdir(parents=True, exist_ok=True)
	if persist_artifacts:
		(run_dir / "llvm_ir").mkdir(parents=True, exist_ok=True)
	_ensure_visualization_launcher(run_dir)
	return run_dir


def _ensure_visualization_launcher(run_dir: Path) -> None:
	"""Create an easy launcher inside experiment dir for the web visualizer."""
	launcher_path = run_dir / "run_visualization.py"
	launcher_code = """#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
	current = Path(__file__).resolve()
	repo_root = current.parents[2]
	internal_script = repo_root / \"scripts\" / \"internal\" / \"visualize_results_web.py\"
	if not internal_script.exists():
		print(f\"Internal visualizer script not found: {internal_script}\", file=sys.stderr)
		return 1

	default_db = current.parent / \"experiments.db\"
	sys.argv = [str(internal_script), \"--db\", str(default_db), *sys.argv[1:]]
	runpy.run_path(str(internal_script), run_name=\"__main__\")
	return 0


if __name__ == \"__main__\":
	raise SystemExit(main())
"""

	launcher_path.write_text(launcher_code)
	os.chmod(launcher_path, 0o755)


def collect_experiments(
	requests: Sequence[ExperimentRequest],
	num_workers: Optional[int],
	resume: bool,
	experiment_root: str,
	experiment_name: Optional[str],
	device_type: str,
	seed: int,
	build_dir: str,
	runs_per_config: int,
	persist_artifacts: bool,
) -> int:
	"""Main collection pipeline from request tuples."""
	if num_workers is None:
		num_workers = max(1, cpu_count() - 1)
	if num_workers <= 0:
		raise ValueError("num_workers must be > 0")
	if runs_per_config <= 0:
		raise ValueError("runs_per_config must be > 0")

	run_dir = _build_experiment_directory(
		experiment_root,
		experiment_name,
		resume=resume,
		persist_artifacts=persist_artifacts,
	)
	configs_root = run_dir / "configs"
	llvm_root = run_dir / "llvm_ir"
	db_path = run_dir / DEFAULT_DB_FILENAME

	print("\nTuning Collection System")
	print("=" * 70)
	print(f"Experiment dir: {run_dir}")
	print(f"Database:       {db_path}")
	print(f"Configs root:   {configs_root}")
	print(f"Artifact store: {llvm_root if persist_artifacts else 'transient worker tempdirs only'}")
	print(f"Build dir:      {build_dir}")
	print(f"Runs/config:    {runs_per_config}")
	print(f"Workers:        {num_workers}")
	print(f"Resume:         {resume}")
	print(f"Persist files:  {persist_artifacts}")
	print("Requests:")
	for req in requests:
		print(f"  - {req.kernel_type:8s} {req.size_string:14s} {req.total_configs:7d}")
	print("=" * 70)

	manifest = {
		"created_at": datetime.now().isoformat(),
		"device_type": device_type,
		"seed": seed,
		"build_dir": build_dir,
		"runs_per_config": runs_per_config,
		"persist_artifacts": persist_artifacts,
		"requests": [
			{
				"kernel_type": req.kernel_type,
				"input_size": req.size_string,
				"total_configs": req.total_configs,
			}
			for req in requests
		],
	}
	with open(run_dir / "experiment_spec.json", "w") as f:
		json.dump(manifest, f, indent=2)

	try:
		resolved_configs = prepare_experiment_configs(
			requests=requests,
			configs_root=configs_root,
			device_type=device_type,
			seed=seed,
		)

		scheduling = schedule_experiment_work(
			resolved_configs=resolved_configs,
			db_path=db_path,
			llvm_root=llvm_root,
			build_dir=build_dir,
			runs_per_config=runs_per_config,
			resume=resume,
			persist_artifacts=persist_artifacts,
		)
		experiments = scheduling.experiments

		if not experiments:
			print("No new experiments to process")
			return 1 if scheduling.error_count or not resolved_configs else 0

		print(f"\nProcessing {len(experiments)} experiments with {num_workers} workers...")
		print("=" * 70)

		completed = 0
		failed = 0
		try:
			with Pool(num_workers) as pool:
				for i, (_, success, _) in enumerate(pool.imap_unordered(process_single_experiment, experiments)):
					if success:
						completed += 1
					else:
						failed += 1
					if (i + 1) % 10 == 0 or (i + 1) == len(experiments):
						print(f"  [{i+1}/{len(experiments)}] {completed} completed, {failed} failed")
		except Exception as exc:
			print(f"Worker pool error: {_format_exception(exc)}", file=sys.stderr)
			return 1

		db = ExperimentDB(str(db_path))
		stats = db.get_statistics()
		db.close()

		print("=" * 70)
		print("Final Statistics:")
		print(f"  Total:       {stats['total']:,}")
		print(f"  Completed:   {stats['completed']:,}")
		print(f"  Failed:      {stats['failed']:,}")
		print(f"  Pending:     {stats['pending']:,}")
		if stats["avg_runtime_ms"]:
			print(f"  Avg runtime: {stats['avg_runtime_ms']:.3f} ms")
		print("=" * 70)

		return 0 if failed == 0 and scheduling.error_count == 0 else 1
	except Exception as e:
		print(f"Collection error: {e}", file=sys.stderr)
		return 1


def parse_requests(args: argparse.Namespace) -> List[ExperimentRequest]:
	requests: List[ExperimentRequest] = []

	if args.request:
		for kernel_type, input_size, total_configs in args.request:
			requests.append(_parse_request(kernel_type, input_size, total_configs))

	if args.requests_file:
		with open(args.requests_file) as f:
			items = json.load(f)
		for item in items:
			requests.append(
				_parse_request(
					str(item["kernel_type"]),
					str(item["input_size"]),
					str(item["total_configs"]),
				)
			)

	if args.test_mode and not requests:
		requests.extend(
			[
				_parse_request("gemm", args.test_gemm_size, str(args.test_configs)),
				_parse_request("gaussian", args.test_gaussian_size, str(args.test_configs)),
			]
		)

	if not requests:
		raise ValueError("No requests provided. Use --request/--requests-file or enable --test-mode.")

	return requests


def main() -> int:
	"""Parse arguments and run collection."""
	parser = argparse.ArgumentParser(
		description="Collect tuning parameter performance and LLVM IR using request-driven config generation"
	)
	parser.add_argument(
		"--test-mode",
		action="store_true",
		help="Quick smoke-test mode: runs small GEMM + Gaussian requests (CPU default unless --device-type is explicitly set)",
	)
	parser.add_argument(
		"--test-configs",
		type=int,
		default=5,
		help="Configs per kernel in test mode when requests are not explicitly provided (default: 5)",
	)
	parser.add_argument(
		"--test-gemm-size",
		default="128x128x128",
		help="GEMM input size for test mode (default: 128x128x128)",
	)
	parser.add_argument(
		"--test-gaussian-size",
		default="256x256",
		help="Gaussian input size for test mode (default: 256x256)",
	)
	parser.add_argument(
		"--request",
		nargs=3,
		action="append",
		metavar=("KERNEL", "INPUT_SIZE", "TOTAL_CONFIGS"),
		help="Experiment request tuple (e.g. --request gemm 256x256x256 200). Repeatable.",
	)
	parser.add_argument(
		"--requests-file",
		help="Path to JSON file containing a list of request objects: kernel_type, input_size, total_configs",
	)
	parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
	parser.add_argument("--resume", action="store_true", help="Resume existing experiment directory")
	parser.add_argument(
		"--experiment-root",
		default=DEFAULT_EXPERIMENT_ROOT,
		help=f"Root directory for experiment runs (default: {DEFAULT_EXPERIMENT_ROOT})",
	)
	parser.add_argument(
		"--experiment-name",
		default=None,
		help="Experiment directory name under experiment-root (default: timestamp)",
	)
	parser.add_argument(
		"--device-type",
		choices=["cpu", "gpu"],
		default="cpu",
		help="Device type used for config generation constraints",
	)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed")
	parser.add_argument(
		"--runs-per-config",
		type=int,
		default=DEFAULT_RUNS_PER_CONFIG,
		help=f"Number of repeated kernel runs per config; stored runtime is the average (default: {DEFAULT_RUNS_PER_CONFIG})",
	)
	parser.add_argument(
		"--build-dir",
		default=DEFAULT_BUILD_DIR,
		help=f"Directory containing kernel executables (default: {DEFAULT_BUILD_DIR})",
	)
	parser.add_argument(
		"--persist-artifacts",
		action="store_true",
		help="Keep dumped OpenCL binaries and runtime LLVM IR files under the experiment directory.",
	)

	args = parser.parse_args()

	if args.test_mode:
		if "--device-type" not in sys.argv:
			args.device_type = "cpu"
		if args.test_configs <= 0:
			print("Argument error: --test-configs must be > 0", file=sys.stderr)
			return 2

	if args.runs_per_config <= 0:
		print("Argument error: --runs-per-config must be > 0", file=sys.stderr)
		return 3
	if args.workers is not None and args.workers <= 0:
		print("Argument error: --workers must be > 0", file=sys.stderr)
		return 3

	try:
		requests = parse_requests(args)
	except Exception as exc:
		print(f"Argument error: {exc}", file=sys.stderr)
		return 2

	return collect_experiments(
		requests=requests,
		num_workers=args.workers,
		resume=args.resume,
		experiment_root=args.experiment_root,
		experiment_name=args.experiment_name,
		device_type=args.device_type,
		seed=args.seed,
		build_dir=args.build_dir,
		runs_per_config=args.runs_per_config,
		persist_artifacts=args.persist_artifacts,
	)


if __name__ == "__main__":
	sys.exit(main())
