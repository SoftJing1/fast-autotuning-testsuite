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

DEFAULT_SEED = 42
DEFAULT_EXPERIMENT_ROOT = "experiments"
DEFAULT_DB_FILENAME = "experiments.db"
DEFAULT_BUILD_DIR = "build"
DEFAULT_RUNS_PER_CONFIG = 3

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


def hash_config(config_dict: Dict) -> str:
	"""Generate deterministic hash of configuration."""
	config_str = json.dumps(config_dict, sort_keys=True)
	return hashlib.md5(config_str.encode()).hexdigest()[:16]


def run_kernel(kernel_type: str, config_file: str, build_dir: str, timeout_sec: int = 60) -> Optional[float]:
	"""Run kernel with configuration and return runtime in milliseconds."""
	try:
		exe_path = os.path.join(build_dir, kernel_type)
		if not os.path.exists(exe_path):
			return None

		result = subprocess.run(
			[exe_path, config_file],
			capture_output=True,
			text=True,
			timeout=timeout_sec,
		)

		if result.returncode != 0:
			return None

		match = re.search(r"Device kernel execution time:\s*(\d+\.?\d*)\s*ms", result.stdout, re.IGNORECASE)
		if match:
			return float(match.group(1))

		raise ValueError(f"Runtime not found in output: {result.stdout}")

	except subprocess.TimeoutExpired:
		return None
	except Exception:
		return None


def run_kernel_average(
	kernel_type: str,
	config_file: str,
	build_dir: str,
	runs_per_config: int,
	timeout_sec: int = 60,
) -> Optional[float]:
	"""Run kernel multiple times and return average runtime in milliseconds."""
	if runs_per_config <= 0:
		raise ValueError("runs_per_config must be > 0")

	runtimes: List[float] = []
	for _ in range(runs_per_config):
		runtime_ms = run_kernel(kernel_type, config_file, build_dir=build_dir, timeout_sec=timeout_sec)
		if runtime_ms is not None:
			runtimes.append(runtime_ms)

	if not runtimes:
		return None

	return sum(runtimes) / len(runtimes)


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
	if not llvm_paths:
		return None
	if len(llvm_paths) == 1:
		return llvm_paths[0]
	return json.dumps(list(llvm_paths))


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
		request_files = _generate_request_configs(
			request=request,
			config_dir=config_dir,
			generator_module=modules[request.kernel_type],
			device_type=device_type,
			seed=seed,
		)
		resolved.extend((request, path) for path in request_files)

	return resolved


def process_single_experiment(args: Tuple[int, str, str, str, str, str, int]) -> Tuple[int, bool, str]:
	"""Process single experiment in worker process."""
	exp_id, kernel_type, config_file, db_path, llvm_output_dir, build_dir, runs_per_config = args

	try:
		db = ExperimentDB(db_path)
		with open(config_file) as f:
			config = json.load(f)

		runtime_ms = run_kernel_average(
			kernel_type,
			config_file,
			build_dir=build_dir,
			runs_per_config=runs_per_config,
		)
		if runtime_ms is None:
			db.mark_failed(exp_id, f"Execution failed across {runs_per_config} run(s)")
			db.close()
			return (exp_id, False, "Execution failed")

		param_hash = hash_config(config)
		llvm_artifacts = compile_to_llvm_ir(kernel_type, config, param_hash, Path(llvm_output_dir))
		db.mark_completed(
			exp_id,
			runtime_ms,
			_serialize_llvm_paths([artifact["llvm_ir_path"] for artifact in llvm_artifacts]),
		)

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
					llvm_ir_path=artifact["llvm_ir_path"],
					bb_counts=inst_result.bb_counts,
					bb_instruction_counts=inst_result.bb_instruction_counts,
					total_instruction_counts=inst_result.total_instruction_counts,
				)
			except Exception as e:
				print(
					f"Warning: failed to extract/store LLVM instruction counts for exp_id={exp_id}, "
					f"template={artifact['template_name']}: {e}",
					file=sys.stderr,
				)
		db.close()
		return (exp_id, True, "")
	except Exception as e:
		try:
			db = ExperimentDB(db_path)
			db.mark_failed(exp_id, str(e))
			db.close()
		except Exception:
			pass
		return (exp_id, False, str(e))


def _build_experiment_directory(root_dir: str, experiment_name: Optional[str], resume: bool) -> Path:
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
) -> int:
	"""Main collection pipeline from request tuples."""
	if num_workers is None:
		num_workers = max(1, cpu_count() - 1)
	if runs_per_config <= 0:
		raise ValueError("runs_per_config must be > 0")

	run_dir = _build_experiment_directory(experiment_root, experiment_name, resume=resume)
	configs_root = run_dir / "configs"
	llvm_root = run_dir / "llvm_ir"
	db_path = run_dir / DEFAULT_DB_FILENAME

	print("\nTuning Collection System")
	print("=" * 70)
	print(f"Experiment dir: {run_dir}")
	print(f"Database:       {db_path}")
	print(f"Configs root:   {configs_root}")
	print(f"LLVM IR root:   {llvm_root}")
	print(f"Build dir:      {build_dir}")
	print(f"Runs/config:    {runs_per_config}")
	print(f"Workers:        {num_workers}")
	print(f"Resume:         {resume}")
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

		db = ExperimentDB(str(db_path))
		experiments: List[Tuple[int, str, str, str, str, str, int]] = []

		for request, config_file in resolved_configs:
			with open(config_file) as f:
				config = json.load(f)

			param_hash = hash_config(config)
			exp_id = db.add_experiment(
				kernel_type=request.kernel_type,
				input_size=request.size_string,
				param_hash=param_hash,
				config_dict=config,
				skip_if_exists=resume,
			)

			if exp_id is not None:
				experiments.append(
					(
						exp_id,
						request.kernel_type,
						str(config_file),
						str(db_path),
						str(llvm_root),
						build_dir,
						runs_per_config,
					)
				)

		db.close()

		if not experiments:
			print("No new experiments to process")
			return 0

		print(f"\nProcessing {len(experiments)} experiments with {num_workers} workers...")
		print("=" * 70)

		completed = 0
		failed = 0
		with Pool(num_workers) as pool:
			for i, (_, success, _) in enumerate(pool.imap_unordered(process_single_experiment, experiments)):
				if success:
					completed += 1
				else:
					failed += 1
				if (i + 1) % 10 == 0 or (i + 1) == len(experiments):
					print(f"  [{i+1}/{len(experiments)}] {completed} completed, {failed} failed")

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

		return 0 if failed == 0 else 1
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
	)


if __name__ == "__main__":
	sys.exit(main())
