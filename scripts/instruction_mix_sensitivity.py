#!/usr/bin/env python3
"""Run single-parameter instruction-mix sensitivity experiments.

The study fixes GEMM at 16x16x16 and Gaussian/conv at 16x16, chooses the
lexicographically smallest valid configuration as the base, then varies one
tuning parameter at a time while reusing the existing config generators, LLVM
compiler helper, and symbolic instruction-count extractor.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from scripts.collect_tuning_performance_simple import hash_config  # noqa: E402
from scripts.internal.compile_to_llvm import compile_opencl_to_llvm  # noqa: E402
from scripts.internal.llvm_to_inst_count import extract_instruction_counts  # noqa: E402


DEFAULT_KERNEL_INPUTS = {
	"gemm": (16, 16, 16),
	"gaussian": (16, 16),
}

TEMPLATE_SPECS = {
	"gemm": [
		("gemm_1", "gemm_1", ROOT / "kernels/kernel-template/gemm_1.cl"),
		("gemm_2", "gemm_2", ROOT / "kernels/kernel-template/gemm_2.cl"),
	],
	"gaussian": [
		("gaussian_static_1", "gaussian_1", ROOT / "kernels/kernel-template/gaussian_static_1.cl"),
	],
}

INPUT_KEYS = {
	"gemm": {"INPUT_SIZE_L_1", "INPUT_SIZE_L_2", "INPUT_SIZE_R_1", "M", "N", "K"},
	"gaussian": {"INPUT_SIZE_1", "INPUT_SIZE_2", "input_size_h", "input_size_w"},
}


@dataclass(frozen=True)
class MixArtifact:
	kernel_type: str
	case_id: str
	parameter: str
	value: Any
	config_path: Path
	template_name: str
	kernel_function: str
	llvm_ir_path: Path
	mix_path: Path
	total_instruction_counts: Dict[str, int]


class MarkdownLog:
	def __init__(self, path: Path):
		self.path = path
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self.path.write_text("# Instruction-Mix Sensitivity Log\n\n")

	def add(self, title: str, body: str = "") -> None:
		with self.path.open("a") as f:
			f.write(f"## {title}\n\n")
			if body:
				f.write(body.rstrip() + "\n\n")


def _load_module_from_path(module_name: str, module_path: Path):
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Failed to load module spec from {module_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _config_generator(kernel_type: str):
	base = ROOT / "scripts/internal"
	if kernel_type == "gemm":
		return _load_module_from_path("config_gen_gemm_sensitivity", base / "config-gen-gemm.py")
	if kernel_type == "gaussian":
		return _load_module_from_path("config_gen_gaussian_sensitivity", base / "config-gen-gaussian.py")
	raise ValueError(f"Unsupported kernel type: {kernel_type}")


def _require_tools() -> None:
	missing = [tool for tool in ("clang", "symb-viewer") if shutil.which(tool) is None]
	if missing:
		raise RuntimeError(f"Required tool(s) not found in PATH: {', '.join(missing)}")


def _device_limits(device_type: str) -> Tuple[Tuple[int, int, int], int]:
	if device_type == "cpu":
		return (8192, 8192, 8192), 8192
	return (1024, 1024, 64), 1024


def _config_key(config: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
	return tuple(sorted(config.items()))


def _iter_valid_configs(
	kernel_type: str,
	module: Any,
	device_type: str,
	kernel_inputs: Dict[str, Tuple[int, ...]],
) -> Iterator[Dict[str, Any]]:
	max_wi_size, max_wg_size = _device_limits(device_type)
	if kernel_type == "gemm":
		m, n, k = kernel_inputs["gemm"]
		yield from module.exhaustive_iterate_configurations(
			m,
			n,
			k,
			max_wi_size=max_wi_size,
			max_wg_size=max_wg_size,
		)
		return

	h, w = kernel_inputs["gaussian"]
	yield from module.exhaustive_iterate_configurations(
		h,
		w,
		max_wi_size=max_wi_size,
		max_wg_size=max_wg_size,
	)


def _collect_valid_configs(
	kernel_type: str,
	module: Any,
	device_type: str,
	kernel_inputs: Dict[str, Tuple[int, ...]],
) -> List[Dict[str, Any]]:
	configs = list(_iter_valid_configs(kernel_type, module, device_type, kernel_inputs))
	if not configs:
		raise RuntimeError(f"No valid configs found for {kernel_type}")
	return sorted(configs, key=_config_key)


def _is_valid_config(kernel_type: str, module: Any, config: Dict[str, Any], device_type: str) -> bool:
	max_wi_size, max_wg_size = _device_limits(device_type)
	if kernel_type == "gemm" and config["P_CB_RES_DEST_LEVEL"] > config["L_CB_RES_DEST_LEVEL"]:
		return False
	return bool(module.is_configuration_valid(config, max_wi_size=max_wi_size, max_wg_size=max_wg_size))


def _tuning_parameters(kernel_type: str, base_config: Dict[str, Any]) -> List[str]:
	excluded = INPUT_KEYS[kernel_type]
	return sorted(key for key in base_config if key not in excluded)


def _select_values(values: Sequence[Any], max_values: int) -> List[Any]:
	unique = sorted(set(values))
	if len(unique) <= max_values:
		return unique
	if max_values < 3:
		return [unique[0], unique[-1]][:max_values]
	indexes = {0, len(unique) - 1}
	for slot in range(1, max_values - 1):
		indexes.add(round(slot * (len(unique) - 1) / (max_values - 1)))
	return [unique[i] for i in sorted(indexes)]


def _parameter_ranges(kernel_type: str, module: Any, kernel_inputs: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[int, int]]:
	if kernel_type == "gemm":
		manipulator = module.get_parameters_definition(*kernel_inputs["gemm"])
	else:
		manipulator = module.get_parameters_definition(*kernel_inputs["gaussian"])
	ranges = {}
	for parameter in getattr(manipulator, "params", []):
		if hasattr(parameter, "min_value") and hasattr(parameter, "max_value"):
			ranges[parameter.name] = (int(parameter.min_value), int(parameter.max_value))
	return ranges


def _constraint_note(kernel_type: str, parameter: str, base_config: Dict[str, Any]) -> str:
	if kernel_type == "gemm":
		for group in (
			{"L_CB_SIZE_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "P_CB_SIZE_L_1"},
			{"L_CB_SIZE_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "P_CB_SIZE_L_2"},
			{"L_CB_SIZE_R_1", "NUM_WG_R_1", "NUM_WI_R_1", "P_CB_SIZE_R_1"},
		):
			if parameter in group:
				return (
					"GEMM dimension decomposition is exact: INPUT_SIZE = L_CB_SIZE * NUM_WG * "
					"NUM_WI * P_CB_SIZE. Changing only one factor changes the product, so no "
					"single-parameter variant can stay valid for this base."
				)
		if parameter in {"OCL_DIM_L_1", "OCL_DIM_L_2", "OCL_DIM_R_1"}:
			return (
				"GEMM requires all three OpenCL dimension assignments to be different. With the "
				"other two assignments fixed, changing one assignment collides with an existing one."
			)
		if parameter == "P_CB_RES_DEST_LEVEL":
			return (
				"GEMM requires P_CB_RES_DEST_LEVEL <= L_CB_RES_DEST_LEVEL in the exhaustive generator. "
				f"The base has L_CB_RES_DEST_LEVEL={base_config['L_CB_RES_DEST_LEVEL']}."
			)
		if parameter in {"G_CB_RES_DEST_LEVEL", "L_REDUCTION", "P_WRITE_BACK", "L_WRITE_BACK"}:
			return "This parameter is fixed by the generator for the selected input/template family."
	else:
		for group in (
			{"GLB_1", "WG_1", "LCL_1", "WI_1", "PRV_1"},
			{"GLB_2", "WG_2", "LCL_2", "WI_2", "PRV_2"},
		):
			if parameter in group:
				return (
					"Gaussian dimension decomposition is exact: INPUT_SIZE = GLB * WG * LCL * "
					"WI * PRV. Changing only one factor changes the product, so no single-parameter "
					"variant can stay valid for this base."
				)
		if parameter in {
			"G_CB_RES_DEST_LEVEL",
			"L_CB_RES_DEST_LEVEL",
			"P_CB_RES_DEST_LEVEL",
			"WG_1_OCL_DIM",
			"WG_2_OCL_DIM",
			"WI_1_OCL_DIM",
			"WI_2_OCL_DIM",
		}:
			return "This parameter is fixed by the Gaussian generator for the selected template family."
	return "Candidate values were generated from the declared parameter range and rejected by the validator."


def _single_parameter_variants(
	kernel_type: str,
	module: Any,
	base_config: Dict[str, Any],
	parameter_ranges: Dict[str, Tuple[int, int]],
	parameter: str,
	device_type: str,
	max_values_per_parameter: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
	base_value = base_config[parameter]
	min_value, max_value = parameter_ranges[parameter]
	candidate_values = [value for value in range(min_value, max_value + 1) if value != base_value]
	valid_candidates = []
	rejected_values = []
	for value in candidate_values:
		candidate = dict(base_config)
		candidate[parameter] = value
		if _is_valid_config(kernel_type, module, candidate, device_type):
			valid_candidates.append(candidate)
		else:
			rejected_values.append(value)

	selected_values = _select_values([config[parameter] for config in valid_candidates], max_values_per_parameter)
	variants = []
	seen = set()
	for value in selected_values:
		candidate = dict(base_config)
		candidate[parameter] = value
		key = _config_key(candidate)
		if key in seen:
			continue
		seen.add(key)
		variants.append(candidate)
	audit = {
		"kernel_type": kernel_type,
		"parameter": parameter,
		"base_value": base_value,
		"declared_min": min_value,
		"declared_max": max_value,
		"candidate_value_count": len(candidate_values),
		"valid_value_count": len(valid_candidates),
		"selected_value_count": len(variants),
		"selected_values": [config[parameter] for config in variants],
		"rejected_values_sample": rejected_values[:20],
		"constraint_note": _constraint_note(kernel_type, parameter, base_config) if not variants else "",
	}
	return variants, audit


def _write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w") as f:
		json.dump(payload, f, indent=2, sort_keys=True)


def _defines_for_config(kernel_type: str, config: Dict[str, Any]) -> str:
	lines = [f"#define {key.upper()} {value}" for key, value in sorted(config.items())]
	if kernel_type == "gaussian":
		lines.extend(
			[
				"#ifndef IN_CACHE_LCL",
				"#define IN_CACHE_LCL IMAGES_CACHE_LCL",
				"#endif",
				"#ifndef IN_CACHE_PRV",
				"#define IN_CACHE_PRV IMAGES_CACHE_PRV",
				"#endif",
			]
		)
	return "\n".join(lines) + "\n"


def _compile_ir(
	kernel_type: str,
	template_path: Path,
	config: Dict[str, Any],
	llvm_path: Path,
	params_dir: Path,
) -> None:
	params_dir.mkdir(parents=True, exist_ok=True)
	params_path = params_dir / f"{llvm_path.stem}_params.cl"
	params_path.write_text(_defines_for_config(kernel_type, config))
	success = compile_opencl_to_llvm(
		str(template_path),
		str(params_path),
		str(llvm_path),
	)
	if not success or not llvm_path.exists():
		raise RuntimeError(f"Failed to compile {template_path} to {llvm_path}")


def _case_id(kernel_type: str, parameter: str, value: Any, config: Dict[str, Any]) -> str:
	config_hash = hash_config(config)
	safe_parameter = parameter.lower()
	return f"{kernel_type}_{safe_parameter}_{value}_{config_hash}"


def _run_case(
	kernel_type: str,
	case_id: str,
	parameter: str,
	value: Any,
	config: Dict[str, Any],
	run_dir: Path,
	log: MarkdownLog,
) -> List[MixArtifact]:
	config_path = run_dir / "configs" / kernel_type / f"{case_id}.json"
	_write_json(config_path, config)

	artifacts = []
	for template_name, kernel_function, template_path in TEMPLATE_SPECS[kernel_type]:
		llvm_path = run_dir / "llvm_ir" / kernel_type / f"{case_id}_{template_name}.ll"
		_compile_ir(
			kernel_type=kernel_type,
			template_path=template_path,
			config=config,
			llvm_path=llvm_path,
			params_dir=run_dir / "params_cl" / kernel_type,
		)
		inst_result = extract_instruction_counts(str(llvm_path), kernel_function=kernel_function)
		mix_path = run_dir / "mixes" / kernel_type / f"{case_id}_{template_name}.json"
		_write_json(mix_path, inst_result.to_dict())
		artifacts.append(
			MixArtifact(
				kernel_type=kernel_type,
				case_id=case_id,
				parameter=parameter,
				value=value,
				config_path=config_path,
				template_name=template_name,
				kernel_function=kernel_function,
				llvm_ir_path=llvm_path,
				mix_path=mix_path,
				total_instruction_counts=dict(inst_result.total_instruction_counts),
			)
		)

	log.add(
		f"Collected {case_id}",
		"Computed LLVM IR and symbolic instruction mix for "
		f"`{kernel_type}` parameter `{parameter}` value `{value}`. "
		"The mix is computed by `extract_instruction_counts`, which multiplies "
		"per-basic-block instruction counts by symbolic basic-block execution counts.",
	)
	return artifacts


def _sum_counts(counts: Iterable[Dict[str, int]]) -> Dict[str, int]:
	total: Dict[str, int] = {}
	for mapping in counts:
		for inst, count in mapping.items():
			total[inst] = total.get(inst, 0) + int(count)
	return total


def _total_count(counts: Dict[str, int]) -> int:
	return sum(int(value) for value in counts.values())


def _delta_rows_for_case(
	artifact: MixArtifact,
	base_counts: Dict[str, int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
	new_counts = artifact.total_instruction_counts
	base_total = _total_count(base_counts)
	new_total = _total_count(new_counts)
	summary = {
		"kernel_type": artifact.kernel_type,
		"case_id": artifact.case_id,
		"parameter": artifact.parameter,
		"value": artifact.value,
		"template_name": artifact.template_name,
		"base_total": base_total,
		"new_total": new_total,
		"delta_total": new_total - base_total,
		"delta_percent": ((new_total - base_total) / base_total * 100.0) if base_total else None,
		"config_path": str(artifact.config_path.relative_to(ROOT)),
		"llvm_ir_path": str(artifact.llvm_ir_path.relative_to(ROOT)),
		"mix_path": str(artifact.mix_path.relative_to(ROOT)),
	}
	per_op = []
	for inst in sorted(set(base_counts) | set(new_counts)):
		base_value = int(base_counts.get(inst, 0))
		new_value = int(new_counts.get(inst, 0))
		if base_value == new_value:
			continue
		per_op.append(
			{
				**summary,
				"instruction": inst,
				"base_count": base_value,
				"new_count": new_value,
				"delta_count": new_value - base_value,
			}
		)
	return summary, per_op


def _write_delta_outputs(run_dir: Path, summaries: List[Dict[str, Any]], per_op_rows: List[Dict[str, Any]]) -> None:
	payload = {"summaries": summaries, "per_instruction_deltas": per_op_rows}
	_write_json(run_dir / "deltas.json", payload)

	csv_path = run_dir / "deltas.csv"
	fields = [
		"kernel_type",
		"case_id",
		"parameter",
		"value",
		"template_name",
		"base_total",
		"new_total",
		"delta_total",
		"delta_percent",
		"instruction",
		"base_count",
		"new_count",
		"delta_count",
		"config_path",
		"llvm_ir_path",
		"mix_path",
	]
	with csv_path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields)
		writer.writeheader()
		for row in per_op_rows:
			writer.writerow({field: row.get(field) for field in fields})


def _write_preliminary_results(run_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
	ranked = sorted(summaries, key=lambda row: abs(int(row["delta_total"])), reverse=True)
	lines = [
		"# Preliminary Instruction-Mix Sensitivity Results",
		"",
		"Rows are ranked by absolute change in total dynamic instruction count versus the base mix.",
		"",
		"| Rank | Kernel | Template | Parameter | Value | Base Total | New Total | Delta | Delta % |",
		"| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
	]
	for idx, row in enumerate(ranked[:100], start=1):
		delta_percent = row["delta_percent"]
		percent_text = "" if delta_percent is None else f"{delta_percent:.2f}"
		lines.append(
			f"| {idx} | {row['kernel_type']} | {row['template_name']} | "
			f"`{row['parameter']}` | {row['value']} | {row['base_total']} | "
			f"{row['new_total']} | {row['delta_total']} | {percent_text} |"
		)
	(run_dir / "preliminary_results.md").write_text("\n".join(lines) + "\n")


def _write_candidate_audit_report(run_dir: Path, candidate_audits: Sequence[Dict[str, Any]]) -> None:
	lines = [
		"# Candidate Generation Audit",
		"",
		"Candidate values are generated directly from the declared parameter ranges, applied to the base config one parameter at a time, and then checked with the generator validator plus the GEMM hierarchy constraint used by exhaustive generation.",
		"",
		"| Kernel | Parameter | Base | Range | Candidates | Valid | Selected | Note |",
		"| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
	]
	for audit in candidate_audits:
		selected = ", ".join(str(value) for value in audit["selected_values"]) or "-"
		note = audit["constraint_note"].replace("|", "\\|") if audit["constraint_note"] else "-"
		lines.append(
			f"| {audit['kernel_type']} | `{audit['parameter']}` | {audit['base_value']} | "
			f"{audit['declared_min']}..{audit['declared_max']} | {audit['candidate_value_count']} | "
			f"{audit['valid_value_count']} | {selected} | {note} |"
		)
	(run_dir / "candidate_generation_audit.md").write_text("\n".join(lines) + "\n")


def _write_analysis_prompt(run_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
	top_rows = sorted(summaries, key=lambda row: abs(int(row["delta_total"])), reverse=True)[:30]
	lines = [
		"# Subagent Analysis Prompt",
		"",
		"Explain the instruction-mix deltas using the OpenCL templates themselves.",
		"",
		"Templates to inspect:",
		"- `kernels/kernel-template/gemm_1.cl`",
		"- `kernels/kernel-template/gemm_2.cl`",
		"- `kernels/kernel-template/gaussian_static_1.cl`",
		"",
		"Focus on macro-controlled loop counts, cache branches, reduction/writeback paths, and OpenCL work item/group mapping.",
		"",
		"Important Gaussian/conv setup note: this controlled analysis defines `IN_CACHE_LCL` from `IMAGES_CACHE_LCL` and `IN_CACHE_PRV` from `IMAGES_CACHE_PRV`, because the host/config names and template branch names differ.",
		"",
		"Top observed total-instruction deltas:",
		"",
		"| Kernel | Template | Parameter | Value | Base Total | New Total | Delta |",
		"| --- | --- | --- | ---: | ---: | ---: | ---: |",
	]
	for row in top_rows:
		lines.append(
			f"| {row['kernel_type']} | {row['template_name']} | `{row['parameter']}` | "
			f"{row['value']} | {row['base_total']} | {row['new_total']} | {row['delta_total']} |"
		)
	lines.extend(
		[
			"",
			"Required output:",
			"- Hypotheses tied to concrete template code locations.",
			"- For each hypothesis, a proposed verification config pattern.",
			"- Mark any hypothesis that cannot be verified by single-parameter sweeps alone.",
		]
	)
	(run_dir / "analysis_prompt.md").write_text("\n".join(lines) + "\n")


def _write_placeholder_analysis(run_dir: Path) -> None:
	(run_dir / "subagent_analysis.md").write_text(
		"# Subagent Analysis\n\n"
		"This file is reserved for the post-run subagent explanation. "
		"Use `analysis_prompt.md`, `preliminary_results.md`, `deltas.json`, "
		"and the generated configs/mixes as the analysis packet.\n"
	)


def _write_verification_seed(run_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
	verification_dir = run_dir / "verification"
	verification_dir.mkdir(parents=True, exist_ok=True)
	top_rows = sorted(summaries, key=lambda row: abs(int(row["delta_total"])), reverse=True)[:10]
	_write_json(
		verification_dir / "verification_candidates.json",
		{
			"purpose": (
				"Initial candidates for hypothesis-instantiating experiments. "
				"After subagent analysis, create paired configs from these high-impact parameters."
			),
			"top_delta_rows": top_rows,
		},
	)
	(verification_dir / "verification_results.md").write_text(
		"# Verification Results\n\n"
		"Populate this after comparing subagent hypotheses against targeted verification experiments.\n"
	)


def _make_run_dir(args: argparse.Namespace) -> Path:
	root = Path(args.experiment_root)
	if not root.is_absolute():
		root = ROOT / root
	root.mkdir(parents=True, exist_ok=True)
	name = args.experiment_name or f"instruction_mix_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	run_dir = root / name
	if run_dir.exists() and not args.resume:
		raise RuntimeError(f"Experiment directory already exists: {run_dir}. Use --resume or choose a new name.")
	run_dir.mkdir(parents=True, exist_ok=True)
	for child in ("configs", "llvm_ir", "mixes", "params_cl", "verification"):
		(run_dir / child).mkdir(parents=True, exist_ok=True)
	return run_dir


def _filter_parameters(parameters: Sequence[str], args: argparse.Namespace) -> List[str]:
	filtered = list(parameters)
	if args.only_parameter:
		wanted = {value.upper() for value in args.only_parameter}
		filtered = [param for param in filtered if param.upper() in wanted]
	if args.max_parameters_per_kernel is not None:
		filtered = filtered[: args.max_parameters_per_kernel]
	return filtered


def _parse_size(value: str, expected_parts: int, label: str) -> Tuple[int, ...]:
	parts = value.lower().split("x")
	if len(parts) != expected_parts:
		raise ValueError(f"{label} must have {expected_parts} dimensions, got: {value}")
	try:
		dims = tuple(int(part) for part in parts)
	except ValueError as exc:
		raise ValueError(f"{label} dimensions must be integers, got: {value}") from exc
	if any(dim <= 0 for dim in dims):
		raise ValueError(f"{label} dimensions must be > 0, got: {value}")
	return dims


def _kernel_inputs_from_args(args: argparse.Namespace) -> Dict[str, Tuple[int, ...]]:
	return {
		"gemm": _parse_size(args.gemm_size, 3, "--gemm-size"),
		"gaussian": _parse_size(args.gaussian_size, 2, "--gaussian-size"),
	}


def run(args: argparse.Namespace) -> Path:
	_require_tools()
	kernel_inputs = _kernel_inputs_from_args(args)
	run_dir = _make_run_dir(args)
	log = MarkdownLog(run_dir / "experiment_log.md")
	log.add(
		"Run Setup",
		"Created the experiment directory and will reuse the existing exhaustive config generators, "
		"`compile_opencl_to_llvm`, and `extract_instruction_counts`. "
		f"GEMM input is fixed to {'x'.join(str(v) for v in kernel_inputs['gemm'])}; "
		f"Gaussian/conv input is fixed to {'x'.join(str(v) for v in kernel_inputs['gaussian'])}.",
	)
	log.add(
		"Gaussian Cache Alias",
		"The Gaussian/conv template branches on `IN_CACHE_LCL` and `IN_CACHE_PRV`, while the "
		"host/config path exposes `IMAGES_CACHE_LCL` and `IMAGES_CACHE_PRV`. This analysis compile "
		"adds aliases from `IN_CACHE_*` to `IMAGES_CACHE_*` without changing the host or template files.",
	)

	selected_kernels = ["gemm", "gaussian"] if args.kernel == "all" else [args.kernel]
	all_artifacts: List[MixArtifact] = []
	base_artifacts: Dict[Tuple[str, str], MixArtifact] = {}
	failed_cases: List[Dict[str, Any]] = []
	candidate_audits: List[Dict[str, Any]] = []

	for kernel_type in selected_kernels:
		module = _config_generator(kernel_type)
		parameter_ranges = _parameter_ranges(kernel_type, module, kernel_inputs)
		all_configs = _collect_valid_configs(kernel_type, module, args.device_type, kernel_inputs)
		base_config = all_configs[0]
		base_case = f"{kernel_type}_base_{hash_config(base_config)}"
		log.add(
			f"{kernel_type} Base Config",
			f"Enumerated {len(all_configs)} valid configs and selected the lexicographically smallest one. "
			f"Base config hash: `{hash_config(base_config)}`.",
		)
		base_results = _run_case(
			kernel_type=kernel_type,
			case_id=base_case,
			parameter="BASE",
			value="BASE",
			config=base_config,
			run_dir=run_dir,
			log=log,
		)
		all_artifacts.extend(base_results)
		for artifact in base_results:
			base_artifacts[(kernel_type, artifact.template_name)] = artifact

		parameters = _filter_parameters(_tuning_parameters(kernel_type, base_config), args)
		log.add(
			f"{kernel_type} Parameter Sweep",
			f"Sweeping {len(parameters)} tuning parameters. For each parameter, the script now generates "
			"candidate values directly from the declared parameter range, applies each value to the base config, "
			"then keeps only variants that remain valid under the generator validation rules.",
		)
		for parameter in parameters:
			variants, audit = _single_parameter_variants(
				kernel_type=kernel_type,
				module=module,
				base_config=base_config,
				parameter_ranges=parameter_ranges,
				parameter=parameter,
				device_type=args.device_type,
				max_values_per_parameter=args.max_values_per_parameter,
			)
			candidate_audits.append(audit)
			log.add(
				f"{kernel_type} `{parameter}` Candidates",
				f"Generated {audit['candidate_value_count']} candidate value(s) from range "
				f"{audit['declared_min']}..{audit['declared_max']}; {audit['valid_value_count']} remained "
				f"valid as single-parameter changes; selected {len(variants)} value(s) for compilation. "
				+ (f"Reason when none: {audit['constraint_note']}" if not variants else ""),
			)
			for config in variants:
				case_id = _case_id(kernel_type, parameter, config[parameter], config)
				try:
					all_artifacts.extend(
						_run_case(
							kernel_type=kernel_type,
							case_id=case_id,
							parameter=parameter,
							value=config[parameter],
							config=config,
							run_dir=run_dir,
							log=log,
						)
					)
				except Exception as exc:
					error = {
						"kernel_type": kernel_type,
						"case_id": case_id,
						"parameter": parameter,
						"value": config[parameter],
						"config_hash": hash_config(config),
						"error": f"{type(exc).__name__}: {exc}",
					}
					failed_cases.append(error)
					log.add(
						f"Skipped {case_id}",
						"The generator accepted this single-parameter variant, but compilation or "
						f"instruction-count extraction failed. The study skips it and records the failure.\n\n"
						f"```text\n{error['error']}\n```",
					)

	summaries: List[Dict[str, Any]] = []
	per_op_rows: List[Dict[str, Any]] = []
	for artifact in all_artifacts:
		if artifact.parameter == "BASE":
			continue
		base = base_artifacts[(artifact.kernel_type, artifact.template_name)]
		summary, per_op = _delta_rows_for_case(artifact, base.total_instruction_counts)
		summaries.append(summary)
		per_op_rows.extend(per_op)

	if "gemm" in selected_kernels:
		grouped: Dict[str, List[MixArtifact]] = {}
		for artifact in all_artifacts:
			if artifact.kernel_type != "gemm":
				continue
			grouped.setdefault(artifact.case_id, []).append(artifact)
		base_combined = _sum_counts(a.total_instruction_counts for a in grouped[next(k for k in grouped if "_base_" in k)])
		for case_id, artifacts in grouped.items():
			if "_base_" in case_id:
				continue
			combined = _sum_counts(a.total_instruction_counts for a in artifacts)
			proxy = artifacts[0]
			combined_artifact = MixArtifact(
				kernel_type="gemm",
				case_id=case_id,
				parameter=proxy.parameter,
				value=proxy.value,
				config_path=proxy.config_path,
				template_name="gemm_combined",
				kernel_function="gemm_1+gemm_2",
				llvm_ir_path=proxy.llvm_ir_path,
				mix_path=proxy.mix_path,
				total_instruction_counts=combined,
			)
			summary, per_op = _delta_rows_for_case(combined_artifact, base_combined)
			summaries.append(summary)
			per_op_rows.extend(per_op)

	_write_delta_outputs(run_dir, summaries, per_op_rows)
	_write_json(run_dir / "failed_cases.json", failed_cases)
	_write_json(run_dir / "candidate_generation_audit.json", candidate_audits)
	_write_candidate_audit_report(run_dir, candidate_audits)
	_write_preliminary_results(run_dir, summaries)
	_write_analysis_prompt(run_dir, summaries)
	_write_placeholder_analysis(run_dir)
	_write_verification_seed(run_dir, summaries)
	_write_json(
		run_dir / "experiment_spec.json",
		{
			"created_at": datetime.now().isoformat(),
			"device_type": args.device_type,
			"kernel": args.kernel,
			"inputs": kernel_inputs,
			"max_values_per_parameter": args.max_values_per_parameter,
			"max_parameters_per_kernel": args.max_parameters_per_kernel,
			"only_parameter": args.only_parameter,
			"candidate_generation": "manual_range_values_then_validation",
			"note": "Gaussian/conv analysis IR aliases IN_CACHE_* to IMAGES_CACHE_*.",
		},
	)
	log.add(
		"Outputs Complete",
		"Generated configs, LLVM IR, raw instruction mixes, delta JSON/CSV, preliminary ranked results, "
		"candidate generation audit, subagent analysis prompt, verification seed files, and skipped-case diagnostics.",
	)
	return run_dir


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--kernel", choices=["all", "gemm", "gaussian"], default="all")
	parser.add_argument("--experiment-root", default="experiments")
	parser.add_argument("--experiment-name", default=None)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--device-type", choices=["cpu", "gpu"], default="cpu")
	parser.add_argument("--gemm-size", default="16x16x16", help="Verification override for GEMM MxNxK.")
	parser.add_argument("--gaussian-size", default="16x16", help="Verification override for Gaussian/conv HxW.")
	parser.add_argument("--max-values-per-parameter", type=int, default=5)
	parser.add_argument("--max-parameters-per-kernel", type=int, default=None)
	parser.add_argument(
		"--only-parameter",
		action="append",
		default=None,
		help="Restrict sweeps to this parameter name. Can be passed multiple times.",
	)
	return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.max_values_per_parameter <= 0:
		parser.error("--max-values-per-parameter must be > 0")
	try:
		run_dir = run(args)
	except Exception as exc:
		print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
		return 1
	print(f"Instruction-mix sensitivity experiment written to: {run_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
