#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VALIDATION_PAYLOAD = "payload.json"
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from scripts.collect_tuning_performance_simple import (  # noqa: E402
	ExperimentRequest,
	hash_config,
	prepare_experiment_configs,
	run_kernel,
	runtime_ir_artifacts_from_dump,
)

LABEL_RE = re.compile(r"^([A-Za-z$._][-A-Za-z$._0-9]*|\d+):")
RUNTIME_SPIR_TRIPLE_RE = re.compile(r'^target triple = "spir64-unknown-unknown"$', re.MULTILINE)
RUNTIME_SPIR_DATALAYOUT_RE = re.compile(r'^target datalayout = ".*"$', re.MULTILINE)
ADDRSPACE_PTR_RE = re.compile(r"ptr addrspace\(\d+\)")
STANDALONE_ADDRSPACE_RE = re.compile(r"\saddrspace\(\d+\)")


@dataclass(frozen=True)
class LaunchGeometry:
	local_sizes: Tuple[int, int, int]
	num_groups: Tuple[int, int, int]


@dataclass(frozen=True)
class ProxyBuildPlan:
	target_function: str
	block_names: List[str]
	arg_lengths: List[int]
	launch: LaunchGeometry


@dataclass(frozen=True)
class GeneratedArtifactPair:
	root_dir: Path
	llvm_ir_path: Path
	config_path: Path
	kernel_function: Optional[str]


class SymbolicCountParseError(RuntimeError):
	def __init__(self, raw_counts: Dict[str, str]):
		self.raw_counts = raw_counts
		preview = ", ".join(f"{name}={value}" for name, value in list(raw_counts.items())[:5])
		super().__init__(f"Non-integer symb-viewer basic block counts: {preview}")


@dataclass(frozen=True)
class RuntimeIdSelection:
	group_ids: Tuple[int, int, int]
	local_ids: Tuple[int, int, int]


def _require_tool(name: str) -> None:
	if shutil.which(name) is None:
		raise RuntimeError(f"Required tool not found in PATH: {name}")


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		list(cmd),
		cwd=str(cwd) if cwd else None,
		check=True,
		capture_output=True,
		text=True,
	)


def _load_json(path: Path) -> Dict:
	with open(path) as f:
		return json.load(f)


def _infer_kernel_type(config: Dict) -> str:
	if {"M", "N", "K"}.issubset(config):
		return "gemm"
	if {"input_size_h", "input_size_w"}.issubset(config):
		return "gaussian"
	raise ValueError("Could not infer kernel type from config JSON")


def _resolve_target_function(llvm_text: str, explicit: Optional[str] = None) -> str:
	if explicit:
		return explicit

	for match in re.finditer(r"^define\s+.*?@([^(]+)\(", llvm_text, flags=re.MULTILINE):
		name = match.group(1)
		if name.startswith("__clang_ocl_kern_imp_"):
			return name

	for match in re.finditer(r"^define\s+.*?\bspir_kernel\b.*?@([^(]+)\(", llvm_text, flags=re.MULTILINE):
		return match.group(1)

	raise ValueError("Could not find a callable kernel function in LLVM IR")


def _find_function_region(lines: List[str], function_name: str) -> Tuple[int, int, str]:
	start = -1
	header = ""
	pattern = re.compile(rf"^define\s+.*@{re.escape(function_name)}\(")
	for i, line in enumerate(lines):
		if pattern.search(line):
			start = i
			header = line
			break
	if start < 0:
		raise ValueError(f"Function not found in LLVM IR: {function_name}")

	for i in range(start + 1, len(lines)):
		if lines[i].strip() == "}":
			return start, i, header

	raise ValueError(f"Function body for {function_name} is not closed")


def _entry_block_name(header: str) -> str:
	match = re.search(rf"\((.*)\)\s*(?:local_unnamed_addr|#\d+|{{)", header)
	params = match.group(1) if match else header
	numbered = [int(value) for value in re.findall(r"%(\d+)", params)]
	if numbered:
		return f"%{max(numbered) + 1}"
	return "%entry"


def _split_blocks(body_lines: List[str], entry_name: str) -> List[Tuple[str, Optional[str], List[str]]]:
	blocks: List[Tuple[str, Optional[str], List[str]]] = []
	current_name = entry_name
	current_label = None
	current_body: List[str] = []

	for line in body_lines:
		label_match = LABEL_RE.match(line)
		if label_match:
			if current_body:
				blocks.append((current_name, current_label, current_body))
			label = label_match.group(1)
			current_name = f"%{label}"
			current_label = line
			current_body = []
			continue
		current_body.append(line)

	if current_body:
		blocks.append((current_name, current_label, current_body))

	return blocks


def _is_phi_line(line: str) -> bool:
	stripped = line.lstrip()
	return bool(re.match(r"%[-A-Za-z$._0-9]+\s*=\s*phi\b", stripped))


def _instrument_function(llvm_text: str, function_name: str) -> Tuple[str, List[str]]:
	lines = llvm_text.splitlines(keepends=True)
	start, end, header = _find_function_region(lines, function_name)
	entry_name = _entry_block_name(header)
	body_lines = lines[start + 1:end]
	blocks = _split_blocks(body_lines, entry_name)
	if not blocks:
		raise ValueError(f"No basic blocks found in function {function_name}")

	instrumented_body: List[str] = []
	block_names: List[str] = []
	for idx, (block_name, label_line, block_body) in enumerate(blocks):
		block_names.append(block_name)
		if label_line is not None:
			instrumented_body.append(label_line)

		insert_at = 0
		while insert_at < len(block_body) and _is_phi_line(block_body[insert_at]):
			insert_at += 1
		instrumented_body.extend(block_body[:insert_at])
		instrumented_body.append(f"  call void @__bb_hit(i32 {idx})\n")
		instrumented_body.extend(block_body[insert_at:])

	new_lines = lines[: start + 1] + instrumented_body + lines[end:]
	new_text = "".join(new_lines)

	if "declare void @__bb_hit(i32)" not in new_text:
		insert_pos = new_text.find("\nattributes #")
		if insert_pos < 0:
			insert_pos = new_text.find("\n!llvm.module.flags")
		if insert_pos < 0:
			insert_pos = len(new_text)
		new_text = (
			new_text[:insert_pos]
			+ "\ndeclare void @__bb_hit(i32)\n"
			+ new_text[insert_pos:]
		)

	return new_text, block_names


def _parse_function_signature(llvm_text: str, function_name: str) -> List[str]:
	lines = llvm_text.splitlines()
	_, _, header = _find_function_region([line + "\n" for line in lines], function_name)
	match = re.search(rf"@{re.escape(function_name)}\((.*)\)", header)
	if not match:
		raise ValueError(f"Could not parse function signature for {function_name}")
	params = match.group(1).strip()
	if not params:
		return []
	return [part.strip() for part in params.split(",")]


def _build_launch_geometry(kernel_type: str, function_name: str, config: Dict) -> LaunchGeometry:
	if kernel_type == "gaussian":
		local_x = int((config["wi_1_ocl_dim"] == 0) * config["wi_1"] + (config["wi_2_ocl_dim"] == 0) * config["wi_2"])
		local_y = int((config["wi_1_ocl_dim"] == 1) * config["wi_1"] + (config["wi_2_ocl_dim"] == 1) * config["wi_2"])
		global_x = int(((config["wg_1_ocl_dim"] == 0) * config["wg_1"] + (config["wg_2_ocl_dim"] == 0) * config["wg_2"]) * local_x)
		global_y = int(((config["wg_1_ocl_dim"] == 1) * config["wg_1"] + (config["wg_2_ocl_dim"] == 1) * config["wg_2"]) * local_y)
		local = (max(local_x, 1), max(local_y, 1), 1)
		return LaunchGeometry(
			local_sizes=local,
			num_groups=(max(global_x // local[0], 1), max(global_y // local[1], 1), 1),
		)

	if kernel_type != "gemm":
		raise ValueError(f"Unsupported kernel type: {kernel_type}")

	is_reduction = function_name.endswith("gemm_2") or function_name == "gemm_2" or function_name.endswith("__clang_ocl_kern_imp_gemm_2")

	local_sizes = [0, 0, 0]
	global_sizes = [0, 0, 0]
	dim_specs = [
		("l_1", config["ocl_dim_l_1"], config["num_wg_l_1"], config["num_wi_l_1"]),
		("l_2", config["ocl_dim_l_2"], config["num_wg_l_2"], config["num_wi_l_2"]),
		("r_1", config["ocl_dim_r_1"], config["num_wg_r_1"], config["num_wi_r_1"]),
	]

	for dim_name, ocl_dim, num_wg, num_wi in dim_specs:
		local_sizes[ocl_dim] += int(num_wi)
		effective_num_wg = 1 if (is_reduction and dim_name == "r_1") else int(num_wg)
		global_sizes[ocl_dim] += effective_num_wg * int(num_wi)

	for i in range(3):
		local_sizes[i] = max(local_sizes[i], 1)
		global_sizes[i] = max(global_sizes[i], local_sizes[i])

	return LaunchGeometry(
		local_sizes=tuple(local_sizes),
		num_groups=tuple(max(global_sizes[i] // local_sizes[i], 1) for i in range(3)),
	)


def _build_arg_lengths(kernel_type: str, function_name: str, config: Dict, arg_count: int) -> List[int]:
	if kernel_type == "gaussian":
		lengths = [
			(int(config["input_size_h"]) + 4) * (int(config["input_size_w"]) + 4),
			int(config["input_size_h"]) * int(config["input_size_w"]),
			int(config["input_size_h"]) * int(config["input_size_w"]),
		]
		if arg_count != len(lengths):
			raise ValueError(f"Gaussian proxy expected 3 args, found {arg_count}")
		return lengths

	if kernel_type != "gemm":
		raise ValueError(f"Unsupported kernel type: {kernel_type}")

	m = int(config["M"])
	n = int(config["N"])
	k = int(config["K"])
	int_res_size = m * n * int(config["num_wg_r_1"])
	res_g_size = m * n
	if int(config["g_cb_res_dest_level"]) == 2:
		res_g_size *= int(config["num_wg_r_1"])
	if int(config["l_cb_res_dest_level"]) == 2:
		res_g_size *= int(config["num_wi_r_1"])

	if function_name.endswith("gemm_1") or function_name.endswith("__clang_ocl_kern_imp_gemm_1") or function_name == "gemm_1":
		lengths = [m * k, k * n, res_g_size, int_res_size]
	else:
		lengths = [int_res_size, res_g_size, m * n]

	if arg_count != len(lengths):
		raise ValueError(
			f"GEMM proxy expected {len(lengths)} args for {function_name}, found {arg_count}"
		)
	return lengths


def _build_proxy_plan(llvm_text: str, config: Dict, explicit_function: Optional[str] = None) -> ProxyBuildPlan:
	kernel_type = _infer_kernel_type(config)
	target_function = _resolve_target_function(llvm_text, explicit=explicit_function)
	signature_parts = _parse_function_signature(llvm_text, target_function)
	if not signature_parts or any(not part.startswith("ptr") for part in signature_parts):
		raise ValueError(f"Only pointer-only kernel helpers are supported, got: {signature_parts}")

	instrumented_text, block_names = _instrument_function(llvm_text, target_function)
	arg_lengths = _build_arg_lengths(kernel_type, target_function, config, len(signature_parts))
	launch = _build_launch_geometry(kernel_type, target_function, config)
	return ProxyBuildPlan(
		target_function=target_function,
		block_names=block_names,
		arg_lengths=arg_lengths,
		launch=launch,
	), instrumented_text


def _normalize_runtime_ir_for_host_execution(llvm_text: str) -> str:
	normalized = llvm_text
	if RUNTIME_SPIR_TRIPLE_RE.search(normalized):
		normalized = RUNTIME_SPIR_DATALAYOUT_RE.sub(
			'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"',
			normalized,
			count=1,
		)
		normalized = RUNTIME_SPIR_TRIPLE_RE.sub(
			'target triple = "x86_64-unknown-linux-gnu"',
			normalized,
			count=1,
		)
	normalized = ADDRSPACE_PTR_RE.sub("ptr", normalized)
	return STANDALONE_ADDRSPACE_RE.sub("", normalized)


def _generate_driver_c(plan: ProxyBuildPlan) -> str:
	arg_decls = []
	call_args = []
	free_lines = []
	for idx, length in enumerate(plan.arg_lengths):
		allocation_length = max(length * 4, 1 << 18)
		arg_decls.append(
			f"  float *arg{idx} = (float *)calloc({allocation_length}ULL, sizeof(float));\n"
			f"  if (arg{idx} == NULL) {{\n"
			f'    fprintf(stderr, "Failed to allocate buffer {idx}\\n");\n'
			f"    return 2;\n"
			f"  }}\n"
		)
		call_args.append(f"arg{idx}")
		free_lines.append(f"  free(arg{idx});")

	extern_args = ", ".join("float *" for _ in plan.arg_lengths)
	call_expr = ", ".join(call_args)

	return f"""#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t bb_counts[{len(plan.block_names)}];
static uint64_t group_ids[3];
static uint64_t local_ids[3];

uint64_t get_group_id(unsigned dim) __asm__("_Z12get_group_idj");
uint64_t get_local_id(unsigned dim) __asm__("_Z12get_local_idj");
void opencl_barrier(unsigned flags) __asm__("_Z7barrierj");

uint64_t get_group_id(unsigned dim) {{
  return dim < 3 ? group_ids[dim] : 0;
}}

uint64_t get_local_id(unsigned dim) {{
  return dim < 3 ? local_ids[dim] : 0;
}}

void __bb_hit(int idx) {{
  if (idx >= 0 && idx < {len(plan.block_names)}) {{
    bb_counts[idx] += 1;
  }}
}}

void opencl_barrier(unsigned flags) {{
  (void)flags;
}}

extern void {plan.target_function}({extern_args});

int main(int argc, char **argv) {{
{''.join(arg_decls)}
  if (argc != 7) {{
    fprintf(stderr, "Expected 6 runtime id arguments, got %d\\n", argc - 1);
    return 3;
  }}
  group_ids[0] = strtoull(argv[1], NULL, 10);
  group_ids[1] = strtoull(argv[2], NULL, 10);
  group_ids[2] = strtoull(argv[3], NULL, 10);
  local_ids[0] = strtoull(argv[4], NULL, 10);
  local_ids[1] = strtoull(argv[5], NULL, 10);
  local_ids[2] = strtoull(argv[6], NULL, 10);
  {plan.target_function}({call_expr});

  for (int i = 0; i < {len(plan.block_names)}; ++i) {{
    printf("BBCOUNT %d %llu\\n", i, (unsigned long long)bb_counts[i]);
  }}

{os.linesep.join(free_lines)}
  return 0;
}}
"""


def _collect_dynamic_counts(output: str, block_names: Sequence[str]) -> Dict[str, int]:
	dynamic_counts: Dict[str, int] = {}
	for line in output.splitlines():
		match = re.match(r"BBCOUNT\s+(\d+)\s+(\d+)", line.strip())
		if not match:
			continue
		idx = int(match.group(1))
		count = int(match.group(2))
		dynamic_counts[block_names[idx]] = count

	if len(dynamic_counts) != len(block_names):
		raise RuntimeError(
			f"Expected {len(block_names)} dynamic counters, got {len(dynamic_counts)}.\n"
			f"stdout:\n{output}"
		)
	return dynamic_counts


def _candidate_runtime_ids(plan: ProxyBuildPlan) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
	def choices(extent: int) -> List[int]:
		if extent <= 1:
			return [0]
		values = {0, 1, extent - 1}
		return sorted(values)

	group_options = [choices(extent) for extent in plan.launch.num_groups]
	local_options = [choices(extent) for extent in plan.launch.local_sizes]
	return [
		(tuple(group_ids), tuple(local_ids))
		for group_ids in itertools.product(*group_options)
		for local_ids in itertools.product(*local_options)
	]


def _dynamic_bb_counts(
	llvm_ir_path: Path,
	config_path: Path,
	kernel_function: Optional[str] = None,
) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int, int]]]:
	_require_tool("clang")
	_require_tool("symb-viewer")

	llvm_text = llvm_ir_path.read_text()
	config = _load_json(config_path)
	plan, instrumented_text = _build_proxy_plan(llvm_text, config, explicit_function=kernel_function)

	with tempfile.TemporaryDirectory(prefix="bb_proxy_") as tmpdir:
		tmp = Path(tmpdir)
		instrumented_ir = tmp / "instrumented.ll"
		driver_c = tmp / "driver.c"
		exe_path = tmp / "bb_proxy.exe"

		instrumented_ir.write_text(_normalize_runtime_ir_for_host_execution(instrumented_text))
		driver_c.write_text(_generate_driver_c(plan))

		_run(["clang", str(instrumented_ir), str(driver_c), "-O0", "-o", str(exe_path)], cwd=ROOT)

		first_dynamic: Optional[Dict[str, int]] = None
		first_ids: Optional[Dict[str, Tuple[int, int, int]]] = None
		for group_ids, local_ids in _candidate_runtime_ids(plan):
			result = _run(
				[
					str(exe_path),
					str(group_ids[0]),
					str(group_ids[1]),
					str(group_ids[2]),
					str(local_ids[0]),
					str(local_ids[1]),
					str(local_ids[2]),
				],
				cwd=ROOT,
			)
			dynamic_counts = _collect_dynamic_counts(result.stdout, plan.block_names)
			if first_dynamic is None:
				first_dynamic = dynamic_counts
				first_ids = {"group_ids": group_ids, "local_ids": local_ids}

	if first_dynamic is None or first_ids is None:
		raise RuntimeError("Failed to execute any dynamic proxy candidates")
	return first_dynamic, first_ids


def _parse_symb_count(raw_count: str) -> int:
	if raw_count.startswith("#x"):
		return int(raw_count[2:], 16)
	if raw_count.startswith("0x"):
		return int(raw_count, 16)
	return int(raw_count)


def _first_runtime_call_dims(llvm_text: str, function_name: str) -> Dict[str, int]:
	lines = llvm_text.splitlines(keepends=True)
	start, end, _ = _find_function_region(lines, function_name)
	body_lines = lines[start + 1:end]
	pattern = re.compile(r"call\s+.*?@(?P<callee>[^(]+)\(i32(?:\s+\w+)*\s+(?P<dim>\d+)\)")
	dims: Dict[str, int] = {}
	for line in body_lines:
		match = pattern.search(line)
		if not match:
			continue
		callee = match.group("callee")
		if callee in {"_Z12get_group_idj", "_Z12get_local_idj"} and callee not in dims:
			dims[callee] = int(match.group("dim"))
	return dims


def _build_symb_substitutions(
	llvm_text: str,
	function_name: str,
	runtime_ids: RuntimeIdSelection,
) -> Dict[str, int]:
	call_dims = _first_runtime_call_dims(llvm_text, function_name)
	substitutions: Dict[str, int] = {}
	for callee, dim in call_dims.items():
		if callee == "_Z12get_group_idj":
			value = int(runtime_ids.group_ids[dim])
		elif callee == "_Z12get_local_idj":
			value = int(runtime_ids.local_ids[dim])
		else:
			continue
		substitutions[f"call_ret_{callee}"] = value
	return substitutions


def _symb_viewer_bb_counts(
	llvm_ir_path: Path,
	function_name: str,
	substitutions: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, int], Dict[str, str], str]:
	_require_tool("symb-viewer")
	with tempfile.TemporaryDirectory(prefix="symb_bb_") as tmpdir:
		tmp = Path(tmpdir)
		output_json = tmp / "bb_counts.json"
		cmd = [
			"symb-viewer",
			"formula",
			str(llvm_ir_path),
			function_name,
			f"--json={output_json}",
		]
		if substitutions:
			subs_json = tmp / "symb_subs.json"
			subs_json.write_text(json.dumps(substitutions, indent=2, sort_keys=True))
			cmd.append(f"-subs={subs_json}")
		result = _run(cmd, cwd=ROOT)
		payload = json.loads(output_json.read_text())
		tool_messages = "\n".join(part for part in [result.stderr.strip(), result.stdout.strip()] if part).strip()

	counts: Dict[str, int] = {}
	raw_counts: Dict[str, str] = {}
	non_integer_counts: Dict[str, str] = {}
	for graph in payload.get("basic_graphs", []):
		if graph.get("graph_type") != "BasicBlock":
			continue
		name = graph.get("name")
		raw_count = str(graph.get("count"))
		block_name = str(name)
		raw_counts[block_name] = raw_count
		try:
			counts[block_name] = _parse_symb_count(raw_count)
		except ValueError:
			non_integer_counts[block_name] = raw_count
	if non_integer_counts:
		raise SymbolicCountParseError(non_integer_counts)
	return counts, raw_counts, tool_messages


def compare_bb_counts(
	llvm_ir_path: Path,
	config_path: Path,
	kernel_function: Optional[str] = None,
) -> Dict[str, object]:
	llvm_text = llvm_ir_path.read_text()
	target_function = _resolve_target_function(llvm_text, explicit=kernel_function)
	dynamic_counts, matched_ids = _dynamic_bb_counts(
		llvm_ir_path,
		config_path,
		kernel_function=target_function,
	)
	runtime_ids = RuntimeIdSelection(
		group_ids=tuple(int(value) for value in matched_ids["group_ids"]),
		local_ids=tuple(int(value) for value in matched_ids["local_ids"]),
	)
	symb_substitutions = _build_symb_substitutions(llvm_text, target_function, runtime_ids)
	symb_counts, symb_raw_counts, substituted_tool_messages = _symb_viewer_bb_counts(
		llvm_ir_path,
		target_function,
		substitutions=symb_substitutions,
	)

	missing_dynamic = sorted(set(symb_counts) - set(dynamic_counts))
	if missing_dynamic:
		raise AssertionError(
			"Dynamic proxy did not report all symb-viewer basic blocks.\n"
			f"missing in dynamic counts: {missing_dynamic}"
		)
	ignored_dynamic = sorted(set(dynamic_counts) - set(symb_counts))
	mismatches = {
		name: {"symb_viewer": symb_counts[name], "dynamic": dynamic_counts[name]}
		for name in sorted(symb_counts)
		if symb_counts[name] != dynamic_counts[name]
	}

	return {
		"llvm_ir_path": str(llvm_ir_path),
		"config_path": str(config_path),
		"kernel_function": target_function,
		"bb_counts": symb_counts,
		"symb_raw_counts": symb_raw_counts,
		"symb_tool_messages": substituted_tool_messages,
		"symb_substitutions": symb_substitutions,
		"dynamic_counts": dynamic_counts,
		"ignored_dynamic_only_blocks": ignored_dynamic,
		"matched_runtime_ids": matched_ids,
		"mismatches": mismatches,
	}


def validate_bb_counts(
	llvm_ir_path: Path,
	config_path: Path,
	kernel_function: Optional[str] = None,
) -> Dict[str, object]:
	comparison = compare_bb_counts(llvm_ir_path, config_path, kernel_function=kernel_function)
	mismatches = comparison["mismatches"]
	if mismatches:
		raise AssertionError(json.dumps(mismatches, indent=2, sort_keys=True))

	return {
		"llvm_ir_path": comparison["llvm_ir_path"],
		"config_path": comparison["config_path"],
		"kernel_function": comparison["kernel_function"],
		"bb_counts": comparison["bb_counts"],
		"symb_raw_counts": comparison["symb_raw_counts"],
		"symb_tool_messages": comparison["symb_tool_messages"],
		"symb_substitutions": comparison["symb_substitutions"],
		"dynamic_counts": comparison["dynamic_counts"],
		"ignored_dynamic_only_blocks": comparison["ignored_dynamic_only_blocks"],
		"matched_runtime_ids": comparison["matched_runtime_ids"],
	}


def _build_validation_run_dir(
	validation_root: Path,
	start_seed: int,
	total_cases: int,
	experiment_name: Optional[str] = None,
) -> Path:
	if experiment_name:
		run_dir = validation_root / experiment_name
	else:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		run_dir = validation_root / f"run_{timestamp}_seed{start_seed}_count{total_cases}"
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def _payload_path_for_run(run_dir: Path) -> Path:
	return run_dir / DEFAULT_VALIDATION_PAYLOAD


def _find_reusable_payload(run_dir: Path) -> Optional[Path]:
	payload_path = _payload_path_for_run(run_dir)
	if payload_path.exists():
		return payload_path
	return None


def _ensure_visualization_launcher(run_dir: Path) -> None:
	launcher_path = run_dir / "run_visualization.py"
	launcher_code = """#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
	current = Path(__file__).resolve()
	repo_root = current.parents[2]
	script_path = repo_root / "scripts" / "dynamic_bb_count_validation.py"
	if not script_path.exists():
		print(f"Validation script not found: {script_path}", file=sys.stderr)
		return 1

	default_payload = current.parent / "payload.json"
	sys.argv = [str(script_path), "--load-json", str(default_payload), *sys.argv[1:]]
	runpy.run_path(str(script_path), run_name="__main__")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
"""
	launcher_path.write_text(launcher_code)
	os.chmod(launcher_path, 0o755)


def _write_payload(run_dir: Path, payload: Dict[str, object]) -> Path:
	payload_path = _payload_path_for_run(run_dir)
	payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
	return payload_path


def _load_payload(path: Path) -> Dict[str, object]:
	return json.loads(path.read_text())


def _generate_random_artifact_pair(seed: int, validation_root: Path, build_dir: str) -> GeneratedArtifactPair:
	rng = random.Random(seed)
	choices = [
		ExperimentRequest("gaussian", (256, 256), 1),
		ExperimentRequest("gemm", (128, 128, 128), 1),
	]
	request = rng.choice(choices)

	with tempfile.TemporaryDirectory(prefix="bb_random_cfg_") as tmpdir:
		config_root = Path(tmpdir) / "configs"
		resolved = prepare_experiment_configs(
			requests=[request],
			configs_root=config_root,
			device_type="cpu",
			seed=seed,
		)
		if not resolved:
			raise RuntimeError("Failed to generate a random configuration")
		_, config_path = resolved[0]
		config = _load_json(config_path)
		param_hash = hash_config(config)
		kernel_type = _infer_kernel_type(config)
		size_tag = (
			f"{config.get('M', 0)}x{config.get('N', 0)}x{config.get('K', 0)}"
			if kernel_type == "gemm"
			else f"{config.get('input_size_h', 0)}x{config.get('input_size_w', 0)}"
		)
		opencl_binary_path = Path(tmpdir) / "llvm" / f"{kernel_type}_{size_tag}_{param_hash}.opencl.bin"
		runtime_ms = run_kernel(
			kernel_type,
			str(config_path),
			build_dir=build_dir,
			dump_opencl_binary=str(opencl_binary_path),
		)
		if runtime_ms is None:
			raise RuntimeError("Failed to run host kernel and dump OpenCL binary")
		artifacts = runtime_ir_artifacts_from_dump(
			kernel_type,
			config,
			param_hash,
			Path(tmpdir) / "llvm",
			opencl_binary_path,
		)
		if not artifacts:
			raise RuntimeError("Failed to extract runtime LLVM IR for generated config")
		chosen = rng.choice(artifacts)

		persist_dir = validation_root / f"seed_{seed:04d}_{kernel_type}"
		if persist_dir.exists():
			shutil.rmtree(persist_dir)
		persist_dir.mkdir(parents=True, exist_ok=True)
		persist_config = persist_dir / Path(config_path).name
		persist_llvm = persist_dir / Path(chosen["llvm_ir_path"]).name
		shutil.copy2(config_path, persist_config)
		shutil.copy2(chosen["llvm_ir_path"], persist_llvm)
		return GeneratedArtifactPair(
			root_dir=persist_dir,
			llvm_ir_path=persist_llvm,
			config_path=persist_config,
			kernel_function=chosen.get("kernel_function"),
		)


def _resolve_inputs(
	llvm_ir: Optional[str],
	config: Optional[str],
	seed: int,
	validation_root: Path,
	build_dir: str,
) -> Tuple[Path, Path]:
	if llvm_ir and config:
		return Path(llvm_ir).resolve(), Path(config).resolve()
	if llvm_ir or config:
		raise ValueError("Pass both --llvm-ir and --config together, or omit both.")
	artifacts = _generate_random_artifact_pair(seed, validation_root=validation_root, build_dir=build_dir)
	return artifacts.llvm_ir_path, artifacts.config_path


def _classify_skip_reason(reason: str) -> str:
	if reason.startswith("{"):
		return "mismatched"
	if "invalid literal for int()" in reason or "bvmul" in reason or "TR_den_" in reason:
		return "symbolic"
	if "SIGABRT" in reason or "died with <Signals." in reason or "host kernel" in reason:
		return "crashed"
	if "not found in PATH" in reason:
		return "tooling"
	return "other"


def _parse_mismatches(reason: str) -> Dict[str, Dict[str, int]]:
	if not reason.startswith("{"):
		return {}
	try:
		payload = json.loads(reason)
	except json.JSONDecodeError:
		return {}
	if not isinstance(payload, dict):
		return {}

	parsed: Dict[str, Dict[str, int]] = {}
	for block_name, values in payload.items():
		if not isinstance(values, dict):
			continue
		try:
			parsed[str(block_name)] = {
				"symb_viewer": int(values["symb_viewer"]),
				"dynamic": int(values["dynamic"]),
			}
		except (KeyError, TypeError, ValueError):
			continue
	return parsed


def _build_case_record(
	case_index: int,
	seed: Optional[int],
	comparison: Optional[Dict[str, object]] = None,
	reason: str = "",
	artifact_root: Optional[Path] = None,
	symb_raw_counts: Optional[Dict[str, str]] = None,
	symb_tool_messages: str = "",
) -> Dict[str, object]:
	mismatches: Dict[str, Dict[str, int]] = {}
	if comparison is not None:
		mismatches = dict(comparison.get("mismatches", {}))  # type: ignore[arg-type]
	elif reason:
		mismatches = _parse_mismatches(reason)

	status = "matched"
	reason_category = "matched"
	if comparison is None:
		reason_category = _classify_skip_reason(reason)
		status = "mismatched" if mismatches else reason_category
	elif mismatches:
		status = "mismatched"
		reason_category = "mismatched"

	symb_counts = dict(comparison.get("bb_counts", {})) if comparison else {}
	if symb_raw_counts is None:
		symb_raw_counts = dict(comparison.get("symb_raw_counts", {})) if comparison else {}
	if not symb_tool_messages and comparison is not None:
		symb_tool_messages = str(comparison.get("symb_tool_messages", ""))
	dynamic_counts = dict(comparison.get("dynamic_counts", {})) if comparison else {}
	block_names = sorted(set(symb_counts) | set(symb_raw_counts) | set(dynamic_counts) | set(mismatches))
	basic_blocks = []
	for block_name in block_names:
		symb_value = symb_counts.get(block_name)
		symb_raw_value = symb_raw_counts.get(block_name)
		dynamic_value = dynamic_counts.get(block_name)
		if block_name in mismatches:
			symb_value = mismatches[block_name]["symb_viewer"]
			dynamic_value = mismatches[block_name]["dynamic"]
		delta = None
		if symb_value is not None and dynamic_value is not None:
			delta = dynamic_value - symb_value
		basic_blocks.append(
			{
				"name": block_name,
				"symb_viewer": symb_value,
				"symb_viewer_raw": symb_raw_value,
				"dynamic": dynamic_value,
				"delta": delta,
				"matches": symb_value == dynamic_value,
			}
		)

	return {
		"case_index": case_index,
		"case_label": f"seed-{seed}" if seed is not None else f"case-{case_index + 1:03d}",
		"seed": seed,
		"status": status,
		"reason_category": reason_category,
		"reason": reason,
		"artifact_root": str(artifact_root) if artifact_root else None,
		"llvm_ir_path": comparison.get("llvm_ir_path") if comparison else None,
		"config_path": comparison.get("config_path") if comparison else None,
		"symb_tool_messages": symb_tool_messages,
		"symb_substitutions": comparison.get("symb_substitutions") if comparison else None,
		"kernel_function": comparison.get("kernel_function") if comparison else None,
		"matched_runtime_ids": comparison.get("matched_runtime_ids") if comparison else None,
		"ignored_dynamic_only_blocks": comparison.get("ignored_dynamic_only_blocks", []) if comparison else [],
		"mismatch_count": sum(1 for item in basic_blocks if not item["matches"]),
		"basic_blocks": basic_blocks,
	}


def _run_validation_case(
	case_index: int,
	seed: Optional[int],
	llvm_ir_path: Path,
	config_path: Path,
	kernel_function: Optional[str] = None,
	artifact_root: Optional[Path] = None,
) -> Dict[str, object]:
	try:
		comparison = compare_bb_counts(llvm_ir_path, config_path, kernel_function=kernel_function)
		reason = ""
		if comparison["mismatches"]:
			reason = json.dumps(comparison["mismatches"], sort_keys=True)
		return _build_case_record(
			case_index=case_index,
			seed=seed,
			comparison=comparison,
			reason=reason,
			artifact_root=artifact_root,
		)
	except SymbolicCountParseError as exc:
		return _build_case_record(
			case_index=case_index,
			seed=seed,
			reason=str(exc),
			artifact_root=artifact_root,
			symb_raw_counts=exc.raw_counts,
		)
	except Exception as exc:
		return _build_case_record(
			case_index=case_index,
			seed=seed,
			reason=str(exc).splitlines()[0][:400],
			artifact_root=artifact_root,
		)


def _run_random_validations(
	total_cases: int,
	start_seed: int,
	validation_root: Path,
	build_dir: str,
	max_attempts: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]], List[Dict[str, object]]]:
	if total_cases <= 0:
		raise ValueError("total_cases must be > 0")

	if max_attempts is None:
		max_attempts = total_cases

	results: List[Dict[str, object]] = []
	skipped: List[Dict[str, str]] = []
	cases: List[Dict[str, object]] = []
	current_seed = start_seed

	while len(results) < total_cases and len(results) + len(skipped) < max_attempts:
		try:
			artifacts = _generate_random_artifact_pair(
				seed=current_seed,
				validation_root=validation_root,
				build_dir=build_dir,
			)
		except Exception as exc:
			case_record = _build_case_record(
				case_index=len(cases),
				seed=current_seed,
				reason=str(exc).splitlines()[0][:400],
			)
			cases.append(case_record)
			skipped.append(
				{
					"seed": str(current_seed),
					"reason": str(case_record.get("reason", ""))[:240],
				}
			)
			current_seed += 1
			continue

		case_record = _run_validation_case(
			case_index=len(cases),
			seed=current_seed,
			llvm_ir_path=artifacts.llvm_ir_path,
			config_path=artifacts.config_path,
			kernel_function=artifacts.kernel_function,
			artifact_root=artifacts.root_dir,
		)
		cases.append(case_record)
		if case_record["status"] == "matched":
			results.append(
				{
					"llvm_ir_path": str(artifacts.llvm_ir_path),
					"config_path": str(artifacts.config_path),
					"kernel_function": case_record["kernel_function"],
					"bb_counts": {item["name"]: item["symb_viewer"] for item in case_record["basic_blocks"]},
					"dynamic_counts": {item["name"]: item["dynamic"] for item in case_record["basic_blocks"]},
					"ignored_dynamic_only_blocks": case_record["ignored_dynamic_only_blocks"],
					"matched_runtime_ids": case_record["matched_runtime_ids"],
					"seed": current_seed,
				}
			)
		else:
			skipped.append(
				{
					"seed": str(current_seed),
					"reason": str(case_record.get("reason", ""))[:240],
				}
			)
		current_seed += 1

	return results, skipped, cases


def _build_validation_summary(cases: Sequence[Dict[str, object]]) -> Dict[str, int]:
	summary = {
		"matched": 0,
		"mismatched": 0,
		"symbolic": 0,
		"crashed": 0,
		"tooling": 0,
		"other": 0,
		"attempted": len(cases),
	}
	for case in cases:
		category = str(case.get("status", ""))
		if category == "matched":
			summary["matched"] += 1
		elif category == "mismatched":
			summary["mismatched"] += 1
		elif category == "symbolic":
			summary["symbolic"] += 1
		elif category == "crashed":
			summary["crashed"] += 1
		elif category == "tooling":
			summary["tooling"] += 1
		else:
			summary["other"] += 1
	summary["other_skipped"] = summary["other"]
	return summary


def _build_visualization_payload(cases: Sequence[Dict[str, object]], summary: Dict[str, int]) -> Dict[str, object]:
	return {
		"summary": summary,
		"cases": list(cases),
	}


def _attach_run_metadata(
	payload: Dict[str, object],
	run_dir: Path,
	mode: str,
	seed: int,
	total_cases: int,
	kernel_function: Optional[str],
) -> Dict[str, object]:
	updated = dict(payload)
	updated["run"] = {
		"run_dir": str(run_dir),
		"payload_path": str(_payload_path_for_run(run_dir)),
		"mode": mode,
		"seed": seed,
		"count": total_cases,
		"kernel_function": kernel_function,
		"created_at": datetime.now().isoformat(),
	}
	return updated


def _pick_server_port(host: str, preferred_port: int) -> Tuple[int, bool]:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			sock.bind((host, preferred_port))
			return preferred_port, False
		except OSError:
			sock.bind((host, 0))
			return int(sock.getsockname()[1]), True


def _maybe_reexec_in_venv_for_web(argv: Sequence[str]) -> None:
	if "--no-web" in argv:
		return
	if importlib.util.find_spec("dash") is not None:
		return

	venv_python = ROOT / ".venv" / "bin" / "python"
	if not venv_python.exists():
		return

	if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
		return

	os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *argv])


def main(argv: Optional[Sequence[str]] = None) -> int:
	argv = list(argv) if argv is not None else sys.argv[1:]
	_maybe_reexec_in_venv_for_web(argv)

	parser = argparse.ArgumentParser(
		description=(
			"Build a runnable proxy for an OpenCL LLVM IR kernel, dynamically "
			"instrument every basic block, and investigate the counts against symb-viewer."
		)
	)
	parser.add_argument("--llvm-ir", default=None, help="Path to the target LLVM IR (.ll)")
	parser.add_argument("--config", default=None, help="Path to the matching kernel config JSON")
	parser.add_argument(
		"--kernel-function",
		default=None,
		help="Optional function name to validate. Defaults to __clang_ocl_kern_imp_* when present.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used when generating a config/LLVM pair.",
	)
	parser.add_argument(
		"--count",
		type=int,
		default=30,
		help="How many randomized validations to run when --llvm-ir/--config are omitted.",
	)
	parser.add_argument("--build-dir", default="build", help="Directory containing gaussian/gemm host executables.")
	parser.add_argument("--host", default="127.0.0.1", help="Server host for the investigation UI")
	parser.add_argument("--port", type=int, default=8766, help="Server port for the investigation UI")
	parser.add_argument(
		"--validation-root",
		default=str(ROOT / "validation" / "dynamic_bb_count"),
		help="Directory where generated validation configs and LLVM IR files will be kept.",
	)
	parser.add_argument(
		"--experiment-name",
		default=None,
		help="Optional stable name for this validation run. If the named run already has a saved payload, it is reused.",
	)
	parser.add_argument(
		"--dump-json",
		default=None,
		help="Optional path to write the investigation payload as JSON.",
	)
	parser.add_argument(
		"--load-json",
		default=None,
		help="Load a previously saved investigation payload JSON and visualize it without rerunning validation.",
	)
	parser.add_argument(
		"--no-web",
		action="store_true",
		help="Print the investigation payload as JSON instead of starting the web UI.",
	)
	args = parser.parse_args(argv)
	validation_root = Path(args.validation_root).resolve()
	validation_root.mkdir(parents=True, exist_ok=True)

	if args.load_json:
		payload = _load_payload(Path(args.load_json).resolve())
	elif args.experiment_name:
		named_run_dir = _build_validation_run_dir(
			validation_root,
			args.seed,
			args.count,
			experiment_name=args.experiment_name,
		)
		reusable_payload = _find_reusable_payload(named_run_dir)
		if reusable_payload is not None:
			payload = _load_payload(reusable_payload)
		else:
			_ensure_visualization_launcher(named_run_dir)
			if args.llvm_ir or args.config:
				llvm_ir_path, config_path = _resolve_inputs(
					args.llvm_ir,
					args.config,
					args.seed,
					validation_root=named_run_dir,
					build_dir=args.build_dir,
				)
				case_record = _run_validation_case(
					case_index=0,
					seed=None,
					llvm_ir_path=llvm_ir_path,
					config_path=config_path,
					kernel_function=args.kernel_function,
					artifact_root=llvm_ir_path.parent if not (args.llvm_ir and args.config) else None,
				)
				payload = _attach_run_metadata(
					_build_visualization_payload([case_record], _build_validation_summary([case_record])),
					run_dir=named_run_dir,
					mode="single",
					seed=args.seed,
					total_cases=1,
					kernel_function=args.kernel_function,
				)
			else:
				_results, _skipped, cases = _run_random_validations(
					total_cases=args.count,
					start_seed=args.seed,
					validation_root=named_run_dir,
					build_dir=args.build_dir,
				)
				payload = _attach_run_metadata(
					_build_visualization_payload(cases, _build_validation_summary(cases)),
					run_dir=named_run_dir,
					mode="random",
					seed=args.seed,
					total_cases=args.count,
					kernel_function=args.kernel_function,
				)
			_write_payload(named_run_dir, payload)
	elif args.llvm_ir or args.config:
		single_case_root = _build_validation_run_dir(validation_root, args.seed, 1)
		_ensure_visualization_launcher(single_case_root)
		llvm_ir_path, config_path = _resolve_inputs(
			args.llvm_ir,
			args.config,
			args.seed,
			validation_root=single_case_root,
			build_dir=args.build_dir,
		)
		case_record = _run_validation_case(
			case_index=0,
			seed=None,
			llvm_ir_path=llvm_ir_path,
			config_path=config_path,
			kernel_function=args.kernel_function,
			artifact_root=llvm_ir_path.parent if not (args.llvm_ir and args.config) else None,
		)
		payload = _attach_run_metadata(
			_build_visualization_payload([case_record], _build_validation_summary([case_record])),
			run_dir=single_case_root,
			mode="single",
			seed=args.seed,
			total_cases=1,
			kernel_function=args.kernel_function,
		)
		_write_payload(single_case_root, payload)
	else:
		run_validation_root = _build_validation_run_dir(validation_root, args.seed, args.count)
		_ensure_visualization_launcher(run_validation_root)
		_results, _skipped, cases = _run_random_validations(
			total_cases=args.count,
			start_seed=args.seed,
			validation_root=run_validation_root,
			build_dir=args.build_dir,
		)
		payload = _attach_run_metadata(
			_build_visualization_payload(cases, _build_validation_summary(cases)),
			run_dir=run_validation_root,
			mode="random",
			seed=args.seed,
			total_cases=args.count,
			kernel_function=args.kernel_function,
		)
		_write_payload(run_validation_root, payload)

	if args.dump_json:
		Path(args.dump_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

	if args.no_web:
		print(json.dumps(payload, indent=2, sort_keys=True))
		return 0

	from scripts.internal.visualize_dynamic_bb_counts_web import create_dash_app

	app = create_dash_app(payload)
	actual_port, used_fallback = _pick_server_port(args.host, args.port)
	if used_fallback:
		print(f"Port {args.port} is already in use. Falling back to available port {actual_port}.")
	print(f"Dynamic BB investigation UI started on http://{args.host}:{actual_port}")
	print("Press Ctrl+C to stop.")
	app.run(host=args.host, port=actual_port, debug=False)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
