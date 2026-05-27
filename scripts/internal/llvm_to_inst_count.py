#!/usr/bin/env python3
"""Extract per-instruction total execution counts from LLVM IR.

This script uses `symb-viewer` subcommands to obtain:
1) per-basic-block instruction counts
2) per-basic-block execution counts

Then computes aggregated total counts per instruction:
  total(inst) = sum_over_blocks(bb_exec_count * bb_inst_count(inst))
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SymbViewerInvocation:
	command: List[str]
	stdout: str
	stderr: str
	returncode: int

	def to_dict(self) -> Dict[str, Any]:
		return {
			"command": list(self.command),
			"stdout": self.stdout,
			"stderr": self.stderr,
			"returncode": self.returncode,
		}


@dataclass
class InstCountResult:
	llvm_ir_path: str
	kernel_function: str
	bb_counts: Dict[str, int]
	bb_instruction_counts: Dict[str, Dict[str, int]]
	total_instruction_counts: Dict[str, int]
	raw_instr_count_json: List[Dict[str, Any]]
	raw_bb_count_json: Dict[str, Any]
	symb_viewer_invocations: Tuple[SymbViewerInvocation, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"llvm_ir_path": self.llvm_ir_path,
			"kernel_function": self.kernel_function,
			"bb_counts": self.bb_counts,
			"bb_instruction_counts": self.bb_instruction_counts,
			"total_instruction_counts": self.total_instruction_counts,
			"raw_instr_count_json": self.raw_instr_count_json,
			"raw_bb_count_json": self.raw_bb_count_json,
			"symb_viewer_invocations": [invocation.to_dict() for invocation in self.symb_viewer_invocations],
		}


@dataclass(frozen=True)
class RuntimeIdSelection:
	group_ids: Tuple[int, int, int]
	local_ids: Tuple[int, int, int]


class SymbolicCountParseError(RuntimeError):
	def __init__(self, raw_counts: Dict[str, str]):
		self.raw_counts = raw_counts
		preview = ", ".join(f"{name}={value}" for name, value in list(raw_counts.items())[:5])
		super().__init__(f"Non-integer symb-viewer basic block counts: {preview}")


# The validation proxy compares symb-viewer against the first candidate runtime
# ID selection, which is all zeroes for every active group/local dimension.
DEFAULT_RUNTIME_IDS = RuntimeIdSelection(
	group_ids=(0, 0, 0),
	local_ids=(0, 0, 0),
)


def _resolve_kernel_function(llvm_ir_path: Path, kernel_function: Optional[str]) -> str:
	if kernel_function:
		return kernel_function
	name = llvm_ir_path.stem.lower()
	if name.endswith("_gemm_2") or "gemm_2" in name:
		return "gemm_2"
	if name.endswith("_gemm_1") or name.startswith("gemm_"):
		return "gemm_1"
	if name.endswith("_gaussian_static_1") or name.endswith("_gaussian_1") or name.startswith("gaussian_"):
		return "gaussian_1"
	raise ValueError(
		"Could not infer kernel function from LLVM filename. "
		"Please pass --kernel-function explicitly."
	)


def _parse_symb_count(value: str) -> int:
	if value.startswith("#x"):
		return int(value[2:], 16)
	if value.startswith("0x"):
		return int(value, 16)
	return int(value)


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
	runtime_ids: RuntimeIdSelection = DEFAULT_RUNTIME_IDS,
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


def _run_symb_viewer(command: List[str]) -> SymbViewerInvocation:
	try:
		result = subprocess.run(command, check=True, capture_output=True, text=True)
		return SymbViewerInvocation(
			command=list(command),
			stdout=result.stdout,
			stderr=result.stderr,
			returncode=result.returncode,
		)
	except FileNotFoundError as exc:
		raise RuntimeError("`symb-viewer` not found in PATH") from exc
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(
			f"symb-viewer command failed: {' '.join(command)}\n"
			f"stdout: {exc.stdout}\nstderr: {exc.stderr}"
		) from exc


def _run_instr_count(llvm_ir_path: Path, kernel_function: str, output_json: Path) -> SymbViewerInvocation:
	# Try the subcommand used in the doc first, then fallback.
	cmd_candidates = [
		["symb-viewer", "inst-count", str(llvm_ir_path), kernel_function, "-o", str(output_json)],
	]
	last_error: Optional[Exception] = None
	for cmd in cmd_candidates:
		try:
			return _run_symb_viewer(cmd)
		except Exception as exc:
			last_error = exc
	if last_error:
		raise last_error
	raise RuntimeError("Failed to run symb-viewer instruction count command")


def _run_formula(
	llvm_ir_path: Path,
	kernel_function: str,
	output_json: Path,
	substitutions: Optional[Dict[str, int]] = None,
) -> SymbViewerInvocation:
	cmd = [
		"symb-viewer",
		"formula",
		str(llvm_ir_path),
		kernel_function,
		f"--json={output_json}",
	]
	if substitutions:
		subs_json = output_json.parent / "symb_subs.json"
		subs_json.write_text(json.dumps(substitutions, indent=2, sort_keys=True))
		cmd.append(f"-subs={subs_json}")
	return _run_symb_viewer(cmd)


def _load_json(path: Path) -> Any:
	with open(path) as f:
		return json.load(f)


def _extract_bb_counts(formula_json: Dict[str, Any]) -> Dict[str, int]:
	counts: Dict[str, int] = {}
	non_integer_counts: Dict[str, str] = {}
	for graph in formula_json.get("basic_graphs", []):
		if graph.get("graph_type") != "BasicBlock":
			continue
		name = graph.get("name")
		count_raw = graph.get("count")
		if not name or count_raw is None:
			continue
		block_name = str(name)
		raw_count = str(count_raw)
		try:
			counts[block_name] = _parse_symb_count(raw_count)
		except ValueError:
			non_integer_counts[block_name] = raw_count
	if non_integer_counts:
		raise SymbolicCountParseError(non_integer_counts)
	return counts


def _extract_bb_instruction_counts(instr_json: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
	result: Dict[str, Dict[str, int]] = {}
	for row in instr_json:
		bb_name = row.get("block_name")
		if not bb_name:
			continue
		inst_map = row.get("instruction_counts", {}) or {}
		normalized: Dict[str, int] = {}
		for inst_name, inst_count in inst_map.items():
			try:
				normalized[str(inst_name)] = int(inst_count)
			except (TypeError, ValueError):
				continue
		result[str(bb_name)] = normalized
	return result


def _aggregate_total_instruction_counts(
	bb_counts: Dict[str, int],
	bb_instruction_counts: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
	total: Dict[str, int] = {}
	for bb_name, inst_map in bb_instruction_counts.items():
		bb_exec = bb_counts.get(bb_name, 0)
		for inst_name, inst_count in inst_map.items():
			total[inst_name] = total.get(inst_name, 0) + bb_exec * inst_count
	return total


def extract_instruction_counts(llvm_ir_path: str, kernel_function: Optional[str] = None) -> InstCountResult:
	llvm_path = Path(llvm_ir_path).resolve()
	if not llvm_path.exists():
		raise FileNotFoundError(f"LLVM IR file not found: {llvm_path}")
	if shutil.which("symb-viewer") is None:
		raise RuntimeError("`symb-viewer` command is not available in PATH")

	resolved_kernel_function = _resolve_kernel_function(llvm_path, kernel_function)

	with tempfile.TemporaryDirectory(prefix="llvm_inst_count_") as tmpdir:
		tmp = Path(tmpdir)
		instr_json_path = tmp / "instrcount.json"
		bb_json_path = tmp / "bbcount.json"

		instr_invocation = _run_instr_count(llvm_path, resolved_kernel_function, instr_json_path)
		symb_substitutions = _build_symb_substitutions(
			llvm_path.read_text(),
			resolved_kernel_function,
		)
		formula_invocation = _run_formula(
			llvm_path,
			resolved_kernel_function,
			bb_json_path,
			substitutions=symb_substitutions,
		)

		instr_json = _load_json(instr_json_path)
		bb_json = _load_json(bb_json_path)

	bb_counts = _extract_bb_counts(bb_json)
	bb_instruction_counts = _extract_bb_instruction_counts(instr_json)
	total_instruction_counts = _aggregate_total_instruction_counts(bb_counts, bb_instruction_counts)

	return InstCountResult(
		llvm_ir_path=str(llvm_path),
		kernel_function=resolved_kernel_function,
		bb_counts=bb_counts,
		bb_instruction_counts=bb_instruction_counts,
		total_instruction_counts=total_instruction_counts,
		raw_instr_count_json=instr_json,
		raw_bb_count_json=bb_json,
		symb_viewer_invocations=(instr_invocation, formula_invocation),
	)


def main() -> int:
	parser = argparse.ArgumentParser(description="Extract total per-instruction counts from LLVM IR")
	parser.add_argument("llvm_ir", help="Path to LLVM IR (.ll) file")
	parser.add_argument(
		"--kernel-function",
		default=None,
		help="Kernel function name in IR (e.g. gemm_1, gaussian_1). Inferred from filename if omitted.",
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Optional output JSON path. If omitted, prints JSON to stdout.",
	)
	args = parser.parse_args()

	result = extract_instruction_counts(args.llvm_ir, kernel_function=args.kernel_function)
	payload = result.to_dict()

	if args.output:
		with open(args.output, "w") as f:
			json.dump(payload, f, indent=2, sort_keys=True)
	else:
		print(json.dumps(payload, indent=2, sort_keys=True))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
