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
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class InstCountResult:
	llvm_ir_path: str
	kernel_function: str
	bb_counts: Dict[str, int]
	bb_instruction_counts: Dict[str, Dict[str, int]]
	total_instruction_counts: Dict[str, int]

	def to_dict(self) -> Dict[str, Any]:
		return {
			"llvm_ir_path": self.llvm_ir_path,
			"kernel_function": self.kernel_function,
			"bb_counts": self.bb_counts,
			"bb_instruction_counts": self.bb_instruction_counts,
			"total_instruction_counts": self.total_instruction_counts,
		}


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


def _parse_hex_count(value: str) -> int:
	if value.startswith("#x"):
		return int(value[2:], 16)
	if value.startswith("0x"):
		return int(value, 16)
	return int(value)


def _run_symb_viewer(command: List[str]) -> None:
	try:
		subprocess.run(command, check=True, capture_output=True, text=True)
	except FileNotFoundError as exc:
		raise RuntimeError("`symb-viewer` not found in PATH") from exc
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(
			f"symb-viewer command failed: {' '.join(command)}\n"
			f"stdout: {exc.stdout}\nstderr: {exc.stderr}"
		) from exc


def _run_instr_count(llvm_ir_path: Path, kernel_function: str, output_json: Path) -> None:
	# Try the subcommand used in the doc first, then fallback.
	cmd_candidates = [
		["symb-viewer", "inst-count", str(llvm_ir_path), kernel_function, "-o", str(output_json)],
	]
	last_error: Optional[Exception] = None
	for cmd in cmd_candidates:
		try:
			_run_symb_viewer(cmd)
			return
		except Exception as exc:
			last_error = exc
	if last_error:
		raise last_error
	raise RuntimeError("Failed to run symb-viewer instruction count command")


def _run_formula(llvm_ir_path: Path, kernel_function: str, output_json: Path) -> None:
	cmd = [
		"symb-viewer",
		"formula",
		str(llvm_ir_path),
		kernel_function,
		f"--json={output_json}",
	]
	_run_symb_viewer(cmd)


def _load_json(path: Path) -> Any:
	with open(path) as f:
		return json.load(f)


def _extract_bb_counts(formula_json: Dict[str, Any]) -> Dict[str, int]:
	counts: Dict[str, int] = {}
	for graph in formula_json.get("basic_graphs", []):
		if graph.get("graph_type") != "BasicBlock":
			continue
		name = graph.get("name")
		count_raw = graph.get("count")
		if not name or count_raw is None:
			continue
		counts[str(name)] = _parse_hex_count(str(count_raw))
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

		_run_instr_count(llvm_path, resolved_kernel_function, instr_json_path)
		_run_formula(llvm_path, resolved_kernel_function, bb_json_path)

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
