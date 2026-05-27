#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from scripts.internal import llvm_to_inst_count as inst_count


def test_build_symb_substitutions_uses_runtime_id_symbols():
	llvm_text = """
define void @gemm_2(ptr %a) {
entry:
  %0 = call i64 @_Z12get_group_idj(i32 2)
  %1 = call i64 @_Z12get_local_idj(i32 1)
  ret void
}
"""
	runtime_ids = inst_count.RuntimeIdSelection(
		group_ids=(3, 4, 5),
		local_ids=(6, 7, 8),
	)

	assert inst_count._build_symb_substitutions(llvm_text, "gemm_2", runtime_ids) == {
		"call_ret__Z12get_group_idj": 5,
		"call_ret__Z12get_local_idj": 7,
	}


def test_dynamic_validation_and_collection_share_substitution_logic():
	from scripts.dynamic_bb_count_validation import (
		RuntimeIdSelection,
		_build_symb_substitutions,
	)

	llvm_text = """
define void @gaussian_1(ptr %a) {
entry:
  %0 = call i64 @_Z12get_group_idj(i32 0)
  ret void
}
"""
	runtime_ids = RuntimeIdSelection(group_ids=(9, 0, 0), local_ids=(0, 0, 0))

	assert _build_symb_substitutions(llvm_text, "gaussian_1", runtime_ids) == {
		"call_ret__Z12get_group_idj": 9,
	}


def test_run_formula_passes_substitution_file(tmp_path, monkeypatch):
	calls = []

	def fake_run_symb_viewer(cmd):
		calls.append(cmd)
		return inst_count.SymbViewerInvocation(command=list(cmd), stdout="", stderr="", returncode=0)

	monkeypatch.setattr(inst_count, "_run_symb_viewer", fake_run_symb_viewer)

	output_json = tmp_path / "bbcount.json"
	inst_count._run_formula(
		tmp_path / "kernel.ll",
		"gemm_2",
		output_json,
		substitutions={"call_ret__Z12get_group_idj": 0},
	)

	assert len(calls) == 1
	assert calls[0][:4] == [
		"symb-viewer",
		"formula",
		str(tmp_path / "kernel.ll"),
		"gemm_2",
	]
	assert calls[0][4] == f"--json={output_json}"

	subs_arg = next(arg for arg in calls[0] if arg.startswith("-subs="))
	subs_path = Path(subs_arg.removeprefix("-subs="))
	assert json.loads(subs_path.read_text()) == {"call_ret__Z12get_group_idj": 0}


def test_inst_count_result_to_dict_includes_raw_json():
	result = inst_count.InstCountResult(
		llvm_ir_path="kernel.ll",
		kernel_function="gemm_1",
		bb_counts={"entry": 1},
		bb_instruction_counts={"entry": {"add": 2}},
		total_instruction_counts={"add": 2},
		raw_instr_count_json=[{"block_name": "entry", "instruction_counts": {"add": 2}}],
		raw_bb_count_json={"basic_graphs": [{"graph_type": "BasicBlock", "name": "entry", "count": "1"}]},
		symb_viewer_invocations=(),
	)

	payload = result.to_dict()

	assert payload["raw_instr_count_json"] == [{"block_name": "entry", "instruction_counts": {"add": 2}}]
	assert payload["raw_bb_count_json"] == {
		"basic_graphs": [{"graph_type": "BasicBlock", "name": "entry", "count": "1"}]
	}


def test_symbolic_basic_block_count_is_not_silently_negative():
	payload = {
		"basic_graphs": [
			{
				"graph_type": "BasicBlock",
				"name": "%if.then",
				"count": "(ite (= call_ret__Z12get_group_idj #x0) #x1 #x0)",
			},
		],
	}

	with pytest.raises(inst_count.SymbolicCountParseError):
		inst_count._extract_bb_counts(payload)
