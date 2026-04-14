#!/usr/bin/env python3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from scripts.dynamic_bb_count_validation import _build_case_record


def test_case_record_does_not_count_dynamic_only_blocks_as_mismatches():
	comparison = {
		"llvm_ir_path": "kernel.ll",
		"config_path": "config.json",
		"kernel_function": "gemm_1",
		"bb_counts": {"%entry": 1},
		"symb_raw_counts": {"%entry": "1"},
		"symb_tool_messages": "",
		"symb_substitutions": {},
		"dynamic_counts": {"%entry": 1, "%dynamic.only": 8},
		"ignored_dynamic_only_blocks": ["%dynamic.only"],
		"matched_runtime_ids": {"group_ids": (0, 0, 0), "local_ids": (0, 0, 0)},
		"mismatches": {},
	}

	case = _build_case_record(case_index=0, seed=44, comparison=comparison)

	assert case["status"] == "matched"
	assert case["mismatch_count"] == 0

	dynamic_only = next(row for row in case["basic_blocks"] if row["name"] == "%dynamic.only")
	assert dynamic_only["matches"] is None
	assert dynamic_only["comparable"] is False
	assert dynamic_only["dynamic_only"] is True


def test_case_record_counts_comparable_mismatches():
	comparison = {
		"llvm_ir_path": "kernel.ll",
		"config_path": "config.json",
		"kernel_function": "gemm_2",
		"bb_counts": {"%entry": 1},
		"symb_raw_counts": {"%entry": "1"},
		"symb_tool_messages": "",
		"symb_substitutions": {},
		"dynamic_counts": {"%entry": 2},
		"ignored_dynamic_only_blocks": [],
		"matched_runtime_ids": {"group_ids": (0, 0, 0), "local_ids": (0, 0, 0)},
		"mismatches": {"%entry": {"symb_viewer": 1, "dynamic": 2}},
	}

	case = _build_case_record(case_index=0, seed=45, comparison=comparison)

	assert case["status"] == "mismatched"
	assert case["mismatch_count"] == 1


def test_visualization_recomputes_mismatch_count_for_old_dynamic_only_payload():
	pytest.importorskip("dash")
	from scripts.internal.visualize_dynamic_bb_counts_web import _case_rows, _detail_layout

	case = {
		"case_label": "seed-44",
		"status": "matched",
		"kernel_function": "gemm_1",
		"mismatch_count": 1,
		"ignored_dynamic_only_blocks": ["%for.end438"],
		"basic_blocks": [
			{
				"name": "%entry",
				"symb_viewer": 1,
				"symb_viewer_raw": "1",
				"dynamic": 1,
				"delta": 0,
				"matches": True,
			},
			{
				"name": "%for.end438",
				"symb_viewer": None,
				"symb_viewer_raw": None,
				"dynamic": 8,
				"delta": None,
				"matches": False,
			},
		],
	}

	assert _case_rows([case])[0]["mismatch_count"] == 0

	detail = _detail_layout(case)
	assert detail[0].children == "All 1 comparable basic blocks matched for this case."
	assert detail[1].children == "Ignored dynamic-only block(s): %for.end438"
