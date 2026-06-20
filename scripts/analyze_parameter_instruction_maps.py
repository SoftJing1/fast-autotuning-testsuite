#!/usr/bin/env python3
"""Analyze how tuning parameters affect instruction maps using the 30000cfg DB.

Pipeline:
1. Load completed experiment rows and per-template instruction counts from SQLite.
2. Reconstruct launch-scaled instruction maps, matching the interactive visualizer.
3. For each kernel parameter, search for exact groups where all other parameters are
   fixed and only the target parameter varies.
4. Choose one representative group per parameter and summarize how the instruction
   map changes across target values.
5. Record parameters with no exact-match groups and explain the structural reason.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_visualizer = _load_module(
    "visualize_results_web_parameter_analysis",
    ROOT / "scripts" / "internal" / "visualize_results_web.py",
)
_gemm_gen = _load_module(
    "config_gen_gemm_parameter_analysis",
    ROOT / "scripts" / "internal" / "config-gen-gemm.py",
)
_gaussian_gen = _load_module(
    "config_gen_gaussian_parameter_analysis",
    ROOT / "scripts" / "internal" / "config-gen-gaussian.py",
)

INPUT_KEYS = {
    "gaussian": {"input_size_h", "input_size_w", "input_size_1", "input_size_2"},
    "gemm": {"M", "N", "K", "input_size_l_1", "input_size_l_2", "input_size_r_1"},
}

PRIMARY_INPUT_SIZE = {
    "gaussian": "512x512",
    "gemm": "128x128x128",
}

PRIMARY_VARIANT = {
    "gaussian": "primary",
    "gemm": "combined",
}

PARAMETER_DESCRIPTIONS = {
    "gaussian": {
        "images_cache_lcl": "Toggle the input image tile cache in local memory.",
        "images_cache_prv": "Toggle the input image tile cache in private memory.",
        "filter_cache_lcl": "Toggle local-memory caching of the Gaussian filter coefficients.",
        "filter_cache_prv": "Toggle private-memory caching of the Gaussian filter coefficients.",
        "out_cache_prv": "Toggle private accumulation for output values before global writeback.",
        "g_cb_res_dest_level": "Fixed global cache-block destination level in this template family.",
        "l_cb_res_dest_level": "Fixed local cache-block destination level in this template family.",
        "p_cb_res_dest_level": "Fixed private cache-block destination level in this template family.",
        "wg_1_ocl_dim": "Maps Gaussian dimension 1 work-groups onto one OpenCL axis.",
        "wg_2_ocl_dim": "Maps Gaussian dimension 2 work-groups onto one OpenCL axis.",
        "wi_1_ocl_dim": "Maps Gaussian dimension 1 work-items onto one OpenCL axis.",
        "wi_2_ocl_dim": "Maps Gaussian dimension 2 work-items onto one OpenCL axis.",
        "glb_1": "Top-level Gaussian dimension-1 strip factor.",
        "wg_1": "Number of dimension-1 work-groups after the GLB_1 split.",
        "lcl_1": "Logical local-cache tiling factor along Gaussian dimension 1.",
        "wi_1": "OpenCL local-size factor along Gaussian dimension 1.",
        "prv_1": "Per-work-item private strip factor along Gaussian dimension 1.",
        "glb_2": "Top-level Gaussian dimension-2 strip factor.",
        "wg_2": "Number of dimension-2 work-groups after the GLB_2 split.",
        "lcl_2": "Logical local-cache tiling factor along Gaussian dimension 2.",
        "wi_2": "OpenCL local-size factor along Gaussian dimension 2.",
        "prv_2": "Per-work-item private strip factor along Gaussian dimension 2.",
    },
    "gemm": {
        "cache_l_cb": "Toggle local-memory caching of cache blocks.",
        "cache_p_cb": "Toggle private-memory caching of cache blocks.",
        "g_cb_res_dest_level": "Fixed global cache-block destination level in this GEMM family.",
        "l_cb_res_dest_level": "Destination level for resolving the local cache block.",
        "p_cb_res_dest_level": "Destination level for resolving the private cache block.",
        "ocl_dim_l_1": "Map GEMM L_1/M dimension onto one OpenCL axis.",
        "ocl_dim_l_2": "Map GEMM L_2/N dimension onto one OpenCL axis.",
        "ocl_dim_r_1": "Map GEMM R_1/K dimension onto one OpenCL axis.",
        "l_cb_size_l_1": "Local cache-block size for GEMM L_1/M.",
        "p_cb_size_l_1": "Private cache-block size for GEMM L_1/M.",
        "num_wg_l_1": "Number of work-groups for GEMM L_1/M.",
        "num_wi_l_1": "Number of work-items per work-group for GEMM L_1/M.",
        "l_cb_size_l_2": "Local cache-block size for GEMM L_2/N.",
        "p_cb_size_l_2": "Private cache-block size for GEMM L_2/N.",
        "num_wg_l_2": "Number of work-groups for GEMM L_2/N.",
        "num_wi_l_2": "Number of work-items per work-group for GEMM L_2/N.",
        "l_cb_size_r_1": "Local cache-block size for GEMM R_1/K reduction dimension.",
        "p_cb_size_r_1": "Private cache-block size for GEMM R_1/K reduction dimension.",
        "num_wg_r_1": "Number of work-groups for GEMM R_1/K.",
        "num_wi_r_1": "Number of work-items per work-group for GEMM R_1/K.",
        "l_reduction": "Fixed setting selecting the local reduction path.",
        "p_write_back": "Fixed setting for private writeback level.",
        "l_write_back": "Fixed setting for local writeback level.",
    },
}


@dataclass
class VariantRecord:
    kernel_type: str
    input_size: str
    variant: str
    config: Dict[str, Any]
    param_hash: str
    runtime_ms: Optional[float]
    scaled_counts: Dict[str, int]
    raw_counts: Dict[str, int]
    launch_threads: int


def _query_rows(db_path: Path, sql: str) -> List[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _sum_counts(mappings: Iterable[Dict[str, int]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for mapping in mappings:
        for op, value in mapping.items():
            total[op] = total.get(op, 0) + int(value)
    return total


def _load_records(db_path: Path) -> Dict[str, List[VariantRecord]]:
    sql = """
        SELECT e.exp_id, e.kernel_type, e.input_size, e.param_hash, e.config_json, e.runtime_ms,
               lic.template_name, lic.total_instruction_counts_json
        FROM experiments e
        JOIN llvm_instruction_counts lic ON lic.exp_id = e.exp_id
        WHERE e.status = 'completed'
        ORDER BY e.exp_id, lic.template_name
    """
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in _query_rows(db_path, sql):
        config = json.loads(row["config_json"])
        raw_counts = {str(op): int(value) for op, value in json.loads(row["total_instruction_counts_json"]).items()}
        geometry = _visualizer._build_launch_geometry(row["kernel_type"], row["template_name"], config)
        scaled_counts = _visualizer._scale_instruction_counts(raw_counts, geometry.thread_count)
        grouped[int(row["exp_id"])].append(
            {
                "kernel_type": row["kernel_type"],
                "input_size": row["input_size"],
                "param_hash": row["param_hash"],
                "config": config,
                "runtime_ms": row["runtime_ms"],
                "template_name": row["template_name"],
                "raw_counts": raw_counts,
                "scaled_counts": scaled_counts,
                "launch_threads": geometry.thread_count,
            }
        )

    by_kernel: Dict[str, List[VariantRecord]] = defaultdict(list)
    for rows in grouped.values():
        kernel_type = rows[0]["kernel_type"]
        if kernel_type == "gaussian":
            row = rows[0]
            by_kernel[kernel_type].append(
                VariantRecord(
                    kernel_type=kernel_type,
                    input_size=row["input_size"],
                    variant="primary",
                    config=dict(row["config"]),
                    param_hash=row["param_hash"],
                    runtime_ms=row["runtime_ms"],
                    scaled_counts=dict(row["scaled_counts"]),
                    raw_counts=dict(row["raw_counts"]),
                    launch_threads=int(row["launch_threads"]),
                )
            )
            continue

        template_map = {row["template_name"]: row for row in rows}
        primary = template_map["gemm_1"]
        reduction = template_map["gemm_2"]
        by_kernel[kernel_type].append(
            VariantRecord(
                kernel_type=kernel_type,
                input_size=primary["input_size"],
                variant="combined",
                config=dict(primary["config"]),
                param_hash=primary["param_hash"],
                runtime_ms=primary["runtime_ms"],
                scaled_counts=_sum_counts([primary["scaled_counts"], reduction["scaled_counts"]]),
                raw_counts=_sum_counts([primary["raw_counts"], reduction["raw_counts"]]),
                launch_threads=int(primary["launch_threads"]) + int(reduction["launch_threads"]),
            )
        )
    return by_kernel


def _total(mapping: Dict[str, int]) -> int:
    return sum(int(value) for value in mapping.values())


def _top_deltas(base: Dict[str, int], new: Dict[str, int], limit: int = 5) -> List[Dict[str, Any]]:
    rows = []
    for op in sorted(set(base) | set(new)):
        delta = int(new.get(op, 0)) - int(base.get(op, 0))
        if delta:
            rows.append(
                {
                    "opcode": op,
                    "base": int(base.get(op, 0)),
                    "new": int(new.get(op, 0)),
                    "delta": delta,
                }
            )
    rows.sort(key=lambda row: (abs(int(row["delta"])), row["opcode"]), reverse=True)
    return rows[:limit]


def _parameter_list(kernel_type: str, records: Sequence[VariantRecord]) -> List[str]:
    sample = records[0].config
    return sorted(key for key in sample if key not in INPUT_KEYS[kernel_type])


def _other_signature(config: Dict[str, Any], parameter: str) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((key, value) for key, value in config.items() if key != parameter and key not in INPUT_KEYS["gaussian"] and key not in INPUT_KEYS["gemm"]))


def _pick_group(records: Sequence[VariantRecord], parameter: str) -> Optional[List[VariantRecord]]:
    grouped: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[VariantRecord]] = defaultdict(list)
    for record in records:
        key = (record.input_size, _other_signature(record.config, parameter))
        grouped[key].append(record)

    candidates = []
    for key, group in grouped.items():
        distinct_values = sorted({entry.config[parameter] for entry in group})
        if len(distinct_values) < 2:
            continue
        group_sorted = sorted(group, key=lambda entry: (entry.config[parameter], entry.param_hash))
        totals = [_total(entry.scaled_counts) for entry in group_sorted]
        spread = max(totals) - min(totals)
        candidates.append((len(distinct_values), spread, len(group_sorted), key[0], group_sorted[0].param_hash, group_sorted))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    return candidates[0][-1]


def _validator_note(kernel_type: str, parameter: str) -> str:
    if kernel_type == "gaussian":
        if parameter in {"glb_1", "wg_1", "lcl_1", "wi_1", "prv_1", "glb_2", "wg_2", "lcl_2", "wi_2", "prv_2"}:
            return "No exact-match group exists because each Gaussian dimension is an exact product: INPUT_SIZE = GLB * WG * LCL * WI * PRV."
        if parameter in {"g_cb_res_dest_level", "l_cb_res_dest_level", "p_cb_res_dest_level", "wg_1_ocl_dim", "wg_2_ocl_dim", "wi_1_ocl_dim", "wi_2_ocl_dim"}:
            return "No exact-match group exists because this parameter is fixed by the Gaussian generator in the 30000cfg study."
        return "No exact-match group was found in the 30000cfg data."

    if parameter in {
        "l_cb_size_l_1",
        "p_cb_size_l_1",
        "num_wg_l_1",
        "num_wi_l_1",
        "l_cb_size_l_2",
        "p_cb_size_l_2",
        "num_wg_l_2",
        "num_wi_l_2",
        "l_cb_size_r_1",
        "p_cb_size_r_1",
        "num_wg_r_1",
        "num_wi_r_1",
    }:
        return "No exact-match group exists because each GEMM dimension is an exact product: INPUT_SIZE = L_CB_SIZE * NUM_WG * NUM_WI * P_CB_SIZE."
    if parameter in {"ocl_dim_l_1", "ocl_dim_l_2", "ocl_dim_r_1"}:
        return "No exact-match group exists because the three OpenCL dimension assignments must be a permutation of (0, 1, 2)."
    if parameter in {"g_cb_res_dest_level", "l_reduction", "p_write_back", "l_write_back"}:
        return "No exact-match group exists because this parameter is fixed by the GEMM generator in the 30000cfg study."
    if parameter == "p_cb_res_dest_level":
        return "No exact-match group was found; in addition, GEMM enforces P_CB_RES_DEST_LEVEL <= L_CB_RES_DEST_LEVEL."
    return "No exact-match group was found in the 30000cfg data."


def _minimum_example(kernel_type: str, parameter: str, record: Optional[VariantRecord]) -> str:
    if kernel_type == "gaussian":
        if parameter.endswith("_1"):
            return "Example: if GLB_1=2, WG_1=4, LCL_1=2, WI_1=8, then PRV_1 must be INPUT_SIZE_1 / (2*4*2*8)."
        if parameter.endswith("_2"):
            return "Example: if GLB_2=2, WG_2=4, LCL_2=2, WI_2=8, then PRV_2 must be INPUT_SIZE_2 / (2*4*2*8)."
        if parameter.startswith("images_cache") or parameter.startswith("filter_cache") or parameter == "out_cache_prv":
            return "Example: flipping the flag from 0 to 1 toggles a whole cache branch guarded by `#if ... == 1`."
        if "ocl_dim" in parameter:
            return "Example: setting `wi_1_ocl_dim=1` means Gaussian dimension 1 uses `get_local_id(1)` instead of another axis."
        return "Example: the generator emits this parameter as a compile-time `#define`, so template branches and loop macros can constant-fold."

    if parameter.startswith("cache_"):
        return "Example: flipping the flag from 0 to 1 enables cache-block buffering code instead of direct accesses."
    if parameter.startswith("ocl_dim_"):
        return "Example: assigning `ocl_dim_l_1=2` maps the GEMM M dimension onto OpenCL axis 2."
    if parameter.startswith("l_cb_size") or parameter.startswith("p_cb_size") or parameter.startswith("num_wg") or parameter.startswith("num_wi"):
        return "Example: choosing L_CB_SIZE=16, NUM_WG=4, NUM_WI=8 forces P_CB_SIZE = INPUT_SIZE / (16*4*8)."
    return "Example: the generator emits this parameter as a compile-time `#define`, so the macro-expanded control flow changes before LLVM sees the kernel."


def _overall_explanation(kernel_type: str, parameter: str) -> str:
    if kernel_type == "gaussian":
        if parameter.startswith("images_cache"):
            return "Controls whether the input image tile is materialized in a software-managed cache before stencil computation."
        if parameter.startswith("filter_cache"):
            return "Controls whether the small filter coefficients are staged in explicit cache arrays instead of being read on demand."
        if parameter == "out_cache_prv":
            return "Controls whether partial outputs accumulate in private arrays before final stores."
        if "ocl_dim" in parameter:
            return "Controls how logical Gaussian dimensions are bound onto OpenCL x/y axes."
        if parameter in {"glb_1", "wg_1", "lcl_1", "wi_1", "prv_1", "glb_2", "wg_2", "lcl_2", "wi_2", "prv_2"}:
            return "Controls one factor in the exact five-level decomposition of a Gaussian input dimension."
        return "Controls a fixed structural choice in the Gaussian generator."

    if parameter.startswith("cache_"):
        return "Controls whether GEMM cache blocks are explicitly buffered at local or private scope."
    if parameter.endswith("_res_dest_level"):
        return "Controls where a cache-block result is resolved in the GEMM macro hierarchy."
    if parameter.startswith("ocl_dim_"):
        return "Controls how GEMM logical dimensions M/N/K are mapped onto OpenCL axes."
    if parameter.startswith("l_cb_size") or parameter.startswith("p_cb_size") or parameter.startswith("num_wg") or parameter.startswith("num_wi"):
        return "Controls one factor in the exact GEMM dimension decomposition INPUT_SIZE = L_CB_SIZE * NUM_WG * NUM_WI * P_CB_SIZE."
    return "Controls a fixed structural choice in the GEMM generator."


def _summarize_group(kernel_type: str, parameter: str, group: List[VariantRecord]) -> Dict[str, Any]:
    values = sorted({record.config[parameter] for record in group})
    ordered = sorted(group, key=lambda record: (record.config[parameter], record.param_hash))
    base = ordered[0]
    value_rows = []
    for record in ordered:
        row = {
            "value": record.config[parameter],
            "param_hash": record.param_hash,
            "total_scaled_insts": _total(record.scaled_counts),
            "launch_threads": record.launch_threads,
            "runtime_ms": record.runtime_ms,
            "top_opcode_deltas_vs_base": _top_deltas(base.scaled_counts, record.scaled_counts),
        }
        value_rows.append(row)
    return {
        "kernel_type": kernel_type,
        "parameter": parameter,
        "input_size": base.input_size,
        "variant": PRIMARY_VARIANT[kernel_type],
        "description": PARAMETER_DESCRIPTIONS[kernel_type].get(parameter, ""),
        "overall_explanation": _overall_explanation(kernel_type, parameter),
        "minimum_example": _minimum_example(kernel_type, parameter, base),
        "base_value": base.config[parameter],
        "all_values": values,
        "value_rows": value_rows,
        "other_params_fixed": {
            key: value
            for key, value in sorted(base.config.items())
            if key != parameter and key not in INPUT_KEYS[kernel_type]
        },
    }


def _format_opcode_delta_rows(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "No opcode delta versus base."
    parts = []
    for row in rows[:3]:
        sign = "+" if int(row["delta"]) > 0 else ""
        parts.append(f"`{row['opcode']}` {sign}{row['delta']}")
    return ", ".join(parts)


def _write_report(path: Path, analyses: Dict[str, List[Dict[str, Any]]], exact_group_counts: Dict[Tuple[str, str], int]) -> None:
    lines = [
        "# Tuning Parameter to Instruction-Map Report",
        "",
        "Method:",
        "1. Restrict to one canonical input size per kernel family: Gaussian `512x512`, GEMM `128x128x128`.",
        "2. Reconstruct launch-scaled instruction maps from `experiments.db` using the same scaling rules as `visualize_results_web.py`.",
        "3. For each tuning parameter, group records by all other tuning parameters fixed and vary only the target parameter.",
        "4. If one or more exact-match groups exist, choose the representative group with the most distinct target values; ties break by smaller input size then config hash.",
        "5. If no exact-match group exists, record that absence and explain the structural reason from the generator constraints.",
        "",
    ]

    for kernel_type in ("gaussian", "gemm"):
        lines.extend([f"## {kernel_type.capitalize()}", ""])
        for analysis in analyses[kernel_type]:
            parameter = analysis["parameter"]
            lines.append(f"### `{parameter}`")
            lines.append("")
            lines.append(f"Overall: {analysis['overall_explanation']}")
            lines.append("")
            lines.append(f"Meaning: {analysis['description']}")
            lines.append("")
            lines.append(f"Minimum example: {analysis['minimum_example']}")
            lines.append("")
            group_count = exact_group_counts[(kernel_type, parameter)]
            if "value_rows" not in analysis:
                lines.append(f"Instruction-map effect in 30000cfg: no exact-match group found. Exact-match group count = {group_count}.")
                lines.append("")
                lines.append(f"Reason: {analysis['constraint_reason']}")
                lines.append("")
                continue

            lines.append(
                f"Instruction-map effect in 30000cfg: exact-match group count = {group_count}; "
                f"representative input size = `{analysis['input_size']}`; values = {analysis['all_values']}."
            )
            lines.append("")
            lines.append("| Value | Total Scaled Insts | Runtime ms | Top opcode deltas vs base |")
            lines.append("| ---: | ---: | ---: | --- |")
            for row in analysis["value_rows"]:
                runtime = "" if row["runtime_ms"] is None else f"{float(row['runtime_ms']):.6f}"
                lines.append(
                    f"| {row['value']} | {row['total_scaled_insts']} | {runtime} | "
                    f"{_format_opcode_delta_rows(row['top_opcode_deltas_vs_base'])} |"
                )
            lines.append("")
            fixed_keys = ", ".join(f"`{k}={v}`" for k, v in analysis["other_params_fixed"].items())
            lines.append(f"Other fixed tuning parameters in the representative group: {fixed_keys}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run(db_path: Path, output_dir: Path) -> None:
    records_by_kernel = _load_records(db_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses: Dict[str, List[Dict[str, Any]]] = {"gaussian": [], "gemm": []}
    exact_group_counts: Dict[Tuple[str, str], int] = {}

    for kernel_type in ("gaussian", "gemm"):
        records = [record for record in records_by_kernel[kernel_type] if record.input_size == PRIMARY_INPUT_SIZE[kernel_type] and record.variant == PRIMARY_VARIANT[kernel_type]]
        parameters = _parameter_list(kernel_type, records)
        for parameter in parameters:
            grouped: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[VariantRecord]] = defaultdict(list)
            for record in records:
                grouped[(record.input_size, _other_signature(record.config, parameter))].append(record)
            exact_count = 0
            for group in grouped.values():
                if len({entry.config[parameter] for entry in group}) > 1:
                    exact_count += 1
            exact_group_counts[(kernel_type, parameter)] = exact_count

            representative = _pick_group(records, parameter)
            if representative is None:
                analyses[kernel_type].append(
                    {
                        "kernel_type": kernel_type,
                        "parameter": parameter,
                        "description": PARAMETER_DESCRIPTIONS[kernel_type].get(parameter, ""),
                        "overall_explanation": _overall_explanation(kernel_type, parameter),
                        "minimum_example": _minimum_example(kernel_type, parameter, None),
                        "constraint_reason": _validator_note(kernel_type, parameter),
                    }
                )
                continue
            analyses[kernel_type].append(_summarize_group(kernel_type, parameter, representative))

    (output_dir / "parameter_analysis.json").write_text(json.dumps(analyses, indent=2, sort_keys=True))
    (output_dir / "exact_group_counts.json").write_text(
        json.dumps(
            {f"{kernel}:{parameter}": count for (kernel, parameter), count in sorted(exact_group_counts.items())},
            indent=2,
            sort_keys=True,
        )
    )
    _write_report(output_dir / "parameter_instruction_map_report.md", analyses, exact_group_counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="experiments/exp_20260518_large_scale_30000cfg/experiments.db",
        help="Path to the SQLite experiment database.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/exp_20260518_large_scale_30000cfg/parameter_instruction_map_analysis",
        help="Directory for markdown and JSON outputs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    run(db_path, output_dir)
    print(f"Parameter instruction-map analysis written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
