"""Parse OpenCL kernel templates to extract tuning parameters.

Provides a small API and a CLI entrypoint.

Functions:
- parse_tuning_parameters(file_path) -> dict
- parse_directory(dir_path, pattern="*.cl") -> dict

CLI: `python -m scripts.get_tuning_parameters [--dir DIR] [--json OUT]`
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List


IDENT_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Z0-9_]+)\b")


def parse_tuning_parameters(file_path: str) -> Dict:
	"""Parse a single OpenCL file and return discovered tuning parameters.

	The heuristic is:
	- collect all UPPER_SNAKE identifiers used in the file
	- collect all identifiers defined with `#define`
	- tuning parameters = used - defined

	Returns a dict with keys: file, parameters (dict{name: {count, lines}})
	"""
	used = defaultdict(lambda: {"count": 0, "lines": []})
	defined = set()

	with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
		for i, line in enumerate(f, start=1):
			m = DEFINE_RE.match(line)
			if m:
				defined.add(m.group(1))
			for ident in IDENT_RE.findall(line):
				used[ident]["count"] += 1
				if len(used[ident]["lines"]) < 20:
					used[ident]["lines"].append(i)

	# filter candidates: used but not defined
	candidates = {}
	for name, info in used.items():
		if name in defined:
			continue
		# ignore obvious non-parameter macros
		if name in {"PRIVATE", "LOCAL", "GLOBAL", "TYPE_T", "TYPE_TS", "CEIL"}:
			continue
		candidates[name] = info

	return {"file": file_path, "parameters": candidates}


def parse_file(file_path: str) -> Dict:
	"""Alias for per-file parsing (kept for clarity)."""
	return parse_tuning_parameters(file_path)


def parse_directory(dir_path: str, pattern: str = "*.cl") -> Dict[str, Dict]:
	"""Parse all .cl files in directory (non-recursive) and return mapping file->result."""
	results = {}
	for entry in os.listdir(dir_path):
		if not entry.endswith(".cl"):
			continue
		path = os.path.join(dir_path, entry)
		if os.path.isfile(path):
			results[entry] = parse_tuning_parameters(path)
	return results


def _cli_main():
	parser = argparse.ArgumentParser(description="Extract tuning parameters from OpenCL kernel templates or a single file")
	parser.add_argument("path", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "kernel-template"), help="Path to a kernel file or directory (defaults to kernel-template)")
	parser.add_argument("--json", "-j", help="Write JSON output to this file")
	parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout")
	args = parser.parse_args()
	target = os.path.abspath(args.path)
	if os.path.isfile(target):
		results = parse_file(target)
	elif os.path.isdir(target):
		results = parse_directory(target)
	else:
		parser.error(f"No such file or directory: {target}")

	# write JSON file if requested
	if args.json:
		with open(args.json, "w", encoding="utf-8") as f:
			json.dump(results, f, indent=2)

	# always print pretty JSON unless user explicitly requested only file write
	if args.pretty or not args.json:
		print(json.dumps(results, indent=2))


if __name__ == "__main__":
	_cli_main()

