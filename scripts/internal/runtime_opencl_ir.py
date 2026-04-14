#!/usr/bin/env python3
"""Extract runtime LLVM IR from an OpenCL program binary."""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


class RuntimeOpenCLIRError(RuntimeError):
	pass


def _extract_elf64_section(blob: bytes, section_name: str) -> bytes:
	if len(blob) < 64 or blob[:4] != b"\x7fELF":
		raise RuntimeOpenCLIRError("OpenCL program binary is not an ELF container")
	if blob[4] != 2:
		raise RuntimeOpenCLIRError("Only ELF64 OpenCL program binaries are supported")

	endian = "<" if blob[5] == 1 else ">"
	section_header_offset = struct.unpack_from(f"{endian}Q", blob, 0x28)[0]
	section_header_entry_size = struct.unpack_from(f"{endian}H", blob, 0x3A)[0]
	section_header_count = struct.unpack_from(f"{endian}H", blob, 0x3C)[0]
	section_name_table_index = struct.unpack_from(f"{endian}H", blob, 0x3E)[0]
	if section_header_offset == 0 or section_header_count == 0:
		raise RuntimeOpenCLIRError("OpenCL program binary has no ELF section headers")
	if section_name_table_index >= section_header_count:
		raise RuntimeOpenCLIRError("OpenCL program binary has an invalid section-name table index")

	section_headers = []
	for idx in range(section_header_count):
		offset = section_header_offset + idx * section_header_entry_size
		if offset + section_header_entry_size > len(blob):
			raise RuntimeOpenCLIRError("OpenCL program binary has truncated ELF section headers")
		section_headers.append(struct.unpack_from(f"{endian}IIQQQQIIQQ", blob, offset))

	_, _, _, _, strtab_offset, strtab_size, _, _, _, _ = section_headers[section_name_table_index]
	strtab = blob[strtab_offset:strtab_offset + strtab_size]

	for name_offset, _, _, _, data_offset, data_size, _, _, _, _ in section_headers:
		name_end = strtab.find(b"\0", name_offset)
		if name_end < 0:
			continue
		name = strtab[name_offset:name_end].decode("utf-8", errors="replace")
		if name == section_name:
			return blob[data_offset:data_offset + data_size]

	raise RuntimeOpenCLIRError(
		f"OpenCL program binary does not contain required section {section_name!r}"
	)


def extract_opencl_runtime_ir(
	opencl_binary_path: Path,
	output_ll_path: Path,
	section_name: str = ".ocl.ir",
) -> Path:
	"""Extract runtime LLVM bitcode from an OpenCL binary and disassemble it to .ll."""
	if shutil.which("llvm-dis") is None:
		raise RuntimeOpenCLIRError("Required tool not found in PATH: llvm-dis")

	bitcode = _extract_elf64_section(opencl_binary_path.read_bytes(), section_name)
	output_ll_path.parent.mkdir(parents=True, exist_ok=True)

	with tempfile.NamedTemporaryFile(suffix=".bc", delete=False) as temp_bc:
		temp_bc.write(bitcode)
		temp_bc_path = Path(temp_bc.name)
	try:
		result = subprocess.run(
			["llvm-dis", str(temp_bc_path), "-o", str(output_ll_path)],
			check=False,
			capture_output=True,
			text=True,
		)
		if result.returncode != 0:
			raise RuntimeOpenCLIRError(f"llvm-dis failed:\n{result.stderr.strip()}")
	finally:
		temp_bc_path.unlink(missing_ok=True)

	return output_ll_path


def main() -> int:
	parser = argparse.ArgumentParser(description="Extract .ocl.ir from an OpenCL binary")
	parser.add_argument("opencl_binary", type=Path, help="Input OpenCL program binary")
	parser.add_argument("output_ll", type=Path, help="Output LLVM IR text file")
	parser.add_argument("--section", default=".ocl.ir", help="ELF section containing LLVM bitcode")
	args = parser.parse_args()

	try:
		extract_opencl_runtime_ir(args.opencl_binary, args.output_ll, section_name=args.section)
	except RuntimeOpenCLIRError as exc:
		print(f"Error: {exc}")
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
