from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

from scripts.collect_tuning_performance_simple import _device_limits, _load_config_generator_modules

from ..core.instruction_map import canonical_config_key


class LiveConfigGenerator:
	def __init__(
		self,
		kernel_type: str,
		input_size: str,
		device_type: str = "cpu",
		random_seed: int = 1,
	):
		self.kernel_type = kernel_type
		self.input_size = input_size
		self.device_type = device_type
		self.random_seed = random_seed
		self.modules = _load_config_generator_modules()
		if kernel_type not in self.modules:
			raise ValueError(f"Unsupported kernel type: {kernel_type}")
		self._generated_keys: set[str] = set()
		self._counter = 0
		self._dims = tuple(int(value) for value in input_size.split("x"))

	def generate(self, count: int) -> list[dict[str, Any]]:
		configs: list[dict[str, Any]] = []
		attempt_round = 0
		while len(configs) < count:
			needed = count - len(configs)
			batch_size = max(needed * 2, 8)
			for config in self._sample_upper_configs(batch_size, attempt_round):
				host_config = self._to_host_config(config)
				if not self.validate_config(host_config):
					continue
				key = canonical_config_key(host_config)
				if key in self._generated_keys:
					continue
				self._generated_keys.add(key)
				configs.append(host_config)
				if len(configs) >= count:
					break
			attempt_round += 1
			if attempt_round > 100 and not configs:
				raise RuntimeError("failed to generate any valid configs")
		return configs

	def _sample_upper_configs(self, count: int, attempt_round: int) -> Iterator[dict[str, Any]]:
		max_wi_size, max_wg_size = _device_limits(self.device_type)
		seed_material = f"{self.kernel_type}:{self.input_size}:{self.random_seed}:{self._counter}:{attempt_round}"
		seed = int(hashlib.md5(seed_material.encode()).hexdigest()[:8], 16)
		self._counter += 1
		module = self.modules[self.kernel_type]
		if self.kernel_type == "gaussian":
			h, w = self._dims
			return module.random_sample_configurations(
				h,
				w,
				count,
				max_wi_size=max_wi_size,
				max_wg_size=max_wg_size,
				seed=seed,
			)
		m, n, k = self._dims
		return module.random_sample_configurations(
			m,
			n,
			k,
			count,
			max_wi_size=max_wi_size,
			max_wg_size=max_wg_size,
			seed=seed,
		)

	def _to_host_config(self, config: dict[str, Any]) -> dict[str, Any]:
		host_config = {key.lower(): int(value) for key, value in config.items()}
		if self.kernel_type == "gaussian":
			h, w = self._dims
			host_config["input_size_h"] = h
			host_config["input_size_w"] = w
		else:
			m, n, k = self._dims
			host_config["M"] = m
			host_config["N"] = n
			host_config["K"] = k
		return host_config

	def fixed_input_config(self) -> dict[str, int]:
		if self.kernel_type == "gaussian":
			h, w = self._dims
			return {
				"input_size_h": h,
				"input_size_w": w,
			}
		m, n, k = self._dims
		return {
			"M": m,
			"N": n,
			"K": k,
		}

	def validate_config(self, host_config: dict[str, Any]) -> bool:
		max_wi_size, max_wg_size = _device_limits(self.device_type)
		module = self.modules[self.kernel_type]
		upper_config = self._to_upper_config(host_config)
		return bool(module.is_configuration_valid(upper_config, max_wi_size, max_wg_size))

	def _to_upper_config(self, host_config: dict[str, Any]) -> dict[str, int]:
		return {
			key.upper(): int(value)
			for key, value in host_config.items()
			if key not in {"input_size_h", "input_size_w", "M", "N", "K"}
		}


def hash_config(config: dict[str, Any]) -> str:
	config_str = json.dumps(config, sort_keys=True)
	return hashlib.md5(config_str.encode()).hexdigest()[:16]
