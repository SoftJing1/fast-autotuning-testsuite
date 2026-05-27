from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.instruction_map import canonical_config_key
from ..core.instruction_space import filter_instruction_opcodes
from ..core.parameter_space import (
	build_generator_parameter_index_specs,
	config_from_parameter_indices,
	indices_from_config,
)
from ..core.resolver import OnlineInstructionMapResolver
from ..core.types import LiveProfile
from .config_space import LiveConfigGenerator
from .executor import LiveKernelExecutor


def instruction_map_distance(
	requested_counts: dict[str, Any],
	candidate_counts: dict[str, Any],
) -> float:
	opcodes = filter_instruction_opcodes(set(requested_counts) | set(candidate_counts))
	return sum(
		(float(requested_counts.get(op, 0)) - float(candidate_counts.get(op, 0))) ** 2
		for op in opcodes
	) ** 0.5


@dataclass(frozen=True)
class MutationSearchResult:
	profile: LiveProfile
	distance: float
	source: str
	mutated_profiles: tuple[LiveProfile, ...] = ()


class LocalInstructionMapMutator:
	def __init__(
		self,
		config_generator: LiveConfigGenerator,
		executor: LiveKernelExecutor,
		mutation_budget: int,
	):
		self.config_generator = config_generator
		self.executor = executor
		self.mutation_budget = max(int(mutation_budget), 0)
		self.parameter_specs = build_generator_parameter_index_specs(
			config_generator.kernel_type,
			config_generator.input_size,
			device_type=config_generator.device_type,
		)
		self.fixed_config = config_generator.fixed_input_config()
		self._dynamic_pool = OnlineInstructionMapResolver()
		self._profile_cache: dict[tuple[tuple[str, Any], ...], LiveProfile | None] = {}

	def refine(
		self,
		requested_counts: dict[str, int],
		reference_profile: LiveProfile,
	) -> MutationSearchResult:
		reference_distance = instruction_map_distance(requested_counts, reference_profile.raw_counts)
		if self.mutation_budget <= 0:
			return MutationSearchResult(
				profile=reference_profile,
				distance=reference_distance,
				source="database",
			)

		best_profile = reference_profile
		best_distance = reference_distance
		best_indices = indices_from_config(reference_profile.config, self.parameter_specs)
		seen_configs = {tuple(sorted(reference_profile.config.items()))}
		mutated_profiles: list[LiveProfile] = []
		remaining_budget = self.mutation_budget

		if self._dynamic_pool.profiles:
			pool_match = self._dynamic_pool.resolve(requested_counts)
			if pool_match.valid and pool_match.profile is not None:
				pool_distance = instruction_map_distance(requested_counts, pool_match.profile.raw_counts)
				if pool_distance < best_distance:
					best_profile = pool_match.profile
					best_distance = pool_distance
					best_indices = indices_from_config(best_profile.config, self.parameter_specs)

		while remaining_budget > 0:
			neighbors = self._neighbor_configs(best_indices, seen_configs)
			if not neighbors:
				break
			improved = False
			for config, indices in neighbors:
				profile, cache_hit = self._profile_or_cache_hit(config)
				if not cache_hit:
					remaining_budget -= 1
				if profile is None:
					if remaining_budget <= 0:
						break
					continue
				mutated_profiles.append(profile)
				self._dynamic_pool.add_profiles([profile])
				distance = instruction_map_distance(requested_counts, profile.raw_counts)
				if distance < best_distance:
					best_profile = profile
					best_distance = distance
					best_indices = indices
					improved = True
				if remaining_budget <= 0:
					break
			if not improved:
				break

		source = "mutation" if best_profile.param_hash != reference_profile.param_hash else "database"
		return MutationSearchResult(
			profile=best_profile,
			distance=best_distance,
			source=source,
			mutated_profiles=tuple(mutated_profiles),
		)

	def _neighbor_configs(
		self,
		index_config: dict[str, int],
		seen_configs: set[tuple[tuple[str, Any], ...]],
	) -> list[tuple[dict[str, Any], dict[str, int]]]:
		neighbors: list[tuple[dict[str, Any], dict[str, int]]] = []
		for name, spec in sorted(self.parameter_specs.items()):
			key = f"idx.{name}"
			if len(spec.values) <= 1:
				continue
			current = int(index_config[key])
			for delta in (-1, 1):
				next_index = current + delta
				if next_index < 0 or next_index >= len(spec.values):
					continue
				candidate_indices = dict(index_config)
				candidate_indices[key] = next_index
				config = config_from_parameter_indices(
					candidate_indices,
					self.parameter_specs,
					fixed_config=self.fixed_config,
				)
				if not self.config_generator.validate_config(config):
					continue
				config_key = tuple(sorted(config.items()))
				if config_key in seen_configs:
					continue
				seen_configs.add(config_key)
				neighbors.append((config, candidate_indices))
		return neighbors

	def _profile_or_cache_hit(self, config: dict[str, Any]) -> tuple[LiveProfile | None, bool]:
		config_key = canonical_config_key(config)
		if config_key in self._profile_cache:
			return self._profile_cache[config_key], True
		result = self.executor.profile_config(config)
		profile = result.profile if result.valid and result.profile is not None else None
		self._profile_cache[config_key] = profile
		return profile, False
