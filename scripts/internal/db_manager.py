"""Database manager for tuning parameter experiments and performance results.

Provides persistent storage and querying of experiment results using SQLite.
Tracks kernel compilations, performance metrics, and LLVM IR generation.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ExperimentResult:
	"""Represents a single experiment result."""
	exp_id: int
	kernel_type: str
	input_size: str
	param_hash: str
	config_json: str
	runtime_ms: Optional[float]
	llvm_ir_path: Optional[str]
	timestamp: str
	status: str  # 'pending', 'completed', 'failed'


class ExperimentDB:
	"""SQLite database manager for experiment tracking and results."""

	def __init__(self, db_path: str = "experiments.db"):
		self.db_path = db_path
		self.conn: Optional[sqlite3.Connection] = None
		self._exp_columns = (
			"exp_id",
			"kernel_type",
			"input_size",
			"param_hash",
			"config_json",
			"runtime_ms",
			"llvm_ir_path",
			"timestamp",
			"status",
		)
		self._connect()
		self._init_schema()

	def _connect(self) -> None:
		try:
			self.conn = sqlite3.connect(self.db_path)
			self.conn.row_factory = sqlite3.Row
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to connect to database {self.db_path}: {e}")

	def _execute(self, query: str, params: Optional[Sequence[Any]] = None):
		if self.conn is None:
			raise RuntimeError("Database is not connected")
		cursor = self.conn.cursor()
		if params is None:
			cursor.execute(query)
		else:
			cursor.execute(query, params)
		self.conn.commit()
		return cursor

	def _init_schema(self) -> None:
		try:
			self._execute(
				"""
				CREATE TABLE IF NOT EXISTS experiments (
					exp_id INTEGER PRIMARY KEY AUTOINCREMENT,
					kernel_type TEXT NOT NULL,
					input_size TEXT NOT NULL,
					param_hash TEXT NOT NULL,
					config_json TEXT NOT NULL,
					runtime_ms REAL,
					llvm_ir_path TEXT,
					timestamp TEXT NOT NULL,
					status TEXT NOT NULL DEFAULT 'pending',
					error_msg TEXT,
					UNIQUE(kernel_type, input_size, param_hash)
				)
				"""
			)
			self._execute("CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)")
			self._execute("CREATE INDEX IF NOT EXISTS idx_kernel_type ON experiments(kernel_type)")
			self._execute("CREATE INDEX IF NOT EXISTS idx_input_size ON experiments(input_size)")
			self._execute(
				"""
				CREATE TABLE IF NOT EXISTS run_metadata (
					run_id INTEGER PRIMARY KEY AUTOINCREMENT,
					start_time TEXT NOT NULL,
					end_time TEXT,
					total_configs INTEGER,
					completed_count INTEGER DEFAULT 0,
					failed_count INTEGER DEFAULT 0,
					notes TEXT
				)
				"""
			)
			self._ensure_llvm_instruction_counts_schema()
			self._ensure_experiment_logs_schema()
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to initialize database schema: {e}")

	def _create_llvm_instruction_counts_table(self) -> None:
		self._execute(
			"""
			CREATE TABLE IF NOT EXISTS llvm_instruction_counts (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				exp_id INTEGER NOT NULL,
				template_name TEXT NOT NULL,
				kernel_function TEXT NOT NULL,
				llvm_ir_path TEXT NOT NULL,
				bb_counts_json TEXT NOT NULL,
				bb_instruction_counts_json TEXT NOT NULL,
				total_instruction_counts_json TEXT NOT NULL,
				created_at TEXT NOT NULL,
				UNIQUE(exp_id, template_name),
				FOREIGN KEY(exp_id) REFERENCES experiments(exp_id)
			)
			"""
		)
		self._execute(
			"CREATE INDEX IF NOT EXISTS idx_llvm_instruction_exp_id ON llvm_instruction_counts(exp_id)"
		)
		self._execute(
			"CREATE INDEX IF NOT EXISTS idx_llvm_instruction_template_name ON llvm_instruction_counts(template_name)"
		)

	def _ensure_llvm_instruction_counts_schema(self) -> None:
		cursor = self._execute(
			"""
			SELECT sql
			FROM sqlite_master
			WHERE type = 'table' AND name = 'llvm_instruction_counts'
			"""
		)
		row = cursor.fetchone()
		if row is None:
			self._create_llvm_instruction_counts_table()
			return

		table_sql = (row[0] or "").replace(" ", "").lower()
		column_rows = self._execute("PRAGMA table_info(llvm_instruction_counts)").fetchall()
		column_names = {str(col[1]) for col in column_rows}
		required_columns = {
			"exp_id",
			"template_name",
			"kernel_function",
			"llvm_ir_path",
			"bb_counts_json",
			"bb_instruction_counts_json",
			"total_instruction_counts_json",
			"created_at",
		}
		needs_migration = not required_columns.issubset(column_names) or "unique(exp_id)" in table_sql
		if not needs_migration:
			self._execute(
				"CREATE INDEX IF NOT EXISTS idx_llvm_instruction_exp_id ON llvm_instruction_counts(exp_id)"
			)
			self._execute(
				"CREATE INDEX IF NOT EXISTS idx_llvm_instruction_template_name ON llvm_instruction_counts(template_name)"
			)
			return

		self._execute("ALTER TABLE llvm_instruction_counts RENAME TO llvm_instruction_counts_legacy")
		self._create_llvm_instruction_counts_table()
		self._execute(
			"""
			INSERT INTO llvm_instruction_counts (
				exp_id,
				template_name,
				kernel_function,
				llvm_ir_path,
				bb_counts_json,
				bb_instruction_counts_json,
				total_instruction_counts_json,
				created_at
			)
			SELECT
				exp_id,
				CASE
					WHEN lower(llvm_ir_path) LIKE '%gaussian%' THEN 'gaussian_static_1'
					WHEN lower(llvm_ir_path) LIKE '%gemm_2%' THEN 'gemm_2'
					WHEN lower(llvm_ir_path) LIKE '%gemm%' THEN 'gemm_1'
					ELSE 'unknown'
				END,
				CASE
					WHEN lower(llvm_ir_path) LIKE '%gaussian%' THEN 'gaussian_1'
					WHEN lower(llvm_ir_path) LIKE '%gemm_2%' THEN 'gemm_2'
					WHEN lower(llvm_ir_path) LIKE '%gemm%' THEN 'gemm_1'
					ELSE 'unknown'
				END,
				llvm_ir_path,
				bb_counts_json,
				bb_instruction_counts_json,
				total_instruction_counts_json,
				created_at
			FROM llvm_instruction_counts_legacy
			"""
		)
		self._execute("DROP TABLE llvm_instruction_counts_legacy")

	def _ensure_experiment_logs_schema(self) -> None:
		self._execute(
			"""
			CREATE TABLE IF NOT EXISTS experiment_logs (
				log_id INTEGER PRIMARY KEY AUTOINCREMENT,
				exp_id INTEGER NOT NULL,
				stage TEXT NOT NULL,
				level TEXT NOT NULL,
				message TEXT NOT NULL,
				payload_json TEXT,
				created_at TEXT NOT NULL,
				FOREIGN KEY(exp_id) REFERENCES experiments(exp_id)
			)
			"""
		)
		self._execute(
			"CREATE INDEX IF NOT EXISTS idx_experiment_logs_exp_id ON experiment_logs(exp_id)"
		)
		self._execute(
			"CREATE INDEX IF NOT EXISTS idx_experiment_logs_stage ON experiment_logs(stage)"
		)

	def add_experiment(
		self,
		kernel_type: str,
		input_size: str,
		param_hash: str,
		config_dict: Dict,
		skip_if_exists: bool = True,
	) -> Optional[int]:
		try:
			config_json = json.dumps(config_dict)
			timestamp = datetime.now().isoformat()

			if skip_if_exists:
				cursor = self._execute(
					"""
					INSERT OR IGNORE INTO experiments
					(kernel_type, input_size, param_hash, config_json, timestamp, status)
					VALUES (?, ?, ?, ?, ?, 'pending')
					""",
					(kernel_type, input_size, param_hash, config_json, timestamp),
				)
				return cursor.lastrowid if cursor.rowcount > 0 else None

			cursor = self._execute(
				"""
				INSERT INTO experiments
				(kernel_type, input_size, param_hash, config_json, timestamp, status)
				VALUES (?, ?, ?, ?, ?, 'pending')
				""",
				(kernel_type, input_size, param_hash, config_json, timestamp),
			)
			return cursor.lastrowid
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to add experiment: {e}")

	def mark_completed(self, exp_id: int, runtime_ms: float, llvm_ir_path: Optional[str] = None) -> bool:
		try:
			cursor = self._execute(
				"""
				UPDATE experiments
				SET status = 'completed', runtime_ms = ?, llvm_ir_path = ?, error_msg = NULL
				WHERE exp_id = ?
				""",
				(runtime_ms, llvm_ir_path, exp_id),
			)
			return cursor.rowcount > 0
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to mark experiment as completed: {e}")

	def mark_profiled(self, exp_id: int, llvm_ir_path: Optional[str] = None) -> bool:
		"""Mark an experiment as statically profiled without a runtime measurement."""
		try:
			cursor = self._execute(
				"""
				UPDATE experiments
				SET status = 'profiled', llvm_ir_path = ?, error_msg = NULL
				WHERE exp_id = ?
				""",
				(llvm_ir_path, exp_id),
			)
			return cursor.rowcount > 0
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to mark experiment as profiled: {e}")

	def mark_failed(self, exp_id: int, error_msg: str = "") -> bool:
		try:
			cursor = self._execute(
				"""
				UPDATE experiments
				SET status = 'failed', error_msg = ?
				WHERE exp_id = ?
				""",
				(error_msg, exp_id),
			)
			return cursor.rowcount > 0
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to mark experiment as failed: {e}")

	def record_error_message(self, exp_id: int, error_msg: str) -> bool:
		"""Attach a non-fatal diagnostic message to an experiment row."""
		try:
			cursor = self._execute(
				"""
				UPDATE experiments
				SET error_msg = ?
				WHERE exp_id = ?
				""",
				(error_msg, exp_id),
			)
			return cursor.rowcount > 0
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to record experiment diagnostic: {e}")

	def add_experiment_log(
		self,
		exp_id: int,
		stage: str,
		level: str,
		message: str,
		payload: Optional[Dict[str, Any]] = None,
	) -> Optional[int]:
		try:
			cursor = self._execute(
				"""
				INSERT INTO experiment_logs
				(exp_id, stage, level, message, payload_json, created_at)
				VALUES (?, ?, ?, ?, ?, ?)
				""",
				(
					exp_id,
					stage,
					level,
					message,
					json.dumps(payload, sort_keys=True) if payload is not None else None,
					datetime.now().isoformat(),
				),
			)
			return int(cursor.lastrowid) if cursor.lastrowid is not None else None
		except (sqlite3.Error, TypeError, ValueError) as e:
			raise RuntimeError(f"Failed to add experiment log: {e}")

	def get_experiment_logs(self, exp_id: int) -> List[Dict[str, Any]]:
		try:
			cursor = self._execute(
				"""
				SELECT log_id, exp_id, stage, level, message, payload_json, created_at
				FROM experiment_logs
				WHERE exp_id = ?
				ORDER BY log_id ASC
				""",
				(exp_id,),
			)
			rows = cursor.fetchall()
			logs: List[Dict[str, Any]] = []
			for row in rows:
				payload = None
				if row[5]:
					payload = json.loads(row[5])
				logs.append(
					{
						"log_id": row[0],
						"exp_id": row[1],
						"stage": row[2],
						"level": row[3],
						"message": row[4],
						"payload": payload,
						"created_at": row[6],
					}
				)
			return logs
		except (sqlite3.Error, json.JSONDecodeError) as e:
			raise RuntimeError(f"Failed to query experiment logs: {e}")

	def get_pending_experiments(self, limit: Optional[int] = None) -> List[ExperimentResult]:
		try:
			if limit is None:
				cursor = self._execute(
					"""
					SELECT exp_id, kernel_type, input_size, param_hash, config_json,
					       runtime_ms, llvm_ir_path, timestamp, status
					FROM experiments
					WHERE status = ?
					""",
					("pending",),
				)
			else:
				cursor = self._execute(
					"""
					SELECT exp_id, kernel_type, input_size, param_hash, config_json,
					       runtime_ms, llvm_ir_path, timestamp, status
					FROM experiments
					WHERE status = ?
					LIMIT ?
					""",
					("pending", limit),
				)
			rows = cursor.fetchall()
			return [self._row_to_experiment(row) for row in rows]
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to query pending experiments: {e}")

	def get_experiment_by_id(self, exp_id: int) -> Optional[ExperimentResult]:
		try:
			cursor = self._execute(
				"""
				SELECT exp_id, kernel_type, input_size, param_hash, config_json,
				       runtime_ms, llvm_ir_path, timestamp, status
				FROM experiments
				WHERE exp_id = ?
				""",
				(exp_id,),
			)
			row = cursor.fetchone()
			return self._row_to_experiment(row) if row else None
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to query experiment: {e}")

	def get_experiment_by_key(
		self,
		kernel_type: str,
		input_size: str,
		param_hash: str,
	) -> Optional[ExperimentResult]:
		try:
			cursor = self._execute(
				"""
				SELECT exp_id, kernel_type, input_size, param_hash, config_json,
				       runtime_ms, llvm_ir_path, timestamp, status
				FROM experiments
				WHERE kernel_type = ? AND input_size = ? AND param_hash = ?
				LIMIT 1
				""",
				(kernel_type, input_size, param_hash),
			)
			row = cursor.fetchone()
			return self._row_to_experiment(row) if row else None
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to query experiment by key: {e}")

	def get_experiments_by_kernel_type(self, kernel_type: str, status: Optional[str] = None) -> List[ExperimentResult]:
		try:
			if status is None:
				cursor = self._execute(
					"""
					SELECT exp_id, kernel_type, input_size, param_hash, config_json,
					       runtime_ms, llvm_ir_path, timestamp, status
					FROM experiments
					WHERE kernel_type = ?
					""",
					(kernel_type,),
				)
			else:
				cursor = self._execute(
					"""
					SELECT exp_id, kernel_type, input_size, param_hash, config_json,
					       runtime_ms, llvm_ir_path, timestamp, status
					FROM experiments
					WHERE kernel_type = ? AND status = ?
					""",
					(kernel_type, status),
				)
			rows = cursor.fetchall()
			return [self._row_to_experiment(row) for row in rows]
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to query experiments: {e}")

	def get_statistics(self) -> Dict:
		try:
			total = self._execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
			completed = self._execute(
				"SELECT COUNT(*) FROM experiments WHERE status = ?",
				("completed",),
			).fetchone()[0]
			failed = self._execute(
				"SELECT COUNT(*) FROM experiments WHERE status = ?",
				("failed",),
			).fetchone()[0]
			pending = self._execute(
				"SELECT COUNT(*) FROM experiments WHERE status = ?",
				("pending",),
			).fetchone()[0]
			avg_runtime = self._execute(
				"SELECT AVG(runtime_ms) FROM experiments WHERE status = ?",
				("completed",),
			).fetchone()[0]
			return {
				"total": total,
				"completed": completed,
				"failed": failed,
				"pending": pending,
				"avg_runtime_ms": avg_runtime,
			}
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to get statistics: {e}")

	def get_config_by_hash(self, param_hash: str) -> Optional[Dict]:
		try:
			cursor = self._execute(
				"SELECT config_json FROM experiments WHERE param_hash = ? LIMIT 1",
				(param_hash,),
			)
			row = cursor.fetchone()
			return json.loads(row[0]) if row else None
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to get config: {e}")

	def upsert_llvm_instruction_counts(
		self,
		exp_id: int,
		template_name: str,
		kernel_function: str,
		llvm_ir_path: str,
		bb_counts: Dict[str, int],
		bb_instruction_counts: Dict[str, Dict[str, int]],
		total_instruction_counts: Dict[str, int],
	) -> None:
		"""Store static LLVM instruction-count data for an experiment."""
		try:
			self._execute(
				"""
				INSERT INTO llvm_instruction_counts
				(exp_id, template_name, kernel_function, llvm_ir_path, bb_counts_json, bb_instruction_counts_json, total_instruction_counts_json, created_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(exp_id, template_name) DO UPDATE SET
					kernel_function = excluded.kernel_function,
					llvm_ir_path = excluded.llvm_ir_path,
					bb_counts_json = excluded.bb_counts_json,
					bb_instruction_counts_json = excluded.bb_instruction_counts_json,
					total_instruction_counts_json = excluded.total_instruction_counts_json,
					created_at = excluded.created_at
				""",
				(
					exp_id,
					template_name,
					kernel_function,
					llvm_ir_path,
					json.dumps(bb_counts),
					json.dumps(bb_instruction_counts),
					json.dumps(total_instruction_counts),
					datetime.now().isoformat(),
				),
			)
		except sqlite3.Error as e:
			raise RuntimeError(f"Failed to upsert llvm instruction counts: {e}")

	def get_llvm_instruction_counts(self, exp_id: Optional[int] = None) -> List[Dict[str, Any]]:
		"""Query LLVM instruction-count rows, optionally filtered by experiment ID."""
		try:
			if exp_id is None:
				cursor = self._execute(
					"""
					SELECT lic.exp_id, lic.template_name, lic.kernel_function, lic.llvm_ir_path, lic.bb_counts_json,
					       lic.bb_instruction_counts_json, lic.total_instruction_counts_json,
					       e.kernel_type, e.input_size
					FROM llvm_instruction_counts lic
					JOIN experiments e ON e.exp_id = lic.exp_id
					ORDER BY lic.exp_id DESC, lic.template_name ASC
					"""
				)
			else:
				cursor = self._execute(
					"""
					SELECT lic.exp_id, lic.template_name, lic.kernel_function, lic.llvm_ir_path, lic.bb_counts_json,
					       lic.bb_instruction_counts_json, lic.total_instruction_counts_json,
					       e.kernel_type, e.input_size
					FROM llvm_instruction_counts lic
					JOIN experiments e ON e.exp_id = lic.exp_id
					WHERE lic.exp_id = ?
					ORDER BY lic.exp_id DESC, lic.template_name ASC
					""",
					(exp_id,),
				)
			rows = cursor.fetchall()
			out: List[Dict[str, Any]] = []
			for row in rows:
				out.append(
					{
						"exp_id": row[0],
						"template_name": row[1],
						"kernel_function": row[2],
						"llvm_ir_path": row[3],
						"bb_counts": json.loads(row[4]),
						"bb_instruction_counts": json.loads(row[5]),
						"total_instruction_counts": json.loads(row[6]),
						"kernel_type": row[7],
						"input_size": row[8],
					}
				)
			return out
		except (sqlite3.Error, json.JSONDecodeError) as e:
			raise RuntimeError(f"Failed to query llvm instruction counts: {e}")

	def _row_to_experiment(self, row: Sequence[Any]) -> ExperimentResult:
		row_dict = dict(zip(self._exp_columns, row))
		return ExperimentResult(
			exp_id=row_dict["exp_id"],
			kernel_type=row_dict["kernel_type"],
			input_size=row_dict["input_size"],
			param_hash=row_dict["param_hash"],
			config_json=row_dict["config_json"],
			runtime_ms=row_dict["runtime_ms"],
			llvm_ir_path=row_dict["llvm_ir_path"],
			timestamp=row_dict["timestamp"],
			status=row_dict["status"],
		)

	def close(self) -> None:
		if self.conn:
			try:
				self.conn.close()
			except sqlite3.Error:
				pass
			self.conn = None

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()
