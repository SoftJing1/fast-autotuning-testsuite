#!/usr/bin/env python3
"""Web-based interactive visualization for experiment SQLite databases.

Run this script against an experiment database (experiments.db) to inspect
summary metrics, filterable results, and runtime trends.
"""

from __future__ import annotations

import argparse
import json
import socket
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

try:
	from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html
except ImportError as exc:
	raise SystemExit(
		"Dash is required for this visualization UI. Install dependencies with "
		"`pip install -r requirements.txt`."
	) from exc


def _query_rows(db_path: Path, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	try:
		rows = conn.execute(sql, params).fetchall()
		return [dict(row) for row in rows]
	finally:
		conn.close()


def _query_one(db_path: Path, sql: str, params: tuple = ()) -> Dict[str, Any]:
	rows = _query_rows(db_path, sql, params)
	return rows[0] if rows else {}


def _build_summary(db_path: Path) -> Dict[str, Any]:
	total = _query_one(db_path, "SELECT COUNT(*) AS v FROM experiments").get("v", 0)
	completed = _query_one(
		db_path,
		"SELECT COUNT(*) AS v FROM experiments WHERE status = ?",
		("completed",),
	).get("v", 0)
	failed = _query_one(
		db_path,
		"SELECT COUNT(*) AS v FROM experiments WHERE status = ?",
		("failed",),
	).get("v", 0)
	pending = _query_one(
		db_path,
		"SELECT COUNT(*) AS v FROM experiments WHERE status = ?",
		("pending",),
	).get("v", 0)
	avg_runtime = _query_one(
		db_path,
		"SELECT AVG(runtime_ms) AS v FROM experiments WHERE status = ?",
		("completed",),
	).get("v")

	by_kernel = _query_rows(
		db_path,
		"""
		SELECT kernel_type, COUNT(*) AS total,
		       AVG(CASE WHEN status = 'completed' THEN runtime_ms END) AS avg_runtime_ms
		FROM experiments
		GROUP BY kernel_type
		ORDER BY kernel_type
		""",
	)

	return {
		"total": total,
		"completed": completed,
		"failed": failed,
		"pending": pending,
		"avg_runtime_ms": avg_runtime,
		"by_kernel": by_kernel,
	}


def _build_options(db_path: Path) -> Dict[str, Any]:
	kernels = [
		row["kernel_type"]
		for row in _query_rows(db_path, "SELECT DISTINCT kernel_type FROM experiments ORDER BY kernel_type")
	]
	sizes = [
		row["input_size"]
		for row in _query_rows(db_path, "SELECT DISTINCT input_size FROM experiments ORDER BY input_size")
	]
	statuses = [row["status"] for row in _query_rows(db_path, "SELECT DISTINCT status FROM experiments ORDER BY status")]
	return {"kernel_types": kernels, "input_sizes": sizes, "statuses": statuses}


def _build_results_payload(
	db_path: Path,
	kernel_type: str | None,
	input_size: str | None,
	status: str | None,
	limit_value: Any,
) -> Dict[str, Any]:
	try:
		limit = max(1, min(2000, int(limit_value or 300)))
	except (TypeError, ValueError):
		limit = 300

	where_clauses = []
	params: List[Any] = []
	if kernel_type:
		where_clauses.append("kernel_type = ?")
		params.append(kernel_type)
	if input_size:
		where_clauses.append("input_size = ?")
		params.append(input_size)
	if status:
		where_clauses.append("status = ?")
		params.append(status)

	where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
	sql = f"""
		SELECT exp_id, kernel_type, input_size, param_hash, runtime_ms, status, timestamp, llvm_ir_path
		FROM experiments
		{where_sql}
		ORDER BY exp_id DESC
		LIMIT ?
	"""
	params.append(limit)
	rows = _query_rows(db_path, sql, tuple(params))
	for row in rows:
		llvm_ir_path = row.get("llvm_ir_path")
		if not llvm_ir_path:
			continue
		try:
			decoded = json.loads(llvm_ir_path)
		except (TypeError, json.JSONDecodeError):
			continue
		if isinstance(decoded, list):
			row["llvm_ir_path"] = ", ".join(str(item) for item in decoded)
	return {"rows": rows, "count": len(rows), "limit": limit}


def _build_inst_counts_payload(
	db_path: Path,
	kernel_type: str | None,
	input_size: str | None,
	limit_value: Any = 100,
) -> Dict[str, Any]:
	try:
		limit = max(1, min(1000, int(limit_value or 100)))
	except (TypeError, ValueError):
		limit = 100

	where_clauses = []
	params: List[Any] = []
	if kernel_type:
		where_clauses.append("e.kernel_type = ?")
		params.append(kernel_type)
	if input_size:
		where_clauses.append("e.input_size = ?")
		params.append(input_size)

	where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
	sql = f"""
		SELECT lic.exp_id, e.kernel_type, e.input_size, lic.template_name, lic.kernel_function, lic.llvm_ir_path,
		       lic.bb_counts_json, lic.total_instruction_counts_json
		FROM llvm_instruction_counts lic
		JOIN experiments e ON e.exp_id = lic.exp_id
		{where_sql}
		ORDER BY lic.exp_id DESC, lic.template_name ASC
		LIMIT ?
	"""
	params.append(limit)
	rows = _query_rows(db_path, sql, tuple(params))

	processed: List[Dict[str, Any]] = []
	for row in rows:
		bb_counts = json.loads(row["bb_counts_json"])
		total_counts = json.loads(row["total_instruction_counts_json"])
		total_dynamic = int(sum(int(v) for v in total_counts.values()))
		top_insts = sorted(total_counts.items(), key=lambda item: int(item[1]), reverse=True)[:5]
		processed.append(
			{
				"exp_id": row["exp_id"],
				"kernel_type": row["kernel_type"],
				"input_size": row["input_size"],
				"template_name": row["template_name"],
				"kernel_function": row["kernel_function"],
				"llvm_ir_path": row["llvm_ir_path"],
				"basic_block_count": len(bb_counts),
				"total_dynamic_instruction_count": total_dynamic,
				"top_instructions": ", ".join(f"{k}:{v}" for k, v in top_insts),
			}
		)

	return {"rows": processed, "count": len(processed), "limit": limit}


def _format_runtime(value: Any) -> str:
	if value is None:
		return "-"
	try:
		return f"{float(value):.4f}"
	except (TypeError, ValueError):
		return str(value)


def _summary_cards(summary: Dict[str, Any]) -> List[html.Div]:
	items = [
		("Total", summary["total"]),
		("Completed", summary["completed"]),
		("Failed", summary["failed"]),
		("Pending", summary["pending"]),
		("Avg runtime (ms)", _format_runtime(summary["avg_runtime_ms"])),
	]
	return [
		html.Div(
			[
				html.Div(label, className="summary-label"),
				html.Div(str(value), className="summary-value"),
			],
			className="summary-card",
		)
		for label, value in items
	]


def _table_columns(rows: List[Dict[str, Any]], selected_columns: List[str] | None = None) -> List[Dict[str, str]]:
	if not rows:
		return []
	all_columns = list(rows[0].keys())
	visible_columns = [column for column in (selected_columns or all_columns) if column in all_columns]
	return [{"name": column, "id": column} for column in visible_columns]


def _pick_server_port(host: str, preferred_port: int) -> tuple[int, bool]:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			sock.bind((host, preferred_port))
			return preferred_port, False
		except OSError:
			sock.bind((host, 0))
			return int(sock.getsockname()[1]), True


def create_dash_app(db_path: Path) -> Dash:
	summary = _build_summary(db_path)
	options = _build_options(db_path)
	app = Dash(__name__)
	app.title = "ATF Experiment Visualization"

	app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body {
        font-family: "Inter", "Segoe UI", sans-serif;
        margin: 0;
        background: #f8fafc;
        color: #0f172a;
      }
      .page {
        max-width: 1400px;
        margin: 0 auto;
        padding: 24px;
      }
      .subtitle {
        color: #475569;
        margin-bottom: 20px;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 24px;
      }
      .summary-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      .summary-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .summary-value {
        margin-top: 8px;
        font-size: 26px;
        font-weight: 700;
      }
      .panel {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      .filters {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        align-items: end;
        margin-bottom: 14px;
      }
      .filter-label {
        font-size: 13px;
        color: #475569;
        margin-bottom: 6px;
      }
      .controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 12px;
      }
      .muted {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 10px;
      }
      .section-title {
        margin: 0 0 12px 0;
      }
      .columns-panel {
        margin-bottom: 12px;
      }
      .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
      .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
        font-family: "SFMono-Regular", Consolas, monospace;
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>"""

	app.layout = html.Div(
		className="page",
		children=[
			html.H2("ATF Experiment Visualization"),
			html.Div(f"Database: {db_path}", className="subtitle"),
			dcc.Store(id="results-store"),
			dcc.Store(id="inst-store"),
			html.Div(_summary_cards(summary), className="summary-grid"),
			html.Div(
				className="panel",
				children=[
					html.H3("Filters", className="section-title"),
					html.Div(
						className="filters",
						children=[
							html.Div(
								[
									html.Div("Kernel", className="filter-label"),
									dcc.Dropdown(
										id="kernel-filter",
										options=[{"label": value, "value": value} for value in options["kernel_types"]],
										placeholder="All kernels",
										clearable=True,
									),
								]
							),
							html.Div(
								[
									html.Div("Input size", className="filter-label"),
									dcc.Dropdown(
										id="size-filter",
										options=[{"label": value, "value": value} for value in options["input_sizes"]],
										placeholder="All input sizes",
										clearable=True,
									),
								]
							),
							html.Div(
								[
									html.Div("Status", className="filter-label"),
									dcc.Dropdown(
										id="status-filter",
										options=[{"label": value, "value": value} for value in options["statuses"]],
										placeholder="All statuses",
										clearable=True,
									),
								]
							),
							html.Div(
								[
									html.Div("Row limit", className="filter-label"),
									dcc.Input(id="limit-input", type="number", min=1, max=2000, step=1, value=300),
								]
							),
						],
					),
					html.Div(
						className="controls",
						children=[
							html.Button("Reset filters", id="reset-button", n_clicks=0),
							html.Button("Select all columns", id="cols-all", n_clicks=0),
							html.Button("Select no columns", id="cols-none", n_clicks=0),
						],
					),
				],
			),
			html.Div(
				className="panel",
				children=[
					html.H3("Results", className="section-title"),
					html.Div(id="rows-info", className="muted"),
					html.Div(
						className="columns-panel",
						children=[
							html.Div("Shown columns", className="filter-label"),
							dcc.Checklist(id="column-selector", inline=True),
						],
					),
					dash_table.DataTable(
						id="results-table",
						page_action="none",
						sort_action="native",
						filter_action="native",
						style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
						style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
						style_header={"backgroundColor": "#f1f5f9", "fontWeight": "600"},
					),
				],
			),
			html.Div(
				className="panel",
				children=[
					html.H3("LLVM Instruction Count Summary", className="section-title"),
					html.Div(id="inst-info", className="muted"),
					dash_table.DataTable(
						id="inst-table",
						page_action="none",
						sort_action="native",
						filter_action="native",
						style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
						style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
						style_header={"backgroundColor": "#f1f5f9", "fontWeight": "600"},
					),
				],
			),
		],
	)

	@callback(
		Output("kernel-filter", "value"),
		Output("size-filter", "value"),
		Output("status-filter", "value"),
		Output("limit-input", "value"),
		Input("reset-button", "n_clicks"),
		prevent_initial_call=True,
	)
	def _reset_filters(_n_clicks: int):
		return None, None, None, 300

	@callback(
		Output("results-store", "data"),
		Input("kernel-filter", "value"),
		Input("size-filter", "value"),
		Input("status-filter", "value"),
		Input("limit-input", "value"),
	)
	def _load_results(
		kernel_type: str | None,
		input_size: str | None,
		status: str | None,
		limit_value: Any,
	):
		return _build_results_payload(db_path, kernel_type, input_size, status, limit_value)

	@callback(
		Output("inst-store", "data"),
		Input("kernel-filter", "value"),
		Input("size-filter", "value"),
	)
	def _load_inst_counts(kernel_type: str | None, input_size: str | None):
		return _build_inst_counts_payload(db_path, kernel_type, input_size)

	@callback(
		Output("column-selector", "options"),
		Output("column-selector", "value"),
		Input("results-store", "data"),
		Input("cols-all", "n_clicks"),
		Input("cols-none", "n_clicks"),
		State("column-selector", "value"),
	)
	def _sync_selected_columns(
		results_payload: Dict[str, Any] | None,
		_cols_all: int,
		_cols_none: int,
		current_columns: List[str] | None,
	):
		rows = (results_payload or {}).get("rows", [])
		all_columns = list(rows[0].keys()) if rows else []
		options = [{"label": column, "value": column} for column in all_columns]
		triggered = ctx.triggered_id

		if not all_columns:
			return options, []
		if triggered == "cols-none":
			return options, []
		if triggered == "cols-all":
			return options, all_columns

		current_columns = current_columns or []
		filtered_columns = [column for column in current_columns if column in all_columns]
		return options, filtered_columns or all_columns

	@callback(
		Output("rows-info", "children"),
		Output("results-table", "columns"),
		Output("results-table", "data"),
		Output("inst-info", "children"),
		Output("inst-table", "columns"),
		Output("inst-table", "data"),
		Input("results-store", "data"),
		Input("inst-store", "data"),
		Input("column-selector", "value"),
	)
	def _render_outputs(
		results_payload: Dict[str, Any] | None,
		inst_payload: Dict[str, Any] | None,
		selected_columns: List[str] | None,
	):
		results_payload = results_payload or {"rows": [], "count": 0, "limit": 300}
		inst_payload = inst_payload or {"rows": [], "count": 0, "limit": 100}
		rows = results_payload["rows"]
		inst_rows = inst_payload["rows"]

		if rows and selected_columns == [] and ctx.triggered_id == "column-selector":
			result_columns = []
		else:
			result_columns = _table_columns(rows, selected_columns)
		inst_columns = _table_columns(inst_rows)

		return (
			f"Showing {results_payload['count']} rows (limit={results_payload['limit']})",
			result_columns,
			rows,
			f"Showing {inst_payload['count']} LLVM instruction-count row(s)",
			inst_columns,
			inst_rows,
		)

	return app


def main() -> int:
	parser = argparse.ArgumentParser(description="Start web visualization server for experiment SQLite database")
	parser.add_argument("--db", default="experiments.db", help="Path to SQLite experiment database")
	parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
	parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
	args = parser.parse_args()

	db_path = Path(args.db).resolve()
	if not db_path.exists():
		raise SystemExit(f"Database not found: {db_path}")

	app = create_dash_app(db_path)
	actual_port, used_fallback = _pick_server_port(args.host, args.port)
	if used_fallback:
		print(f"Port {args.port} is already in use. Falling back to available port {actual_port}.")
	print(f"Visualization server started on http://{args.host}:{actual_port}")
	print(f"Using database: {db_path}")
	print("Press Ctrl+C to stop.")
	app.run(host=args.host, port=actual_port, debug=False)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
