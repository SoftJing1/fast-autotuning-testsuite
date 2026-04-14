#!/usr/bin/env python3
"""Interactive investigation UI for dynamic BB count validation payloads."""

from __future__ import annotations

from typing import Any, Dict, List

try:
	from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
except ImportError as exc:
	raise SystemExit(
		"Dash is required for this visualization UI. Install dependencies with "
		"`pip install -r requirements.txt`."
	) from exc


def _summary_cards(summary: Dict[str, Any]) -> List[html.Div]:
	items = [
		("Matched", summary.get("matched", 0)),
		("Mismatched", summary.get("mismatched", 0)),
		("Symbolic", summary.get("symbolic", 0)),
		("Crashed", summary.get("crashed", 0)),
		("Tooling", summary.get("tooling", 0)),
		("Other", summary.get("other", summary.get("other_skipped", 0))),
		("Attempted", summary.get("attempted", 0)),
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


def _block_is_comparable(row: Dict[str, Any]) -> bool:
	return row.get("symb_viewer") is not None and row.get("dynamic") is not None


def _block_is_mismatch(row: Dict[str, Any]) -> bool:
	return _block_is_comparable(row) and row.get("symb_viewer") != row.get("dynamic")


def _case_mismatch_count(case: Dict[str, Any]) -> int:
	basic_blocks = case.get("basic_blocks", [])
	if not basic_blocks:
		return int(case.get("mismatch_count", 0) or 0)
	return sum(1 for row in basic_blocks if _block_is_mismatch(row))


def _case_dynamic_only_blocks(case: Dict[str, Any]) -> List[str]:
	ignored = [str(name) for name in case.get("ignored_dynamic_only_blocks", [])]
	if ignored:
		return ignored
	return [
		str(row.get("name"))
		for row in case.get("basic_blocks", [])
		if row.get("symb_viewer") is None and row.get("dynamic") is not None
	]


def _case_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	rows = []
	for case in cases:
		rows.append(
			{
				"case_label": case.get("case_label"),
				"status": case.get("status"),
				"kernel_function": case.get("kernel_function"),
				"mismatch_count": _case_mismatch_count(case),
				"llvm_ir_path": case.get("llvm_ir_path"),
				"config_path": case.get("config_path"),
				"artifact_root": case.get("artifact_root"),
				"reason": case.get("reason", ""),
			}
		)
	return rows


def _metadata_rows(case: Dict[str, Any]) -> List[Dict[str, str]]:
	runtime_ids = case.get("matched_runtime_ids") or {}
	ignored = _case_dynamic_only_blocks(case)
	return [
		{"field": "Status", "value": str(case.get("status", ""))},
		{"field": "Kernel function", "value": str(case.get("kernel_function", ""))},
		{"field": "Validation folder", "value": str(case.get("artifact_root", "")) or "-"},
		{"field": "LLVM IR path", "value": str(case.get("llvm_ir_path", "")) or "-"},
		{"field": "Config path", "value": str(case.get("config_path", "")) or "-"},
		{"field": "Matched group ids", "value": str(runtime_ids.get("group_ids", "")) or "-"},
		{"field": "Matched local ids", "value": str(runtime_ids.get("local_ids", "")) or "-"},
		{"field": "Dynamic-only blocks", "value": ", ".join(ignored) if ignored else "-"},
	]


def _category_options(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
	ordered = ["matched", "mismatched", "crashed", "symbolic", "tooling", "other", "failed"]
	seen = {str(case.get("status", "")) for case in cases}
	options = [{"label": "All", "value": "all"}]
	for value in ordered:
		if value in seen:
			options.append({"label": value, "value": value})
	return options


def _detail_layout(case: Dict[str, Any]) -> List[Any]:
	status = str(case.get("status", ""))
	if status == "mismatched":
		basic_blocks = case.get("basic_blocks", [])
		mismatched_rows = [row for row in basic_blocks if _block_is_mismatch(row)]
		mismatch_count = len(mismatched_rows) if basic_blocks else _case_mismatch_count(case)
		symb_tool_messages = str(case.get("symb_tool_messages", "")).strip()
		return [
			html.Div(
				f"{mismatch_count} mismatched basic block(s). "
				"Use the table below to inspect raw symbolic and dynamic counts.",
				className="muted",
			),
			html.Div("symb-viewer output", className="muted") if symb_tool_messages else html.Div(),
			html.Pre(symb_tool_messages, className="error-box") if symb_tool_messages else html.Div(),
			dash_table.DataTable(
				id="blocks-table",
				columns=[
					{"name": "name", "id": "name"},
					{"name": "symb_viewer", "id": "symb_viewer"},
					{"name": "symb_viewer_raw", "id": "symb_viewer_raw"},
					{"name": "dynamic", "id": "dynamic"},
					{"name": "delta", "id": "delta"},
				],
				data=mismatched_rows,
				page_action="none",
				sort_action="native",
				filter_action="native",
				style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
				style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
				style_header={"backgroundColor": "#efe5d8", "fontWeight": "600"},
			),
		]

	if status == "crashed":
		return [
			html.Div("Crash message", className="muted"),
			html.Pre(str(case.get("reason", "")) or "-", className="error-box"),
		]

	if status in {"symbolic", "tooling", "other", "failed"}:
		raw_symbolic_rows = [
			{
				"name": row.get("name"),
				"symb_viewer_raw": row.get("symb_viewer_raw"),
			}
			for row in case.get("basic_blocks", [])
			if row.get("symb_viewer_raw") is not None and row.get("symb_viewer") is None
		]
		return [
			html.Div("Case message", className="muted"),
			html.Pre(str(case.get("reason", "")) or "-", className="error-box"),
			html.Div("Raw symbolic counts", className="muted") if raw_symbolic_rows else html.Div(),
			dash_table.DataTable(
				id="raw-symbolic-table",
				columns=[
					{"name": "name", "id": "name"},
					{"name": "symb_viewer_raw", "id": "symb_viewer_raw"},
				],
				data=raw_symbolic_rows,
				page_action="none",
				sort_action="native",
				filter_action="native",
				style_table={"overflowX": "auto", "maxHeight": "320px", "overflowY": "auto"},
				style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
				style_header={"backgroundColor": "#efe5d8", "fontWeight": "600"},
			) if raw_symbolic_rows else html.Div(),
		]

	comparable_count = sum(1 for row in case.get("basic_blocks", []) if _block_is_comparable(row))
	dynamic_only_blocks = _case_dynamic_only_blocks(case)
	children: List[Any] = [
		html.Div(
			f"All {comparable_count} comparable basic blocks matched for this case.",
			className="muted",
		)
	]
	if dynamic_only_blocks:
		children.append(
			html.Div(
				"Ignored dynamic-only block(s): " + ", ".join(dynamic_only_blocks),
				className="muted",
			)
		)
	return children


def create_dash_app(payload: Dict[str, Any]) -> Dash:
	cases = list(payload.get("cases", []))
	summary = dict(payload.get("summary", {}))
	case_lookup = {str(case.get("case_label")): case for case in cases}

	app = Dash(__name__)
	app.title = "Dynamic BB Investigation"
	app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --bg: #f5efe6;
        --panel: #fffdf9;
        --ink: #1f2937;
        --muted: #6b7280;
        --line: #dfd6c7;
      }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      }
      .page {
        max-width: 1380px;
        margin: 0 auto;
        padding: 28px 20px 40px;
      }
      .subtitle {
        color: var(--muted);
        margin: 8px 0 20px;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
      }
      .summary-card, .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(41, 37, 36, 0.05);
      }
      .summary-card {
        padding: 14px 16px;
      }
      .summary-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }
      .summary-value {
        font-size: 28px;
        margin-top: 6px;
        font-weight: 700;
      }
      .panel {
        padding: 18px;
        margin-bottom: 18px;
      }
      .controls {
        display: grid;
        grid-template-columns: minmax(240px, 320px) 1fr;
        gap: 12px;
        align-items: end;
      }
      .label {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      .two-col {
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(340px, 0.85fr);
        gap: 18px;
      }
      .muted {
        color: var(--muted);
        font-size: 13px;
      }
      .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        color: #991b1b;
        padding: 12px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }
      @media (max-width: 980px) {
        .controls, .two-col {
          grid-template-columns: 1fr;
        }
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
			html.H2("Dynamic BB Count Investigation"),
			html.Div("Start from the overall summary, then inspect a category and select a specific case.", className="subtitle"),
			dcc.Store(id="selected-case-store"),
			html.Div(_summary_cards(summary), className="summary-grid"),
			html.Div(
				className="panel",
				children=[
					html.Div(
						className="controls",
						children=[
							html.Div(
								[
									html.Div("Category", className="label"),
									dcc.Dropdown(
										id="category-filter",
										options=_category_options(cases),
										value="all",
										clearable=False,
									),
								]
							),
							html.Div(id="case-count-text", className="muted"),
						],
					),
				],
			),
			html.Div(
				className="panel",
				children=[
					html.H3("Cases"),
					dash_table.DataTable(
						id="cases-table",
						columns=[
							{"name": "case_label", "id": "case_label"},
							{"name": "status", "id": "status"},
							{"name": "kernel_function", "id": "kernel_function"},
							{"name": "mismatch_count", "id": "mismatch_count"},
							{"name": "llvm_ir_path", "id": "llvm_ir_path"},
							{"name": "config_path", "id": "config_path"},
						],
						data=[],
						page_action="none",
						sort_action="native",
						row_selectable="single",
						selected_rows=[],
						style_table={"overflowX": "auto", "maxHeight": "320px", "overflowY": "auto"},
						style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
						style_header={"backgroundColor": "#efe5d8", "fontWeight": "600"},
					),
				],
			),
			html.Div(
				className="two-col",
				children=[
					html.Div(
						className="panel",
						children=[
							html.H3("Case Investigation"),
							html.Div(id="case-detail"),
						],
					),
					html.Div(
						className="panel",
						children=[
							html.H3("Selected Case Metadata"),
							dash_table.DataTable(
								id="metadata-table",
								columns=[
									{"name": "field", "id": "field"},
									{"name": "value", "id": "value"},
								],
								page_action="none",
								style_table={"overflowX": "auto"},
								style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
								style_header={"backgroundColor": "#efe5d8", "fontWeight": "600"},
							),
						],
					),
				],
			),
		],
	)

	@callback(
		Output("cases-table", "data"),
		Output("cases-table", "selected_rows"),
		Output("case-count-text", "children"),
		Input("category-filter", "value"),
	)
	def _filter_cases(category: str | None):
		filtered = cases
		if category and category != "all":
			filtered = [case for case in cases if str(case.get("status", "")) == category]
		rows = _case_rows(filtered)
		selected_rows = [0] if rows else []
		return rows, selected_rows, f"Showing {len(rows)} case(s)"

	@callback(
		Output("selected-case-store", "data"),
		Input("cases-table", "selected_rows"),
		State("cases-table", "data"),
	)
	def _select_case(selected_rows: List[int] | None, rows: List[Dict[str, Any]] | None):
		rows = rows or []
		if selected_rows and rows:
			index = selected_rows[0]
			if 0 <= index < len(rows):
				return rows[index]["case_label"]
		if rows:
			return rows[0]["case_label"]
		return None

	@callback(
		Output("case-detail", "children"),
		Output("metadata-table", "data"),
		Input("selected-case-store", "data"),
	)
	def _render_case(case_label: str | None):
		if not case_label:
			return html.Div("No case selected.", className="muted"), []
		case = case_lookup.get(str(case_label))
		if not case:
			return html.Div("Selected case is no longer available.", className="muted"), []
		return _detail_layout(case), _metadata_rows(case)

	return app
