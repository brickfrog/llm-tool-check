"""
HTML generation functions for website generation.
"""

import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from generator.data_loader import load_models_mapping
from generator.data_processor import (
    get_all_providers,
    create_unified_model_list,
    has_structured_output_data,
)
from generator.status_calculator import get_cell_status, format_reasons_for_tooltip


def load_template(template_name):
    """Load a template file from the templates directory."""
    template_path = f"src/generator/templates/{template_name}"
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            return f.read()
    return ""


def generate_html_head(title="LLM Tool Support Matrix"):
    """Generate the HTML head section."""
    css_content = load_template("styles.css")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css_content}
    </style>
</head>"""


def generate_legend():
    """Generate the status legend."""
    return """
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color success-swatch"></div>
            <span>3/3 Success</span>
        </div>
        <div class="legend-item">
            <div class="legend-color partial-swatch"></div>
            <span>1-2/3 Partial</span>
        </div>
        <div class="legend-item">
            <div class="legend-color failure-swatch"></div>
            <span>0/3 Failure</span>
        </div>
        <div class="legend-item">
            <div class="legend-color not-available-swatch"></div>
            <span>Not Available</span>
        </div>
    </div>"""


def generate_filter_controls():
    """Generate the platform filter controls."""
    return """
    <div class="filter-controls">
        <h3>Filter by Platform</h3>
        <div class="filter-checkboxes">
            <div class="filter-checkbox">
                <input type="checkbox" id="filter-OR" checked>
                <label for="filter-OR">OpenRouter</label>
            </div>
            <div class="filter-checkbox">
                <input type="checkbox" id="filter-HF" checked>
                <label for="filter-HF">Hugging Face</label>
            </div>
            <div class="filter-checkbox">
                <input type="checkbox" id="filter-iointel" checked>
                <label for="filter-iointel">io.net</label>
            </div>
        </div>
    </div>"""


def generate_table_headers(providers):
    """Generate table headers for providers."""
    headers = ['<th class="model-header">Model</th>']
    for i, provider in enumerate(providers):
        headers.append(
            f'<th class="provider-header" data-provider="{provider}" data-col-index="{i + 1}">{provider}</th>'
        )
    return "".join(headers)


def generate_table_row(model, providers, data_type="tool_support"):
    """Generate a single table row for a model."""
    platform = model["platform"]
    model_data = model["model_data"]
    display_name = model["display_name"]

    cells = [f'<td class="model-name-cell">{display_name}</td>']

    for provider in providers:
        status, text, reasons = get_cell_status(model_data, provider, data_type)
        tooltip = format_reasons_for_tooltip(reasons) if reasons else ""

        cell_html = f'<span class="cell {status}"'
        if tooltip:
            cell_html += f' title="{tooltip}"'
        cell_html += f">{text}</span>"

        cells.append(
            f'<td class="provider-cell" data-provider="{provider}">{cell_html}</td>'
        )

    return f'<tr data-platform="{platform}">{"".join(cells)}</tr>'


def generate_table(unified_models, providers, data_type="tool_support", table_id=""):
    """Generate a complete table for the given data type."""
    headers = generate_table_headers(providers)

    rows = []
    for model in unified_models:
        rows.append(generate_table_row(model, providers, data_type))

    table_html = f"""
    <div class="table-container">
        <table id="{table_id}">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>"""

    return table_html


def generate_library_content(iointel_library_results):
    """Generate content for the Library tab showing iointel library results."""
    if not iointel_library_results:
        return """
            <div class="table-container">
                <p style="text-align: center; color: #6c757d; padding: 40px;">
                    No iointel library test results available.<br>
                    <span style="font-size: 12px; color: #868e96;">
                        Run src/checkers/iointel/ionet.py to generate library-based test results.
                    </span>
                </p>
            </div>"""

    # Check if we have structured output data
    has_structured = has_structured_output_data(iointel_library_results)

    # Create tables for library results
    library_models = []
    for model in iointel_library_results.get("models", []):
        library_models.append(
            {
                "display_name": model.get("model_name", model["model_id"]),
                "platform": "iointel-library",
                "model_data": model,
            }
        )

    # Generate tables
    tool_support_table = generate_table(
        library_models, ["io.net"], "tool_support", "library-tool-support-table"
    )

    if not has_structured:
        # Only tool support
        return f"""
            <div class="library-notice" style="background: #f8f9fa; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
                <strong>iointel Library Results</strong><br>
                <span style="font-size: 14px; color: #6c757d;">
                    These results show AI model capabilities when accessed through the iointel Python library,
                    which may differ from direct HTTP API calls.
                </span>
            </div>
            {tool_support_table}"""
    else:
        # Both tool support and structured output
        structured_output_table = generate_table(
            library_models,
            ["io.net"],
            "structured_output",
            "library-structured-output-table",
        )

        return f"""
            <div class="library-notice" style="background: #f8f9fa; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
                <strong>iointel Library Results</strong><br>
                <span style="font-size: 14px; color: #6c757d;">
                    These results show AI model capabilities when accessed through the iointel Python library,
                    which may differ from direct HTTP API calls.
                </span>
            </div>
            <div class="nested-tabs">
                <div class="nested-tab active" id="nested-tab-library-tool">Tool Support</div>
                <div class="nested-tab" id="nested-tab-library-structured">Structured Output</div>
            </div>

            <div class="nested-tab-content active" id="content-library-tool">
                {tool_support_table}
            </div>

            <div class="nested-tab-content" id="content-library-structured">
                {structured_output_table}
            </div>"""


def generate_tabs_structure(
    unified_models, providers, has_structured_data, iointel_library_results=None
):
    """Generate the complete tabs structure exactly like the original with nested tabs."""
    tool_support_table = generate_table(
        unified_models, providers, "tool_support", "tool-support-table"
    )

    if not has_structured_data:
        # Only tool support, no nested tabs needed
        return f"""
        <div class="tabs">
            <div class="tab active" id="tab-http">HTTP</div>
            <div class="tab" id="tab-library">iointel</div>
        </div>
        <div class="tab-content active" id="content-http">
            {tool_support_table}
        </div>
        <div class="tab-content" id="content-library">
            {generate_library_content(iointel_library_results)}
        </div>"""
    else:
        # Both tool support and structured output with nested tabs
        structured_output_table = generate_table(
            unified_models, providers, "structured_output", "structured-output-table"
        )

        return f"""
        <div class="tabs">
            <div class="tab active" id="tab-http">HTTP</div>
            <div class="tab" id="tab-library">iointel</div>
        </div>
        <div class="tab-content active" id="content-http">
            <div class="nested-tabs">
                <div class="nested-tab active" id="nested-tab-http-tool">Tool Support</div>
                <div class="nested-tab" id="nested-tab-http-structured">Structured Output</div>
            </div>

            <div class="nested-tab-content active" id="content-http-tool">
                {tool_support_table}
            </div>

            <div class="nested-tab-content" id="content-http-structured">
                {structured_output_table}
            </div>
        </div>

        <div class="tab-content" id="content-library">
            {generate_library_content(iointel_library_results)}
        </div>"""


def generate_footer(generated_at):
    """Generate the footer section."""
    return f"""
    <div class="footer">
        <p>Last updated: {generated_at}</p>
        <p>
            <a href="https://openrouter.ai" target="_blank">OpenRouter</a> |
            <a href="https://huggingface.co" target="_blank">Hugging Face</a> |
            <a href="https://io.net" target="_blank">io.net</a>
        </p>
    </div>"""


def generate_html_end():
    """Generate the HTML end section with JavaScript."""
    js_content = load_template("script.js")
    return f"""
    <script>
{js_content}
    </script>
</body>
</html>"""


def generate_complete_html(
    results, hf_results=None, iointel_results=None, iointel_library_results=None
):
    """Generate the complete HTML page."""
    # Load models mapping
    models_mapping = load_models_mapping()

    # Use the most recent timestamp
    timestamps = []
    if results:
        timestamps.append(results.get("generated_at", datetime.now().isoformat()))
    if hf_results:
        timestamps.append(hf_results.get("generated_at", datetime.now().isoformat()))
    if iointel_results:
        timestamps.append(
            iointel_results.get("generated_at", datetime.now().isoformat())
        )

    generated_at = max(timestamps) if timestamps else datetime.now().isoformat()

    # Get all providers and check for structured output data
    all_providers = get_all_providers(results, has_iointel=bool(iointel_results))
    has_structured_data = (
        has_structured_output_data(results)
        or has_structured_output_data(hf_results)
        or has_structured_output_data(iointel_results)
    )

    # Create unified model list
    unified_models = create_unified_model_list(
        results, hf_results, iointel_results, models_mapping
    )

    # Generate HTML sections
    html_head = generate_html_head()

    body_content = f"""
<body>
    <div class="container">
        <h1>AI Model Tool Support Matrix</h1>
        <div class="subtitle">Testing function calling capabilities across OpenRouter, Hugging Face, and io.net platforms</div>

        {generate_legend()}
        {generate_filter_controls()}
        {generate_tabs_structure(unified_models, all_providers, has_structured_data, iointel_library_results)}
        {generate_footer(generated_at)}
    </div>
    {generate_html_end()}"""

    return html_head + body_content
