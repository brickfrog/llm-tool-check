#!/usr/bin/env python3
"""
Generate a static website from the tool support test results.
"""

import json
import os
from datetime import datetime
from collections import defaultdict


def load_latest_results():
    """Load the latest test results from JSON."""
    results_file = "data.json"

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_hf_results():
    """Load the latest Hugging Face test results from JSON."""
    results_file = "data_hf.json"

    if not os.path.exists(results_file):
        print(f"HF results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_iointel_results():
    """Load the latest iointel test results from JSON."""
    results_file = "data_iointel.json"

    if not os.path.exists(results_file):
        print(f"iointel results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_models_mapping():
    """Load the models mapping file."""
    models_file = "models.json"

    if not os.path.exists(models_file):
        print(f"Models mapping file not found: {models_file}")
        return {}

    with open(models_file, "r") as f:
        return json.load(f)


def group_models(models):
    """Group models by base name (e.g., group 'model' and 'model:free' together)."""
    grouped = defaultdict(list)

    for model in models:
        model_id = model["model_id"]
        # Extract base model name (before :free or other suffixes)
        base_name = model_id.split(":")[0]
        grouped[base_name].append(model)

    return grouped


def create_unified_model_list(or_results, hf_results, iointel_results, models_mapping):
    """Create a unified list of models with user-friendly names and platform suffixes."""
    unified_models = []

    # Create a reverse mapping from model IDs to user-friendly names
    id_to_name_hf = {}
    id_to_name_iointel = {}
    for friendly_name, platforms in models_mapping.items():
        for hf_id in platforms.get("huggingface", []):
            id_to_name_hf[hf_id] = (friendly_name, "HF")
        iointel_id = platforms.get("iointel")
        if iointel_id:
            id_to_name_iointel[iointel_id] = (friendly_name, "iointel")

    # Process OpenRouter results by iterating through models_mapping
    if or_results:
        or_models_data_map = {m["model_id"]: m for m in or_results.get("models", [])}

        for friendly_name, platforms in models_mapping.items():
            or_ids = platforms.get("openrouter", [])
            if not or_ids:
                continue

            model_data_variants = {}
            actual_model_id_for_sort = None
            has_regular_variant_defined = False
            has_free_variant_defined = False

            # Check for regular (non-free) and free variants based on suffix
            # Assumes models.json lists specific IDs like 'model/name' and 'model/name:free'
            regular_id_found = None
            free_id_found = None

            for or_id in or_ids:
                if or_id.endswith(":free"):
                    if not free_id_found:  # Take first free ID
                        free_id_found = or_id
                        has_free_variant_defined = True
                else:
                    if not regular_id_found:  # Take first regular ID
                        regular_id_found = or_id
                        has_regular_variant_defined = True

            if regular_id_found and regular_id_found in or_models_data_map:
                model_data_variants["regular"] = or_models_data_map[regular_id_found]
                actual_model_id_for_sort = regular_id_found

            if free_id_found and free_id_found in or_models_data_map:
                model_data_variants["free"] = or_models_data_map[free_id_found]
                if not actual_model_id_for_sort:
                    actual_model_id_for_sort = free_id_found

            if model_data_variants:  # If data found for at least one variant
                # Store whether variants were defined, to distinguish from single-entry models later
                model_data_variants["_has_regular_defined"] = (
                    has_regular_variant_defined
                )
                model_data_variants["_has_free_defined"] = has_free_variant_defined

                unified_models.append(
                    {
                        "display_name": f"{friendly_name} (OR)",
                        "platform": "OR",
                        "model_data": model_data_variants,  # Dict: {"regular": ..., "free": ..., "_has_..."}
                        "sort_key": (
                            friendly_name,
                            "OR",
                            actual_model_id_for_sort
                            if actual_model_id_for_sort
                            else "",
                        ),
                    }
                )

    # Process HuggingFace results (largely unchanged)
    if hf_results:
        for model_data in hf_results.get("models", []):
            model_id = model_data["model_id"]
            if model_id in id_to_name_hf:
                friendly_name, _ = id_to_name_hf[model_id]
                unified_models.append(
                    {
                        "display_name": f"{friendly_name} (HF)",
                        "platform": "HF",
                        "model_data": model_data,  # Direct model_data for HF
                        "sort_key": (friendly_name, "HF", model_id),
                    }
                )

    # Process iointel results
    if iointel_results:
        for model_data in iointel_results.get("models", []):
            model_id = model_data["model_id"]
            if model_id in id_to_name_iointel:
                friendly_name, _ = id_to_name_iointel[model_id]
                unified_models.append(
                    {
                        "display_name": f"{friendly_name} (io.net)",
                        "platform": "iointel",
                        "model_data": model_data,  # Direct model_data for iointel
                        "sort_key": (friendly_name, "iointel", model_id),
                    }
                )

    # Sort by friendly name, then platform
    unified_models.sort(key=lambda x: x["sort_key"])

    return unified_models


def normalize_provider_names_in_results(results_data):
    """Normalize provider names to lowercase in the results data."""
    if not results_data or "models" not in results_data:
        return

    for model_idx, model in enumerate(results_data["models"]):
        # Normalize in 'providers' list
        if "providers" in model and isinstance(model["providers"], list):
            for provider_idx, provider_info in enumerate(model["providers"]):
                if (
                    isinstance(provider_info, dict)
                    and "provider_name" in provider_info
                    and provider_info["provider_name"]
                ):
                    # Ensure in-place modification
                    normalized_name = str(provider_info["provider_name"]).lower()
                    if normalized_name == "fireworks-ai":
                        normalized_name = "fireworks"
                    results_data["models"][model_idx]["providers"][provider_idx][
                        "provider_name"
                    ] = normalized_name

        # Normalize in 'structured_output' list (if it exists)
        if "structured_output" in model and isinstance(
            model["structured_output"], list
        ):
            for provider_idx, provider_info in enumerate(model["structured_output"]):
                if (
                    isinstance(provider_info, dict)
                    and "provider_name" in provider_info
                    and provider_info["provider_name"]
                ):
                    # Ensure in-place modification
                    normalized_name = str(provider_info["provider_name"]).lower()
                    if normalized_name == "fireworks-ai":
                        normalized_name = "fireworks"
                    results_data["models"][model_idx]["structured_output"][
                        provider_idx
                    ]["provider_name"] = normalized_name


def get_all_providers(results, has_iointel=False):
    """Get a sorted list of all unique providers across all models."""
    providers = set()

    for model in results["models"]:
        for provider in model["providers"]:
            providers.add(provider["provider_name"])

    # Add an "io.net" column for direct platform testing
    if has_iointel:
        providers.add("io.net")

    return sorted(list(providers))


def has_structured_output_data(results):
    """Check if the results contain structured output test data."""
    if not results or "models" not in results:
        return False

    # Check if any model has structured_output data
    for model in results["models"]:
        if "structured_output" in model:
            return True

    return False


# Helper function to get status for a single model variant
def _get_single_model_provider_status(
    single_model_data, provider_name, data_type="tool_support"
):
    """Get the status for a specific model-provider combination for a single model data object."""
    if not single_model_data:  # Added check for None
        return "none", "-", None

    # Check if this is an iointel model (has 'summary' and 'structured_output' but no 'providers')
    is_iointel_model = (
        "summary" in single_model_data
        and "structured_output" in single_model_data
        and "providers" not in single_model_data
    )

    # Special handling for io.net models (they don't have providers, results are direct)
    if provider_name == "io.net":
        if data_type == "structured_output":
            # Check for structured output data
            if (
                "structured_output" in single_model_data
                and single_model_data["structured_output"]
            ):
                structured_summary = single_model_data["structured_output"][0][
                    "summary"
                ]
                success_count = structured_summary.get("success_count", 0)
                if success_count == 3:
                    return "success", "3/3", None
                elif success_count > 0:
                    return "partial", f"{success_count}/3", None
                else:
                    return "failure", "0/3", None
            else:
                return "none", "-", None
        else:  # data_type == "tool_support"
            # Check for tool support data
            if "summary" in single_model_data:
                summary = single_model_data["summary"]
                success_count = summary.get("success_count", 0)
                if success_count == 3:
                    return "success", "3/3", None
                elif success_count > 0:
                    return "partial", f"{success_count}/3", None
                else:
                    return "failure", "0/3", None
            else:
                return "none", "-", None

    # For iointel models with non-"io.net" providers, return early since they don't have provider data
    if is_iointel_model:
        return "none", "-", None

    providers_list_key = ""
    if data_type == "structured_output":
        if "structured_output" in single_model_data:
            providers_list_key = "structured_output"
        else:
            # If requesting structured_output and the key doesn't exist for this model variant,
            # then there's no SO data for it.
            return "none", "-", None
    else:  # data_type == "tool_support" (or default)
        # For tool_support, we assume the 'providers' key should exist if single_model_data is valid.
        # If it's missing, the check below will handle it.
        providers_list_key = "providers"

    if (
        providers_list_key not in single_model_data
        or not single_model_data[providers_list_key]
    ):
        return "none", "-", None

    providers_list = single_model_data[providers_list_key]

    for provider in providers_list:
        if provider["provider_name"] == provider_name:
            summary = provider.get("summary", {})
            success_count = summary.get(
                "success_count", -1
            )  # Default to -1 if not found

            if success_count == -1:  # Provider listed but no summary/success_count
                return "none", "?", ["Missing summary data"]

            # Determine status and details
            if success_count == 3:
                return "success", f"{success_count}/3", None
            elif success_count == 0:
                reasons = []
                for run in provider.get("test_runs", []):
                    if run.get("status") == "error" and run.get("error"):
                        error = str(run["error"])[:100]
                        if error not in reasons:
                            reasons.append(error)
                    elif run.get("status") == "unclear":
                        reasons.append("Empty response")
                    elif run.get("status") in [
                        "no_tool_call",
                        "invalid_json",
                        "invalid_schema",
                    ]:
                        if run.get("response_content"):
                            reasons.append(
                                f"No proper response: {str(run['response_content'])[:50]}..."
                            )
                        else:
                            reasons.append("No proper response (empty)")
                return (
                    "failure",
                    f"{success_count}/3",
                    reasons if reasons else ["Unknown failure"],
                )
            else:  # Partial success (1 or 2)
                reasons = []
                for run in provider.get("test_runs", []):
                    if run.get("status") != "success":
                        if run.get("status") == "error" and run.get("error"):
                            reasons.append(f"Error: {str(run['error'])[:50]}...")
                        elif run.get("status") == "unclear":
                            reasons.append("Empty response")
                        elif run.get("status") in [
                            "no_tool_call",
                            "invalid_json",
                            "invalid_schema",
                        ]:
                            reasons.append("Invalid response format")
                return (
                    "partial",
                    f"{success_count}/3",
                    reasons if reasons else ["Unknown partial failure"],
                )

    return "none", "-", None  # Provider not found for this model


def get_cell_status(model_data_container, provider_name, data_type="tool_support"):
    """Get the status for a specific model-provider combination.

    Args:
        model_data_container: For HF, the model data dict. For OR, a dict like
                              {'regular': data, 'free': data, '_has_regular_defined': bool, '_has_free_defined': bool}.
        provider_name: The provider name to check.
        data_type: Either "tool_support" (default) or "structured_output".
    """
    # Check if it's an OpenRouter model with variants structure
    if isinstance(model_data_container, dict) and (
        "regular" in model_data_container or "free" in model_data_container
    ):
        reg_data = model_data_container.get("regular")
        free_data = model_data_container.get("free")
        has_regular_defined = model_data_container.get("_has_regular_defined", False)
        has_free_defined = model_data_container.get("_has_free_defined", False)

        status_reg, text_reg, reasons_reg = _get_single_model_provider_status(
            reg_data, provider_name, data_type
        )
        status_free, text_free, reasons_free = _get_single_model_provider_status(
            free_data, provider_name, data_type
        )

        # Determine if data exists for this provider for each variant
        has_reg_provider_data = status_reg != "none"
        has_free_provider_data = status_free != "none"

        # Case 1: Only one variant type was defined in models.json for this model (e.g. only paid, no free counterpart)
        if has_regular_defined and not has_free_defined:
            return status_reg, text_reg, reasons_reg
        if has_free_defined and not has_regular_defined:
            return status_free, text_free, reasons_free

        # Case 2: Both variant types defined in models.json, now combine their provider status
        combined_text_parts = []
        if has_reg_provider_data:
            combined_text_parts.append(text_reg)
        if has_free_provider_data:
            combined_text_parts.append(text_free)

        final_text = " | ".join(combined_text_parts) if combined_text_parts else "-"

        # Determine overall status class (prioritize success > partial > failure > none)
        final_status = "none"
        # If either is success, final is success. If either is partial (and none success), final is partial etc.
        if status_reg == "success" or status_free == "success":
            final_status = "success"
        elif status_reg == "partial" or status_free == "partial":
            final_status = "partial"
        elif status_reg == "failure" or status_free == "failure":
            final_status = "failure"
        else:  # both are none or undefined
            final_status = "none"

        if not has_reg_provider_data and not has_free_provider_data:
            final_status = (
                "none"  # Ensure if no data for provider from either, it's 'none'
            )
            final_text = "-"

        combined_reasons = []
        if reasons_reg:
            combined_reasons.extend(reasons_reg)
        if reasons_free:
            combined_reasons.extend(reasons_free)

        return final_status, final_text, combined_reasons if combined_reasons else None
    else:
        # Fallback for HF models or non-variant OR models (should be direct model_data object)
        return _get_single_model_provider_status(
            model_data_container, provider_name, data_type
        )


def format_reasons_for_tooltip(reasons):
    """Format reasons for tooltip, escaping HTML-sensitive characters."""
    if not reasons:
        return ""
    # Escape single quotes, double quotes, and ampersands for HTML attribute
    return (
        " | ".join(reasons)
        .replace("&", "&amp;")
        .replace("'", "&apos;")
        .replace('"', "&quot;")
    )


def generate_html(results, hf_results=None, iointel_results=None):
    """Generate the HTML content for the website."""
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
    generated_at = datetime.fromisoformat(generated_at).strftime("%Y-%m-%d %H:%M UTC")

    # Create unified model list
    unified_models = create_unified_model_list(
        results, hf_results, iointel_results, models_mapping
    )

    # Check if we have iointel results to add Platform column
    has_iointel_data = (
        iointel_results is not None and len(iointel_results.get("models", [])) > 0
    )

    # Get all providers from all platforms
    or_providers = get_all_providers(results, has_iointel_data) if results else []
    hf_providers = get_all_providers(hf_results, has_iointel_data) if hf_results else []
    all_providers = sorted(list(set(or_providers + hf_providers)))

    # Check if structured output data is available
    has_structured_data = False
    if results:
        has_structured_data = has_structured_data or has_structured_output_data(results)
    if hf_results:
        has_structured_data = has_structured_data or has_structured_output_data(
            hf_results
        )
    if iointel_results:
        has_structured_data = has_structured_data or has_structured_output_data(
            iointel_results
        )

    # CSS styles - add tab styles if we have structured output data
    style_sheet = """<style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.4; /* Further reduced line height */
            color: #333;
            background-color: #f4f7f9;
            margin: 0;
            padding: 5px; /* Further reduced padding */
        }
        .container {
            max-width: 99%; /* Maximize width */
            margin: 0 auto;
            padding: 10px; /* Further reduced padding */
            background-color: #fff;
            border-radius: 4px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        }
        h1 {
            font-size: 22px; /* Further reduced font size */
            margin-bottom: 4px; /* Further reduced margin */
            color: #2c3e50;
            text-align: center;
        }
        .subtitle {
            color: #555;
            font-size: 12px; /* Further reduced font size */
            margin-bottom: 10px; /* Further reduced margin */
            text-align: center;
        }
        .legend {
            margin-bottom: 10px; /* Further reduced margin */
            padding: 8px; /* Further reduced padding */
            background-color: #f8f9fa;
            border-radius: 3px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px 12px; /* Further reduced gap */
            justify-content: center;
            border: 1px solid #dee2e6;
        }
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 11px; /* Further reduced font size */
        }
        .legend-color {
            width: 12px; /* Smaller swatch */
            height: 12px; /* Smaller swatch */
            margin-right: 5px;
            border-radius: 2px;
        }
        .legend-color.success-swatch { background-color: #d4edda; border: 1px solid #155724; }
        .legend-color.partial-swatch { background-color: #fff3cd; border: 1px solid #856404; }
        .legend-color.failure-swatch { background-color: #f8d7da; border: 1px solid #721c24; }
        .legend-color.not-available-swatch { background-color: #e9ecef; border: 1px solid #adb5bd; }

        .filter-controls {
            margin: 15px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        .filter-controls h3 {
            margin: 0 0 8px 0;
            font-size: 14px;
            color: #495057;
        }
        .filter-checkboxes {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .filter-checkbox {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .filter-checkbox input[type="checkbox"] {
            margin: 0;
        }
        .filter-checkbox label {
            font-size: 12px;
            color: #495057;
            cursor: pointer;
        }

        .table-container {
            overflow-x: auto;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            margin-bottom: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px; /* Further reduced font size */
            min-width: 800px; /* Adjusted min-width */
        }
        th, td {
            padding: 4px 6px; /* Further reduced padding */
            text-align: left;
            border: 1px solid #e9ecef;
            vertical-align: top;
        }
        th {
            background-color: #f1f3f5;
            color: #343a40;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        th.model-header {
            min-width: 180px; /* Further reduced min-width */
            text-align: left;
            position: sticky;
            left: 0;
            z-index: 11;
            background-color: #e9ecef;
        }
        th.provider-header {
            writing-mode: vertical-rl;
            text-orientation: mixed;
            white-space: nowrap;
            text-align: center;
            min-width: 30px; /* Further reduced min-width */
            max-width: 35px; /* Further reduced max-width */
            height: 120px; /* Further reduced height */
            vertical-align: bottom;
            padding-bottom: 4px;
        }
        tbody tr:hover {
            background-color: #f8f9fa;
        }
        .model-name-cell {
            font-weight: 500;
            color: #2c3e50;
            background-color: #f8f9fa;
            position: sticky;
            left: 0;
            z-index: 5;
        }
        .provider-cell {
            text-align: center;
            min-width: 50px; /* Further reduced min-width */
        }
        .variant-info {
            padding: 1px 0; /* Further reduced padding */
            /* Consider removing border if labels are gone and only one item usually appears */
            /* border-bottom: 1px dashed #e0e0e0; */
        }
        .variant-info:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }
        .variant-info:first-child {
            padding-top: 0;
        }
        /* .variant-label is no longer used for text, but class might be on div */

        .cell {
            display: inline-block;
            padding: 1px 4px; /* Further reduced padding */
            border-radius: 2px;
            font-weight: bold;
            font-size: 10px; /* Further reduced font size */
            cursor: default; /* No longer help cursor as tooltip is gone */
            border: 1px solid transparent;
        }
        .cell.success {
            background-color: #d4edda;
            color: #155724;
            border-color: #c3e6cb;
        }
        .cell.partial {
            background-color: #fff3cd;
            color: #856404;
            border-color: #ffeeba;
        }
        .cell.failure {
            background-color: #f8d7da;
            color: #721c24;
            border-color: #f5c6cb;
        }
        .cell.none {
            color: #6c757d;
            font-weight: normal;
        }
        .footer {
            text-align: center;
            margin-top: 15px; /* Further reduced margin */
            font-size: 10px; /* Further reduced font size to 10px */
            color: #6c757d;
        }
        .footer a {
            color: #007bff;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }

        /* Tab styles for switching between Tool Support and Structured Output */
        .tabs {
            display: flex;
            margin-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }
        .tab {
            padding: 8px 12px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            font-size: 14px;
            background-color: #f8f9fa;
            margin-right: 2px;
        }
        .tab.active {
            background-color: #fff;
            border-color: #dee2e6;
            border-bottom-color: white;
            margin-bottom: -1px;
            font-weight: 600;
            color: #2c3e50;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .nested-tabs {
            display: flex;
            margin: 15px 0 10px 0;
            border-bottom: 1px solid #e9ecef;
        }
        .nested-tab {
            padding: 6px 10px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 3px 3px 0 0;
            font-size: 12px;
            background-color: #f1f3f4;
            margin-right: 1px;
            color: #495057;
        }
        .nested-tab.active {
            background-color: #fff;
            border-color: #e9ecef;
            border-bottom-color: white;
            margin-bottom: -1px;
            font-weight: 500;
            color: #2c3e50;
        }
        .nested-tab-content {
            display: none;
        }
        .nested-tab-content.active {
            display: block;
        }
        .tab-heading {
            font-size: 16px;
            font-weight: 600;
            text-align: center;
            margin: 10px 0;
            color: #2c3e50;
        }
    </style>"""

    # HTML Structure
    html_start = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Tool Support Matrix</title>
    {style_sheet}
</head>
<body>
    <div class="container">
        <h1>AI Model Support Matrix</h1>
        <p class="subtitle">Last updated: {generated_at}</p>

        <div class="legend">
            <div class="legend-item"><span class="legend-color success-swatch"></span>Full support (3/3)</div>
            <div class="legend-item"><span class="legend-color partial-swatch"></span>Partial support (1-2/3)</div>
            <div class="legend-item"><span class="legend-color failure-swatch"></span>No support (0/3)</div>
            <div class="legend-item"><span class="legend-color not-available-swatch"></span>Not available</div>
        </div>

        <div class="filter-controls">
            <h3>Filter by Platform:</h3>
            <div class="filter-checkboxes">
                <div class="filter-checkbox">
                    <input type="checkbox" id="filter-or" checked>
                    <label for="filter-or">OpenRouter (OR)</label>
                </div>
                <div class="filter-checkbox">
                    <input type="checkbox" id="filter-hf" checked>
                    <label for="filter-hf">HuggingFace (HF)</label>
                </div>
                <div class="filter-checkbox">
                    <input type="checkbox" id="filter-ionet" checked>
                    <label for="filter-ionet">io.net</label>
                </div>
            </div>
        </div>
"""

    # Create top-level tabs for HTTP vs Library
    html_start += """
        <div class="tabs">
            <div class="tab active" id="tab-http">HTTP Calls</div>
            <div class="tab" id="tab-library">Library Calls</div>
        </div>

        <div class="tab-content active" id="content-http">
"""

    # If we have structured output data, create nested tabs
    if has_structured_data:
        html_start += """
            <div class="nested-tabs">
                <div class="nested-tab active" id="nested-tab-http-tool">Tool Support</div>
                <div class="nested-tab" id="nested-tab-http-structured">Structured Output</div>
            </div>

            <div class="nested-tab-content active" id="content-http-tool">
"""

    # Provider Headers - reused for both tables
    provider_headers = ""
    for i, provider_name in enumerate(all_providers):
        provider_headers += f"<th class='provider-header' data-provider='{provider_name}' data-col-index='{i + 1}'>{provider_name}</th>"

    # Start tool support table
    html_start += (
        """
                <div class="table-container">
                    <table>
                <thead>
                    <tr>
                        <th class="model-header">Model</th>
"""
        + provider_headers
        + """</tr>
                </thead>
                <tbody>
"""
    )

    # Table Rows for Tool Support
    tool_support_rows_html = ""
    for unified_model in unified_models:
        display_name = unified_model["display_name"]
        model_data = unified_model["model_data"]
        platform = unified_model["platform"]

        # Add data attribute for filtering
        platform_class = platform.lower().replace("iointel", "ionet")
        tool_support_rows_html += f"<tr data-platform='{platform_class}'><td class='model-name-cell'>{display_name}</td>"

        for i, provider_name in enumerate(all_providers):
            # Check if this provider is relevant for this platform
            if platform == "OR":
                # For OpenRouter models, use the existing logic, except show "-" for io.net column
                if provider_name == "io.net":
                    status = "none"
                    text = "-"
                    reasons = None
                else:
                    status, text, reasons = get_cell_status(
                        model_data, provider_name, "tool_support"
                    )
            elif platform == "HF":
                # For HuggingFace models, use the existing logic, except show "-" for io.net column
                if provider_name == "io.net":
                    status = "none"
                    text = "-"
                    reasons = None
                else:
                    status, text, reasons = get_cell_status(
                        model_data, provider_name, "tool_support"
                    )
            elif platform == "iointel":
                # iointel models show results in "io.net" column, "-" for provider columns
                if provider_name == "io.net":
                    # Get the tool support status directly from the model data
                    summary = model_data.get("summary", {})
                    success_count = summary.get("success_count", 0)
                    if success_count == 3:
                        status = "success"
                        text = "3/3"
                    elif success_count > 0:
                        status = "partial"
                        text = f"{success_count}/3"
                    else:
                        status = "failure"
                        text = "0/3"
                    reasons = None
                else:
                    status = "none"
                    text = "-"
                    reasons = None
            else:
                status = "none"
                text = "-"
                reasons = None

            if status != "none":
                tool_support_rows_html += (
                    f"<td class='provider-cell' data-provider='{provider_name}' data-col-index='{i + 1}'>"
                    f"<span class='cell {status}'>{text}</span>"
                    f"</td>"
                )
            else:
                tool_support_rows_html += f"<td class='provider-cell' data-provider='{provider_name}' data-col-index='{i + 1}'><span class='cell none'>-</span></td>"

        tool_support_rows_html += "</tr>"

    # Close tool support table
    tool_support_table_end = """
                </tbody>
                    </table>
                </div>
            </div> <!-- End HTTP tool content -->
"""

    # If we have structured output data, create a structured output table
    structured_output_html = ""
    if has_structured_data:
        structured_output_html = (
            """

            <div class="nested-tab-content" id="content-http-structured">
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th class="model-header">Model</th>
"""
            + provider_headers
            + """</tr>
                        </thead>
                        <tbody>
"""
        )

        # Table Rows for Structured Output using unified model list
        for unified_model in unified_models:
            display_name = unified_model["display_name"]
            model_data = unified_model["model_data"]
            platform = unified_model["platform"]

            # Add data attribute for filtering
            platform_class = platform.lower().replace("iointel", "ionet")
            structured_output_html += f"<tr data-platform='{platform_class}'><td class='model-name-cell'>{display_name}</td>"

            for i, provider_name in enumerate(all_providers):
                # The get_cell_status function already handles checking for structured_output
                # in the right place (variant data for OR, direct for HF)
                status, text, reasons = get_cell_status(
                    model_data, provider_name, "structured_output"
                )

                if status != "none":
                    structured_output_html += (
                        f"<td class='provider-cell' data-provider='{provider_name}' data-col-index='{i + 1}'>"
                        f"<span class='cell {status}'>{text}</span>"
                        f"</td>"
                    )
                else:
                    structured_output_html += f"<td class='provider-cell' data-provider='{provider_name}' data-col-index='{i + 1}'><span class='cell none'>-</span></td>"

            structured_output_html += "</tr>"

        # Close structured output table
        structured_output_html += """
                        </tbody>
                    </table>
                </div>
            </div> <!-- End HTTP structured content -->
        </div> <!-- End HTTP tab content -->

        <div class="tab-content" id="content-library">
            <div class="table-container">
                <p style="text-align: center; color: #6c757d; padding: 40px;">
                    Library-based testing coming soon...<br>
                    <span style="font-size: 12px; color: #868e96;">
                        This will test AI models using native client libraries instead of HTTP APIs.
                    </span>
                </p>
            </div>
        </div> <!-- End library tab content -->
"""

    # Footer content
    footer_html = """
        <br>
        Sometimes, the model (or the provider) does not properly call tools or return structured output. That's why every call is made three times.<br>
        The code to generate this website is available on <a href="https://github.com/Xeophon/llm-tool-check" target="_blank">GitHub</a>.

        <div class="footer">
            <p>Generated automatically by <a href="https://github.com/Xeophon/llm-tool-check" target="_blank">AI Model Tool Support Tracker</a>.</p>
            <p>Updates every 12 hours &bull; Data sources: <a href="https://openrouter.ai/docs/api-reference" target="_blank">OpenRouter API</a> & <a href="https://huggingface.co/docs/api-inference" target="_blank">HuggingFace Inference API</a></p>
        </div>
    </div>

    <!-- JavaScript for tab switching and filtering -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');

            // Tab switching functionality
            tabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    // Remove active class from all tabs and content
                    tabs.forEach(t => t.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));

                    // Add active class to clicked tab
                    this.classList.add('active');

                    // Show corresponding content
                    const contentId = this.id.replace('tab', 'content');
                    document.getElementById(contentId).classList.add('active');
                });
            });

            // Nested tab switching functionality
            const nestedTabs = document.querySelectorAll('.nested-tab');
            const nestedTabContents = document.querySelectorAll('.nested-tab-content');

            nestedTabs.forEach(nestedTab => {
                nestedTab.addEventListener('click', function() {
                    // Find parent tab content to scope nested tab switching
                    const parentTabContent = this.closest('.tab-content');
                    const parentNestedTabs = parentTabContent.querySelectorAll('.nested-tab');
                    const parentNestedContents = parentTabContent.querySelectorAll('.nested-tab-content');

                    // Remove active class from all nested tabs and content in this parent
                    parentNestedTabs.forEach(t => t.classList.remove('active'));
                    parentNestedContents.forEach(c => c.classList.remove('active'));

                    // Add active class to clicked nested tab
                    this.classList.add('active');

                    // Show corresponding nested content
                    const contentId = this.id.replace('nested-tab', 'content');
                    document.getElementById(contentId).classList.add('active');
                });
            });

            // Platform filtering functionality
            const filterCheckboxes = document.querySelectorAll('.filter-checkbox input[type="checkbox"]');

            function applyFilters() {
                const activeFilters = new Set();
                filterCheckboxes.forEach(checkbox => {
                    if (checkbox.checked) {
                        const platform = checkbox.id.replace('filter-', '');
                        activeFilters.add(platform);
                    }
                });

                // Filter all table rows in both tool support and structured output tables
                const allRows = document.querySelectorAll('table tbody tr[data-platform]');
                allRows.forEach(row => {
                    const platform = row.getAttribute('data-platform');
                    if (activeFilters.has(platform)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });

                // Hide/show columns based on visible data
                updateColumnVisibility();
            }

            function updateColumnVisibility() {
                // Get all provider columns
                const allProviderHeaders = document.querySelectorAll('th[data-provider]');

                allProviderHeaders.forEach(header => {
                    const provider = header.getAttribute('data-provider');
                    const colIndex = header.getAttribute('data-col-index');

                    // Check if any visible row has non-dash content for this provider
                    const visibleRows = document.querySelectorAll('table tbody tr[data-platform]:not([style*="display: none"])');
                    let hasData = false;

                    visibleRows.forEach(row => {
                        const cell = row.querySelector(`td[data-provider="${provider}"]`);
                        if (cell) {
                            const cellText = cell.textContent.trim();
                            // Check if cell has meaningful data (not just "-")
                            if (cellText !== '-' && cellText !== '') {
                                hasData = true;
                            }
                        }
                    });

                    // Hide/show the column header and all cells in this column
                    const allCellsInColumn = document.querySelectorAll(`th[data-provider="${provider}"], td[data-provider="${provider}"]`);
                    allCellsInColumn.forEach(cell => {
                        if (hasData) {
                            cell.style.display = '';
                        } else {
                            cell.style.display = 'none';
                        }
                    });
                });
            }

            // Add event listeners to filter checkboxes
            filterCheckboxes.forEach(checkbox => {
                checkbox.addEventListener('change', applyFilters);
            });

            // Apply initial filters
            applyFilters();
        });
    </script>
</body>
</html>"""

    # Combine all parts
    if has_structured_data:
        # Need to close the HTTP tab content and add library tab
        return (
            html_start
            + tool_support_rows_html
            + tool_support_table_end
            + structured_output_html
            + footer_html
        )
    else:
        # No structured output, so close HTTP content immediately after tool support
        no_structured_html = """
        </div> <!-- End HTTP tab content -->

        <div class="tab-content" id="content-library">
            <div class="table-container">
                <p style="text-align: center; color: #6c757d; padding: 40px;">
                    Library-based testing coming soon...<br>
                    <span style="font-size: 12px; color: #868e96;">
                        This will test AI models using native client libraries instead of HTTP APIs.
                    </span>
                </p>
            </div>
        </div> <!-- End library tab content -->
"""
        return (
            html_start
            + tool_support_rows_html
            + tool_support_table_end
            + no_structured_html
            + footer_html
        )


def main():
    """Main function to generate the website."""
    print("Loading test results...")
    results = load_latest_results()

    if not results:
        print("No results found. Run check_all_models.py first.")
        return

    # Load HF results if available
    print("Loading HF test results...")
    hf_results = load_hf_results()

    # Load iointel results if available
    print("Loading iointel test results...")
    iointel_results = load_iointel_results()

    # Normalize provider names in the loaded data
    if results:
        normalize_provider_names_in_results(results)
    if hf_results:
        normalize_provider_names_in_results(hf_results)
    # iointel doesn't have providers, so no normalization needed

    load_models_mapping()

    print("Generating HTML...")
    html = generate_html(results, hf_results, iointel_results)

    # Create output directory
    os.makedirs("docs", exist_ok=True)

    # Write HTML file
    output_file = "docs/index.html"
    with open(output_file, "w") as f:
        f.write(html)

    print(f"Website generated: {output_file}")


if __name__ == "__main__":
    main()
