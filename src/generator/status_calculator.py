"""
Status calculation functions for website generation.
"""


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

    # Special handling for io.net and iointel library models (they don't have providers, results are direct)
    if (
        provider_name == "io.net" or provider_name == "iointel (Library)"
    ) and is_iointel_model:
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
