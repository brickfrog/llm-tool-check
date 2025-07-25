"""
Data processing functions for website generation.
"""

from collections import defaultdict


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
