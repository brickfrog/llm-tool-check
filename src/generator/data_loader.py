"""
Data loading functions for website generation.
"""

import json
import os


def load_latest_results():
    """Load the latest test results from JSON."""
    results_file = "data/data.json"

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_hf_results():
    """Load the latest Hugging Face test results from JSON."""
    results_file = "data/data_hf.json"

    if not os.path.exists(results_file):
        print(f"HF results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_iointel_results():
    """Load the latest io.net HTTP test results from JSON."""
    results_file = "data/data_ionet.json"

    if not os.path.exists(results_file):
        print(f"io.net HTTP results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_iointel_library_results():
    """Load the latest io.net iointel library test results from JSON."""
    results_file = "data/data_ionet_iointel.json"

    if not os.path.exists(results_file):
        print(f"io.net iointel library results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_models_mapping():
    """Load the models mapping file."""
    models_file = "config/models.json"

    if not os.path.exists(models_file):
        print(f"Models mapping file not found: {models_file}")
        return {}

    with open(models_file, "r") as f:
        return json.load(f)
