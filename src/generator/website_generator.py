#!/usr/bin/env python3
"""
Main website generator orchestrator.
Simplified and modular version of the original website.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from generator.data_loader import (
    load_latest_results,
    load_hf_results,
    load_iointel_results,
    load_iointel_library_results,
)
from generator.data_processor import normalize_provider_names_in_results
from generator.html_builder import generate_complete_html


def main():
    """Main function to generate the website."""
    print("Loading test results...")
    results = load_latest_results()

    print("Loading HF test results...")
    hf_results = load_hf_results()

    print("Loading io.net HTTP test results...")
    iointel_results = load_iointel_results()

    print("Loading io.net iointel library test results...")
    iointel_library_results = load_iointel_library_results()

    # Normalize provider names for consistency
    if results:
        normalize_provider_names_in_results(results)
    if hf_results:
        normalize_provider_names_in_results(hf_results)
    if iointel_results:
        normalize_provider_names_in_results(iointel_results)
    if iointel_library_results:
        normalize_provider_names_in_results(iointel_library_results)

    print("Generating HTML...")
    html_content = generate_complete_html(
        results, hf_results, iointel_results, iointel_library_results
    )

    # Write HTML file
    output_file = "docs/index.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"Website generated: {output_file}")


if __name__ == "__main__":
    main()
