#!/usr/bin/env python3
"""
Check tool support for models on iointel.
Generates a comprehensive report for multiple models.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()


class IoIntelToolSupportChecker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.intelligence.io.solutions/api/v1"
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(3)  # Lower limit for iointel

    async def test_model(self, model_id: str) -> Dict[str, Any]:
        """Test if a specific model supports tool calling."""
        result = {
            "model_id": model_id,
            "supports_tools": False,
            "tool_call_made": False,
            "status": "unknown",  # "success", "error", "unclear", "no_tool_call"
            "error": None,
            "response_content": None,
            "tool_calls": None,
            "finish_reason": None,
            "timestamp": datetime.now().isoformat(),
        }

        async with self.semaphore:  # Limit concurrent requests
            try:
                # Create the completion with tools
                response = await self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": "What's the weather like in San Francisco?",
                        }
                    ],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the current weather in a given location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "The city and state, e.g. San Francisco, CA",
                                        },
                                        "unit": {
                                            "type": "string",
                                            "enum": ["celsius", "fahrenheit"],
                                            "description": "The unit of temperature",
                                        },
                                    },
                                    "required": ["location"],
                                },
                            },
                        }
                    ],
                    max_tokens=1000,
                )

                # Extract debugging information
                message = response.choices[0].message
                result["finish_reason"] = response.choices[0].finish_reason

                # Check if the model made a tool call
                if hasattr(message, "tool_calls") and message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                    result["tool_call_made"] = True
                    result["status"] = "success"
                    result["supports_tools"] = True
                elif message.content:
                    result["response_content"] = message.content
                    result["tool_call_made"] = False
                    result["status"] = "no_tool_call"
                    result["supports_tools"] = False
                else:
                    result["status"] = "unclear"
                    result["supports_tools"] = None

            except Exception as e:
                error_str = str(e)
                result["error"] = error_str
                result["status"] = "error"

                # Analyze error type
                if "tool" in error_str.lower() or "function" in error_str.lower():
                    result["supports_tools"] = False
                elif "404" in error_str:
                    result["supports_tools"] = False
                else:
                    # Other errors - unclear if tools are supported
                    result["supports_tools"] = None

            return result

    async def test_model_structured_output(self, model_id: str) -> Dict[str, Any]:
        """Test if a specific model supports structured output."""
        result = {
            "model_id": model_id,
            "supports_structured_output": False,
            "status": "unknown",  # "success", "error", "unclear", "invalid_json", "invalid_schema"
            "error": None,
            "response_content": None,
            "finish_reason": None,
            "timestamp": datetime.now().isoformat(),
        }

        async with self.semaphore:  # Limit concurrent requests
            try:
                # Create the completion with structured output format
                response = await self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": "What's the weather like in London?",
                        }
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "weather",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City or location name",
                                    },
                                    "temperature": {
                                        "type": "number",
                                        "description": "Temperature in Celsius",
                                    },
                                    "conditions": {
                                        "type": "string",
                                        "description": "Weather conditions description",
                                    },
                                },
                                "required": ["location", "temperature", "conditions"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    max_tokens=1000,
                )

                # Extract debugging information
                message = response.choices[0].message
                result["finish_reason"] = response.choices[0].finish_reason

                # Check if the model returned valid JSON according to our schema
                if message.content:
                    result["response_content"] = message.content
                    try:
                        json_response = json.loads(message.content)
                        if all(
                            key in json_response
                            for key in ["location", "temperature", "conditions"]
                        ):
                            result["supports_structured_output"] = True
                            result["status"] = "success"
                        else:
                            result["supports_structured_output"] = False
                            result["status"] = "invalid_schema"
                    except json.JSONDecodeError:
                        result["supports_structured_output"] = False
                        result["status"] = "invalid_json"
                else:
                    result["status"] = "unclear"
                    result["supports_structured_output"] = None

            except Exception as e:
                error_str = str(e)
                result["error"] = error_str
                result["status"] = "error"
                logging.error(
                    f"{model_id}: Exception in structured output test - {error_str}"
                )

                # Analyze error type
                if any(
                    keyword in error_str.lower()
                    for keyword in [
                        "response_format",
                        "json_schema",
                        "not supported",
                        "invalid",
                    ]
                ):
                    result["supports_structured_output"] = False
                else:
                    # Other errors - unclear if structured output is supported
                    result["supports_structured_output"] = None

            return result

    async def check_model(self, model_id: str, model_name: str) -> Dict[str, Any]:
        """Check tool support for a specific model."""
        print(f"\n{'=' * 60}")
        print(f"Checking tool support for: {model_name} ({model_id})")
        print(f"{'=' * 60}")

        # Create 3 test tasks for tool support
        tasks = []
        for run in range(3):
            task = self.test_model(model_id)
            tasks.append(task)

        print("Running 3 tool support tests concurrently...")

        # Execute all tests concurrently
        results = await asyncio.gather(*tasks)

        # Calculate summary
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        unclear_count = sum(1 for r in results if r["status"] == "unclear")
        no_tool_call_count = sum(1 for r in results if r["status"] == "no_tool_call")

        # Display results
        print(f"  Summary: {success_count}/3 successful")
        if error_count > 0:
            print(f"  Errors: {error_count}/3")
        if unclear_count > 0:
            print(f"  Unclear: {unclear_count}/3")
        if no_tool_call_count > 0:
            print(f"  No tool call: {no_tool_call_count}/3")

        return {
            "model_id": model_id,
            "model_name": model_name,
            "test_runs": results,
            "summary": {
                "total_runs": 3,
                "success_count": success_count,
                "error_count": error_count,
                "unclear_count": unclear_count,
                "no_tool_call_count": no_tool_call_count,
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def check_model_structured_output(
        self, model_id: str, model_name: str
    ) -> Dict[str, Any]:
        """Check structured output support for a specific model."""
        print(f"\n{'=' * 60}")
        print(f"Checking structured output for: {model_name} ({model_id})")
        print(f"{'=' * 60}")

        # Create 3 test tasks for structured output
        tasks = []
        for run in range(3):
            task = self.test_model_structured_output(model_id)
            tasks.append(task)

        print("Running 3 structured output tests concurrently...")

        # Execute all tests concurrently
        results = await asyncio.gather(*tasks)

        # Calculate summary
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        unclear_count = sum(1 for r in results if r["status"] == "unclear")
        invalid_count = sum(
            1 for r in results if r["status"] in ["invalid_json", "invalid_schema"]
        )

        # Display results
        print(f"  Summary: {success_count}/3 successful")
        if error_count > 0:
            print(f"  Errors: {error_count}/3")
        if unclear_count > 0:
            print(f"  Unclear: {unclear_count}/3")
        if invalid_count > 0:
            print(f"  Invalid: {invalid_count}/3")

        return {
            "model_id": model_id,
            "model_name": model_name,
            "structured_output": [
                {
                    "model_id": model_id,
                    "test_runs": results,
                    "summary": {
                        "total_runs": 3,
                        "success_count": success_count,
                        "error_count": error_count,
                        "unclear_count": unclear_count,
                        "invalid_count": invalid_count,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "timestamp": datetime.now().isoformat(),
        }


async def main():
    api_key = os.getenv("IO_API_KEY")
    if not api_key:
        print("Error: IO_API_KEY environment variable not set")
        return

    # Load models from unified models file
    with open("config/models.json", "r") as f:
        models_data = json.load(f)

    # Extract iointel models
    models = []
    for model_name, platforms in models_data.items():
        if "iointel" in platforms and platforms["iointel"]:
            models.append((platforms["iointel"], model_name))

    print("iointel Tool Support Checker")
    print(f"Testing {len(models)} models")
    print("=" * 60)

    checker = IoIntelToolSupportChecker(api_key)

    # Check all models
    all_results = {
        "generated_at": datetime.now().isoformat(),
        "total_models": len(models),
        "models": [],
    }

    # Process all models
    for model_id, model_name in models:
        # Run both tool support and structured output tests
        tool_result = await checker.check_model(model_id, model_name)
        structured_result = await checker.check_model_structured_output(
            model_id, model_name
        )

        # Combine results like OpenRouter does
        tool_result["structured_output"] = structured_result["structured_output"]
        all_results["models"].append(tool_result)

    # Save final results
    final_output = "data/data_ionet.json"
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {final_output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_models = len(models)
    total_fully_supporting_tools = 0  # 3/3 success
    total_partially_supporting_tools = 0  # 1-2/3 success
    total_not_supporting_tools = 0  # 0/3 success

    total_fully_supporting_structured = 0  # 3/3 success
    total_partially_supporting_structured = 0  # 1-2/3 success
    total_not_supporting_structured = 0  # 0/3 success

    for model_data in all_results["models"]:
        # Tool support summary
        summary = model_data["summary"]
        if summary["success_count"] == 3:
            total_fully_supporting_tools += 1
        elif summary["success_count"] > 0:
            total_partially_supporting_tools += 1
        else:
            total_not_supporting_tools += 1

        # Structured output summary
        if "structured_output" in model_data and model_data["structured_output"]:
            structured_summary = model_data["structured_output"][0]["summary"]
            if structured_summary["success_count"] == 3:
                total_fully_supporting_structured += 1
            elif structured_summary["success_count"] > 0:
                total_partially_supporting_structured += 1
            else:
                total_not_supporting_structured += 1

    print(f"\nTotal models tested: {total_models}")
    print("\nTool Support:")
    print(f"  Fully supporting (3/3): {total_fully_supporting_tools}")
    print(f"  Partially supporting (1-2/3): {total_partially_supporting_tools}")
    print(f"  Not supporting (0/3): {total_not_supporting_tools}")

    print("\nStructured Output:")
    print(f"  Fully supporting (3/3): {total_fully_supporting_structured}")
    print(f"  Partially supporting (1-2/3): {total_partially_supporting_structured}")
    print(f"  Not supporting (0/3): {total_not_supporting_structured}")


if __name__ == "__main__":
    asyncio.run(main())
