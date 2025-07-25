#!/usr/bin/env python3
"""
Check tool support for models on io.net using the iointel library.
This uses the iointel library's native Agent and tool registration system
instead of raw HTTP calls.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from iointel import Agent, register_tool
from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()


# Register the test tool using iointel decorator
@register_tool("get_weather")
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather in a given location

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit of temperature (celsius or fahrenheit)

    Returns:
        Weather information as a string
    """
    # This is a dummy implementation for testing
    return f"The weather in {location} is sunny with a temperature of 22 degrees {unit}"


class WeatherResponse(BaseModel):
    """Structured output for weather responses"""

    location: str = Field(..., description="City or location name")
    temperature: float = Field(..., description="Temperature in Celsius")
    conditions: str = Field(..., description="Weather conditions description")


class IoIntelToolSupportChecker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.intelligence.io.solutions/api/v1"
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(3)  # Lower limit for io.net

    async def test_model_with_tools(self, model_id: str) -> Dict[str, Any]:
        """Test if a specific model supports tool calling using iointel Agent."""
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
                # Create an agent with tools
                agent = Agent(
                    name=f"ToolTestAgent_{model_id}",
                    model=model_id,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    instructions="You are a helpful assistant that can use tools when appropriate. Use the get_weather tool to check weather.",
                    tools=[get_weather],  # Pass the actual tool function
                    model_settings=ModelSettings(temperature=0.1, max_tokens=1000),
                )

                # Test the agent
                agent_result = await agent.run(
                    "What's the weather like in San Francisco?"
                )

                # Check if tools were called
                tool_called = False
                tools_used = []

                # Check for tool usage in different possible attributes
                if hasattr(agent_result, "tool_calls") and agent_result.tool_calls:
                    tool_called = True
                    tools_used = [str(tc) for tc in agent_result.tool_calls]
                elif (
                    hasattr(agent_result, "tool_usage_results")
                    and agent_result.tool_usage_results
                ):
                    # Filter out 'final_result' tool which is internal
                    actual_tools = [
                        t
                        for t in agent_result.tool_usage_results
                        if getattr(t, "tool_name", None) != "final_result"
                    ]
                    if actual_tools:
                        tool_called = True
                        tools_used = [
                            f"{getattr(t, 'tool_name', 'unknown')}: {getattr(t, 'output', getattr(t, 'result', str(t)))}"
                            for t in actual_tools
                        ]

                # Extract response
                if hasattr(agent_result, "result"):
                    response_text = str(agent_result.result)
                elif isinstance(agent_result, dict) and "result" in agent_result:
                    response_text = str(agent_result["result"])
                else:
                    response_text = str(agent_result)

                result["response_content"] = response_text
                result["tool_calls"] = tools_used if tools_used else None

                if tool_called:
                    result["tool_call_made"] = True
                    result["status"] = "success"
                    result["supports_tools"] = True
                else:
                    result["tool_call_made"] = False
                    result["status"] = "no_tool_call"
                    result["supports_tools"] = False

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
        """Test if a specific model supports structured output using iointel Agent."""
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
                # Create an agent with structured output
                agent = Agent(
                    name=f"StructuredTestAgent_{model_id}",
                    model=model_id,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    instructions="You are a weather information assistant. Provide weather information in the requested format.",
                    output_type=WeatherResponse,  # Use Pydantic model for structured output
                    model_settings=ModelSettings(temperature=0.1, max_tokens=1000),
                )

                # Test the agent
                agent_result = await agent.run("What's the weather like in London?")

                # Check if we got structured output
                if hasattr(agent_result, "result") and isinstance(
                    agent_result.result, WeatherResponse
                ):
                    weather_data = agent_result.result
                    result["response_content"] = json.dumps(
                        {
                            "location": weather_data.location,
                            "temperature": weather_data.temperature,
                            "conditions": weather_data.conditions,
                        }
                    )
                    result["supports_structured_output"] = True
                    result["status"] = "success"
                elif isinstance(agent_result, dict) and "result" in agent_result:
                    # Try to extract structured data from dict
                    try:
                        if isinstance(agent_result["result"], dict):
                            json_response = agent_result["result"]
                        else:
                            json_response = json.loads(str(agent_result["result"]))

                        if all(
                            key in json_response
                            for key in ["location", "temperature", "conditions"]
                        ):
                            result["response_content"] = json.dumps(json_response)
                            result["supports_structured_output"] = True
                            result["status"] = "success"
                        else:
                            result["supports_structured_output"] = False
                            result["status"] = "invalid_schema"
                    except (json.JSONDecodeError, TypeError):
                        result["supports_structured_output"] = False
                        result["status"] = "invalid_json"
                else:
                    # Try to parse as JSON string
                    response_text = str(agent_result)
                    result["response_content"] = response_text
                    try:
                        json_response = json.loads(response_text)
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
            task = self.test_model_with_tools(model_id)
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

    # Extract io.net models
    models = []
    for model_name, platforms in models_data.items():
        if "iointel" in platforms and platforms["iointel"]:
            models.append((platforms["iointel"], model_name))

    print("io.net Tool Support Checker (iointel library)")
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

        # Combine results like the HTTP checker does
        tool_result["structured_output"] = structured_result["structured_output"]
        all_results["models"].append(tool_result)

        # Add a delay between models to avoid rate limiting
        await asyncio.sleep(5)

    # Save final results
    final_output = "data/data_ionet_iointel.json"
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
    print("\nTool Support (using iointel library):")
    print(f"  Fully supporting (3/3): {total_fully_supporting_tools}")
    print(f"  Partially supporting (1-2/3): {total_partially_supporting_tools}")
    print(f"  Not supporting (0/3): {total_not_supporting_tools}")

    print("\nStructured Output (using iointel library):")
    print(f"  Fully supporting (3/3): {total_fully_supporting_structured}")
    print(f"  Partially supporting (1-2/3): {total_partially_supporting_structured}")
    print(f"  Not supporting (0/3): {total_not_supporting_structured}")


if __name__ == "__main__":
    asyncio.run(main())
