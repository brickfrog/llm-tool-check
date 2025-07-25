#!/usr/bin/env python3
"""Test structured output using pydantic-ai style approach."""

import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


async def test_pydantic_style():
    api_key = os.getenv("IO_API_KEY")
    if not api_key:
        print("Error: IO_API_KEY not set")
        return

    client = AsyncOpenAI(
        api_key=api_key, base_url="https://api.intelligence.io.solutions/api/v1"
    )

    # Test 1: System message approach (how pydantic-ai might do it)
    print("Test 1: System message with JSON schema")
    schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "temperature": {"type": "number"},
            "conditions": {"type": "string"},
        },
        "required": ["location", "temperature", "conditions"],
    }

    try:
        response = await client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that responds ONLY with valid JSON matching this schema: {json.dumps(schema)}. Do not include any other text, markdown formatting, or code blocks.",
                },
                {"role": "user", "content": "What's the weather like in London?"},
            ],
            max_tokens=500,
        )
        print(f"Response: {response.choices[0].message.content}")
        try:
            parsed = json.loads(response.choices[0].message.content)
            print(f"✓ Valid JSON with keys: {list(parsed.keys())}")
        except (json.JSONDecodeError, KeyError, IndexError):
            print("✗ Failed to parse JSON")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Tool/Function calling approach
    print("\nTest 2: Using tool/function format")
    try:
        response = await client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in London? Use the get_weather function.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather data",
                        "parameters": schema,
                    },
                }
            ],
            tool_choice="required",
            max_tokens=500,
        )

        if response.choices[0].message.tool_calls:
            print(
                f"✓ Tool call made: {response.choices[0].message.tool_calls[0].function.name}"
            )
            print(
                f"Arguments: {response.choices[0].message.tool_calls[0].function.arguments}"
            )
        else:
            print(f"✗ No tool call. Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: JSON mode (simpler)
    print("\nTest 3: JSON mode")
    try:
        response = await client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[
                {"role": "system", "content": "Respond only with valid JSON."},
                {
                    "role": "user",
                    "content": "What's the weather in London? Include location, temperature (as number), and conditions fields.",
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        print(f"Response: {response.choices[0].message.content}")
        try:
            parsed = json.loads(response.choices[0].message.content)
            print(f"✓ Valid JSON with keys: {list(parsed.keys())}")
        except (json.JSONDecodeError, KeyError, IndexError):
            print("✗ Failed to parse JSON")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_pydantic_style())
