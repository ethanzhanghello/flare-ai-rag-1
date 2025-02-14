import json
from typing import Any


def parse_openrouter_response(response: dict) -> str:
    """Parse response from chat completion endpoint"""
    return response.get("choices", [])[0].get("message", {}).get("content", "")


def parse_json_response(response: dict) -> dict[str, Any]:
    """Parse response from chat completion endpoint"""
    json_data = parse_openrouter_response(response)
    return json.loads(json_data)
