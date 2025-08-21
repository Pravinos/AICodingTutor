"""
LM Studio integration helper functions.
"""

import requests
import logging
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)


class LMStudioError(Exception):
    """Custom exception for LM Studio errors."""
    pass


def get_available_models(endpoint: str = "http://localhost:1234",
                        timeout: int = 10) -> List[str]:
    """Get available chat models from LM Studio."""
    try:
        response = requests.get(f"{endpoint}/v1/models", timeout=timeout)
        response.raise_for_status()
        
        all_models = [model["id"] for model in response.json().get("data", [])]
        
        # Filter out embedding models
        chat_models = [m for m in all_models if not any(
            keyword in m.lower() for keyword in ['embedding', 'embed', 'e5-']
        )]
        
        if not chat_models:
            logger.warning(f"No chat models found. Available: {all_models}")
            return ["No chat models available"]
            
        logger.info(f"Found {len(chat_models)} chat models")
        return chat_models
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch models: {e}")
        raise LMStudioError(f"Unable to connect to LM Studio: {e}")


def call_lm_studio(prompt: str, model: str, 
                  context: Optional[List[Dict[str, str]]] = None,
                  endpoint: str = "http://localhost:1234",
                  max_tokens: int = 2000,
                  temperature: float = 0.5,
                  timeout: int = 60) -> str:
    """Call LM Studio for text generation."""
    try:
        if not model or model == "No chat models available":
            raise LMStudioError("No valid model selected")
        
        # Prepare messages
        messages = []
        system_content = ""
        
        # Process context messages
        if context:
            for msg in context:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    continue
                
                role = str(msg["role"])
                content = str(msg["content"])
                
                if role == "system":
                    system_content = content
                elif role in ["user", "assistant"]:
                    messages.append({"role": role, "content": content})
        
        # Add current prompt with system content if any
        user_prompt = f"{system_content}\n\nUser: {prompt}" if system_content else prompt
        messages.append({"role": "user", "content": user_prompt})
        
        # Make API call
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        if not response.ok:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = f" - {error_data.get('error', {}).get('message', 'Unknown')}"
            except:
                error_detail = f" - {response.text[:200]}"
            raise requests.HTTPError(f"{response.status_code} {response.reason}{error_detail}")
        
        data = response.json()
        
        if not data.get("choices") or not data["choices"]:
            raise LMStudioError("No response generated")
            
        response_text = data["choices"][0]["message"]["content"].strip()
        
        if not response_text:
            raise LMStudioError("Empty response generated")
            
        return response_text
        
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise LMStudioError(f"Failed to get response: {e}")
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        logger.error(f"Invalid response format: {e}")
        raise LMStudioError(f"Invalid response format: {e}")


def test_connection(endpoint: str = "http://localhost:1234",
                   timeout: int = 5) -> bool:
    """Test if LM Studio is accessible."""
    try:
        response = requests.get(f"{endpoint}/v1/models", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False
