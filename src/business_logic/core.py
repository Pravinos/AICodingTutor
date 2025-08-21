"""
Core business logic for the Interactive Coding Tutor.
"""

import logging
from typing import Dict, List, Any, Optional

from helpers.lm_studio import call_lm_studio, LMStudioError
from helpers.prompt_loader import load_prompt, format_prompt, create_system_prompt, PromptError

logger = logging.getLogger(__name__)


class TutorError(Exception):
    """Custom exception for tutor-related errors."""
    pass


# Supported programming languages
LANGUAGES = {
    "python": "Python",
    "javascript": "JavaScript", 
    "java": "Java",
    "cpp": "C++",
    "csharp": "C#"
}

# Core programming topics
TOPICS = [
    "variables", "data_types", "input_output", "conditionals", "loops",
    "functions", "arrays_lists", "strings", "basic_math", "error_handling"
]


def get_supported_languages() -> Dict[str, str]:
    """Get supported programming languages."""
    return LANGUAGES.copy()


def get_topics(language: str) -> List[str]:
    """Get available topics for a programming language."""
    return TOPICS.copy()


def _create_context(language: str, history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    """Create conversation context with system prompt."""
    context = []
    
    # Add filtered history (user and assistant only)
    if history:
        context.extend([msg for msg in history if msg.get("role") in ["user", "assistant"]])
    
    # Add system prompt
    system_prompt = create_system_prompt(language)
    context.insert(0, {"role": "system", "content": system_prompt})
    
    return context


def handle_query(user_input: str, language: str, model: str, 
                history: Optional[List[Dict[str, Any]]] = None,
                endpoint: str = "http://localhost:1234",
                config: Optional[Dict[str, Any]] = None) -> str:
    """Handle a general query from the user."""
    try:
        context = _create_context(language, history)
        
        # Get generation settings from config
        gen_settings = config.get("generation_settings", {}) if config else {}
        max_tokens = gen_settings.get("max_tokens", 2000)
        temperature = gen_settings.get("temperature", 0.5)
        timeout = gen_settings.get("timeout", 60)
        
        response = call_lm_studio(
            user_input, model, context, endpoint, 
            max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )
        logger.info(f"Handled query for {language}")
        return response
    except (LMStudioError, PromptError) as e:
        logger.error(f"Query failed: {e}")
        raise TutorError(f"Unable to process question: {e}")


def explain_concept(concept: str, language: str, model: str,
                   endpoint: str = "http://localhost:1234",
                   config: Optional[Dict[str, Any]] = None) -> str:
    """Generate explanation for a programming concept."""
    try:
        template = load_prompt("explain_concept", language)
        prompt = format_prompt(template, {
            "concept": concept,
            "language": LANGUAGES.get(language, language)
        })
        
        context = _create_context(language)
        
        # Get generation settings from config
        gen_settings = config.get("generation_settings", {}) if config else {}
        max_tokens = gen_settings.get("max_tokens", 2000)
        temperature = gen_settings.get("temperature", 0.5)
        timeout = gen_settings.get("timeout", 60)
        
        response = call_lm_studio(
            prompt, model, context, endpoint,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )
        logger.info(f"Explained '{concept}' in {language}")
        return response
    except (LMStudioError, PromptError) as e:
        logger.error(f"Concept explanation failed: {e}")
        raise TutorError(f"Unable to explain {concept}: {e}")


def generate_exercise(topic: str, language: str, model: str,
                     endpoint: str = "http://localhost:1234",
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Generate a practice exercise for a topic."""
    try:
        template = load_prompt("generate_exercise", language)
        prompt = format_prompt(template, {
            "topic": topic,
            "language": LANGUAGES.get(language, language)
        })
        
        context = _create_context(language)
        
        # Get generation settings from config
        gen_settings = config.get("generation_settings", {}) if config else {}
        max_tokens = gen_settings.get("max_tokens", 2000)
        temperature = gen_settings.get("temperature", 0.5)
        timeout = gen_settings.get("timeout", 60)
        
        response = call_lm_studio(
            prompt, model, context, endpoint,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )
        logger.info(f"Generated exercise for '{topic}' in {language}")
        
        return {
            "topic": topic,
            "language": language,
            "exercise_text": response,
            "task_description": f"Practice: {topic} in {LANGUAGES.get(language, language)}"
        }
    except (LMStudioError, PromptError) as e:
        logger.error(f"Exercise generation failed: {e}")
        raise TutorError(f"Unable to generate exercise for {topic}: {e}")


def evaluate_user_code(code: str, task: str, language: str, model: str,
                      endpoint: str = "http://localhost:1234",
                      config: Optional[Dict[str, Any]] = None) -> str:
    """Evaluate user-submitted code."""
    try:
        template = load_prompt("evaluate_code", language)
        prompt = format_prompt(template, {
            "user_code": code,
            "task": task,
            "language": LANGUAGES.get(language, language)
        })
        
        context = _create_context(language)
        
        # Get generation settings from config
        gen_settings = config.get("generation_settings", {}) if config else {}
        max_tokens = gen_settings.get("max_tokens", 2000)
        temperature = gen_settings.get("temperature", 0.5)
        timeout = gen_settings.get("timeout", 60)
        
        response = call_lm_studio(
            prompt, model, context, endpoint,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )
        logger.info(f"Evaluated code in {language}")
        return response
    except (LMStudioError, PromptError) as e:
        logger.error(f"Code evaluation failed: {e}")
        raise TutorError(f"Unable to evaluate code: {e}")


def validate_inputs(user_input: str, language: str, model: str) -> bool:
    """Validate user inputs."""
    if not user_input.strip():
        raise TutorError("Please enter a question.")
    
    if language not in LANGUAGES:
        raise TutorError(f"Unsupported language: {language}")
    
    if not model.strip():
        raise TutorError("No model selected.")
    
    if len(user_input) > 2000:
        raise TutorError("Input too long. Keep under 2000 characters.")
    
    return True


def create_chat_message(role: str, content: str) -> Dict[str, str]:
    """Create a chat message."""
    return {"role": role, "content": content}
