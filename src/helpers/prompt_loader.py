"""
Prompt loading and formatting utilities.
"""

import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptError(Exception):
    """Custom exception for prompt-related errors."""
    pass


def _get_prompts_directory() -> Path:
    """Get the prompts directory path."""
    # Try multiple strategies to find the prompts directory
    
    # Strategy 1: Go up from src/helpers/ to project root, then to prompts
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    
    if prompts_dir.exists():
        return prompts_dir
    
    # Strategy 2: Look for prompts directory starting from current file and going up
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "prompts"
        if candidate.exists() and (candidate / "common").exists():
            return candidate
    
    # Strategy 3: Try relative to the working directory
    cwd_prompts = Path.cwd() / "prompts"
    if cwd_prompts.exists():
        return cwd_prompts
    
    # If all strategies fail, provide detailed error info
    logger.error(f"Prompts directory search failed:")
    logger.error(f"  Current file: {Path(__file__).resolve()}")
    logger.error(f"  Working directory: {Path.cwd()}")
    logger.error(f"  Tried: {prompts_dir}")
    logger.error(f"  Tried: {cwd_prompts}")
    
    raise PromptError(f"Prompts directory not found: {prompts_dir}")


def load_prompt(filename: str, language: Optional[str] = None) -> str:
    """Load a prompt template from file."""
    prompts_dir = _get_prompts_directory()
    
    # Try language-specific first
    if language:
        lang_file = prompts_dir / language / f"{filename}.md"
        if lang_file.exists():
            return lang_file.read_text(encoding='utf-8').strip()
    
    # Fall back to common
    common_file = prompts_dir / "common" / f"{filename}.md"
    if common_file.exists():
        return common_file.read_text(encoding='utf-8').strip()
    
    raise PromptError(f"Prompt '{filename}.md' not found")


def format_prompt(template: str, variables: Dict[str, str]) -> str:
    """Format a prompt template with variables."""
    try:
        return template.format(**{k: str(v) for k, v in variables.items()})
    except KeyError as e:
        raise PromptError(f"Missing variable: {e}")


def create_system_prompt(language: str) -> str:
    """Create a system prompt for the given language."""
    template = load_prompt("system_tutor", language)
    return format_prompt(template, {"language": language})
