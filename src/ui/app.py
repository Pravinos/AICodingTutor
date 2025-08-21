"""
Streamlit UI for the Interactive Coding Tutor.
"""

import streamlit as st
import logging
import sys
import os
import yaml
import json
from typing import Dict, List, Any
import traceback
from datetime import datetime

# Add src to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..')
root_dir = os.path.join(src_dir, '..')
sys.path.append(src_dir)

try:
    from business_logic.core import (
        handle_query, explain_concept, generate_exercise, evaluate_user_code,
        get_supported_languages, get_topics, validate_inputs, create_chat_message,
        TutorError
    )
    from business_logic.export_manager import ExportManager
    from helpers.lm_studio import get_available_models, test_connection, LMStudioError
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG = {
    "llm_endpoint": "http://localhost:1234",
    "supported_languages": ["python", "javascript", "java", "cpp", "csharp"],
    "default_model": "phi-3-mini"
}

# Page configuration
st.set_page_config(
    page_title="Interactive Coding Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=30)
def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = os.path.join(root_dir, "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return DEFAULT_CONFIG


def get_chat_history_file_path() -> str:
    """Get the path for the chat history file"""
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "chat_history.json")
def save_chat_history():
    """Persist current chat history and sessions to disk."""
    try:
        # Clean up sessions before saving
        cleanup_chat_sessions()
        
        history_file = get_chat_history_file_path()
        
        # If we're in an existing session with messages, update that session in the sessions list
        if st.session_state.current_session_id and st.session_state.chat_history:
            session_updated = False
            # Find and update the existing session
            for i, session in enumerate(st.session_state.chat_sessions):
                if session["id"] == st.session_state.current_session_id:
                    # Update the existing session
                    st.session_state.chat_sessions[i].update({
                        "messages": st.session_state.chat_history.copy(),
                        "message_count": len(st.session_state.chat_history),
                        "language": st.session_state.selected_language,
                        "model": st.session_state.selected_model
                    })
                    session_updated = True
                    break
            
            # If session doesn't exist in sessions list but we have messages, add it
            if not session_updated and len(st.session_state.chat_history) > 0:
                new_session = {
                    "id": st.session_state.current_session_id,
                    "created_at": datetime.now().isoformat(),
                    "language": st.session_state.selected_language,
                    "model": st.session_state.selected_model,
                    "messages": st.session_state.chat_history.copy(),
                    "message_count": len(st.session_state.chat_history)
                }
                st.session_state.chat_sessions.append(new_session)
                
                # Clean up again after adding
                cleanup_chat_sessions()
        
        history_data = {
            "last_updated": datetime.now().isoformat(),
            "chat_history": st.session_state.chat_history,
            "chat_sessions": st.session_state.chat_sessions,
            "current_session_id": st.session_state.current_session_id,
            "selected_language": st.session_state.selected_language,
            "selected_model": st.session_state.selected_model
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")


def load_chat_history():
    """Load chat history and sessions from disk if available."""
    try:
        history_file = get_chat_history_file_path()
        
        if not os.path.exists(history_file):
            return
            
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
            
        # Validate and load chat history
        if "chat_history" in history_data and isinstance(history_data["chat_history"], list):
            st.session_state.chat_history = history_data["chat_history"]
            
        # Validate and load chat sessions
        if "chat_sessions" in history_data and isinstance(history_data["chat_sessions"], list):
            # Validate each session has required fields
            valid_sessions = []
            for session in history_data["chat_sessions"]:
                if (isinstance(session, dict) and 
                    "id" in session and 
                    "messages" in session and 
                    isinstance(session["messages"], list)):
                    valid_sessions.append(session)
            st.session_state.chat_sessions = valid_sessions
            
        # Load current session ID
        if "current_session_id" in history_data:
            st.session_state.current_session_id = history_data["current_session_id"]
            
        # Optionally restore last used settings
        if ("selected_language" in history_data and 
            not st.session_state.get("language_manually_changed") and
            history_data["selected_language"] in get_supported_languages()):
            st.session_state.selected_language = history_data["selected_language"]
            
        if ("selected_model" in history_data and 
            not st.session_state.get("model_manually_changed")):
            st.session_state.selected_model = history_data["selected_model"]
            
        logger.info(f"Loaded {len(st.session_state.chat_history)} chat messages and {len(st.session_state.chat_sessions)} sessions from history")
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse chat history JSON: {e}")
        # Reset to empty state if file is corrupted
        st.session_state.chat_history = []
        st.session_state.chat_sessions = []
    except Exception as e:
        logger.warning(f"Failed to load chat history: {e}")


def start_new_chat():
    """Archive current chat (if any) and reset state for a new session."""
    if st.session_state.chat_history and st.session_state.current_session_id:
        session = {
            "id": st.session_state.current_session_id,
            "created_at": datetime.now().isoformat(),
            "language": st.session_state.selected_language,
            "model": st.session_state.selected_model,
            "messages": st.session_state.chat_history.copy(),
            "message_count": len(st.session_state.chat_history)
        }
        
        session_exists = False
        for i, existing_session in enumerate(st.session_state.chat_sessions):
            if existing_session["id"] == session["id"]:
                st.session_state.chat_sessions[i] = session
                session_exists = True
                break
        
        if not session_exists:
            st.session_state.chat_sessions.append(session)
            if len(st.session_state.chat_sessions) > 5:
                st.session_state.chat_sessions.pop(0)
    
    st.session_state.chat_history = []
    st.session_state.current_session_id = None
    st.session_state.current_exercise = None
    st.session_state.current_explanation = None
    
    save_chat_history()


def load_chat_session(session_id: str):
    """Switch to a stored chat session by id."""
    # Find the session to load
    session_to_load = None
    for session in st.session_state.chat_sessions:
        if session["id"] == session_id:
            session_to_load = session
            break
    
    if not session_to_load:
        logger.warning(f"Session {session_id} not found")
        return
    
    # Save current chat as a new session only if it has messages and is different
    if (st.session_state.chat_history and 
        st.session_state.current_session_id and
        st.session_state.current_session_id != session_id):
        
        # Check if current session already exists in sessions list
        current_exists = any(s["id"] == st.session_state.current_session_id for s in st.session_state.chat_sessions)
        
        if not current_exists:
            current_session = {
                "id": st.session_state.current_session_id,
                "created_at": datetime.now().isoformat(),
                "language": st.session_state.selected_language,
                "model": st.session_state.selected_model,
                "messages": st.session_state.chat_history.copy(),
                "message_count": len(st.session_state.chat_history)
            }
            
            # Add to sessions list (keep only latest 5)
            st.session_state.chat_sessions.append(current_session)
            if len(st.session_state.chat_sessions) > 5:
                st.session_state.chat_sessions.pop(0)  # Remove oldest session
    
    # Load selected session
    st.session_state.chat_history = session_to_load["messages"].copy()
    st.session_state.current_session_id = session_to_load["id"]
    st.session_state.selected_language = session_to_load.get("language", "python")
    st.session_state.selected_model = session_to_load.get("model", None)
    
    # Clear current exercise and explanation when switching sessions
    st.session_state.current_exercise = None
    st.session_state.current_explanation = None
    
    # Automatically switch to chat tab when loading a session
    st.session_state.active_tab = "💬 Chat"
    
    save_chat_history()


def clear_chat_history_file():
    """Delete persisted chat history file if it exists."""
    try:
        history_file = get_chat_history_file_path()
        if os.path.exists(history_file):
            os.remove(history_file)
    except Exception as e:
        logger.warning(f"Failed to clear chat history file: {e}")


def cleanup_chat_sessions():
    """Deduplicate and trim stored sessions (keep latest 5)."""
    if not st.session_state.chat_sessions:
        return
    
    # Remove duplicates based on session ID
    seen_ids = set()
    unique_sessions = []
    
    for session in st.session_state.chat_sessions:
        session_id = session.get("id")
        if session_id and session_id not in seen_ids:
            # Validate session has required fields
            if (isinstance(session.get("messages"), list) and
                session.get("message_count", 0) > 0):
                seen_ids.add(session_id)
                unique_sessions.append(session)
    
    # Sort by creation date, newest first
    unique_sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    # Keep only latest 5
    st.session_state.chat_sessions = unique_sessions[:5]


def initialize_session_state():
    """Ensure required session_state keys exist then lazy-load history."""
    defaults = {
        "chat_history": [],
        "chat_sessions": [],  # Store multiple chat sessions
        "current_session_id": None,
        "selected_language": "python",
        "selected_model": None,
        "current_exercise": None,
        "current_explanation": None,
        "history_loaded": False,
        "language_manually_changed": False,
        "model_manually_changed": False,
        "export_manager": ExportManager(),
        "active_tab": "💬 Chat",
        "is_loading": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if not st.session_state.history_loaded:
        load_chat_history()
        st.session_state.history_loaded = True


@st.cache_data(ttl=30)
def check_lm_studio_connection(endpoint: str, config: Dict[str, Any]) -> bool:
    """Return True if LM Studio endpoint responds within timeout."""
    timeout = config.get("app_settings", {}).get("connection_check_interval", 5)
    return test_connection(endpoint=endpoint, timeout=timeout)


@st.cache_data(ttl=60)
def get_models(endpoint: str, config: Dict[str, Any]) -> List[str]:
    """Return list of available chat models from LM Studio or fallback."""
    try:
        timeout = config.get("generation_settings", {}).get("timeout", 10)
        return get_available_models(endpoint=endpoint, timeout=timeout)
    except LMStudioError:
        return ["No models available"]


def render_sidebar(config: Dict[str, Any]) -> bool:
    """Render sidebar; return False if mandatory resources missing."""
    with st.sidebar:
        st.title("🚀 Your AI Coding Buddy")
        st.markdown("---")
        
        endpoint = config.get("llm_endpoint", "http://localhost:1234")
        
        # Connection status
        if not _render_connection_status(endpoint, config):
            return False
        
        # Model selection
        if not _render_model_selection(endpoint, config):
            return False
        
        # Language selection
        _render_language_selection()
        
        # Chat history
        _render_chat_history_sidebar()
        
        # Action buttons
        _render_sidebar_buttons()
        
        return True


def _render_connection_status(endpoint: str, config: Dict[str, Any]) -> bool:
    """Show connection status; return connectivity boolean."""
    is_connected = check_lm_studio_connection(endpoint, config)
    
    if is_connected:
        st.success("✅ LM Studio Connected")
        return True
    else:
        st.error("❌ LM Studio Disconnected")
        st.warning("Please start LM Studio and load a model")
        return False


def _render_model_selection(endpoint: str, config: Dict[str, Any]) -> bool:
    """Render model selector; persist manual change flags; return success."""
    st.subheader("🤖 AI Model")
    available_models = get_models(endpoint, config)
    
    if not available_models or available_models[0] == "No models available":
        st.error("No chat models available in LM Studio")
        st.warning("Make sure to load a chat/instruct model")
        return False
    
    model_index = 0
    if st.session_state.selected_model in available_models:
        model_index = available_models.index(st.session_state.selected_model)
    
    old_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "Choose a model:",
        available_models,
        index=model_index,
        help="Select the AI model to use for tutoring. Keep in mind that depending on your system's capabilities, some models may perform better than others. Responses usually take 20-40 seconds to generate."
    )
    
    if (old_model != st.session_state.selected_model and 
        old_model is not None and 
        st.session_state.get("history_loaded", False)):
        st.session_state.model_manually_changed = True
        save_chat_history()
    
    if len(available_models) < 3:
        st.info("ℹ️ Embedding models filtered out")
    
    return True


def _render_language_selection():
    """Render programming language selector and track manual changes."""
    st.subheader("💻 Programming Language")
    languages = get_supported_languages()
    lang_codes = list(languages.keys())
    
    lang_index = 0
    if st.session_state.selected_language in lang_codes:
        lang_index = lang_codes.index(st.session_state.selected_language)
    
    old_language = st.session_state.selected_language
    st.session_state.selected_language = st.selectbox(
        "Choose a language:",
        lang_codes,
        format_func=lambda x: languages[x],
        index=lang_index,
        help="Select the programming language to learn"
    )
    
    if (old_language != st.session_state.selected_language and 
        old_language != "python" and
        st.session_state.get("history_loaded", False)):
        st.session_state.language_manually_changed = True
        save_chat_history()


def _render_chat_history_sidebar():
    """Render stored sessions and export control without extra spacing."""
    st.markdown("---")
    st.subheader("💬 Chat History")

    had_sessions = bool(st.session_state.chat_sessions)

    if had_sessions:
        st.markdown("**Previous Chats:**")
        # Show latest 5 sessions, newest first
        for idx, session in enumerate(sorted(st.session_state.chat_sessions, key=lambda x: x.get("created_at", ""), reverse=True)[:5]):
            try:
                session_date = datetime.fromisoformat(session["created_at"]).strftime("%m/%d %H:%M")
            except Exception:
                session_date = "Unknown"

            language = session.get("language", "unknown").title()
            snippet = ""
            for m in session.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    snippet = m.get("content", "")
                    break
            if not snippet and session.get("messages"):
                snippet = session["messages"][0].get("content", "")
                
            if snippet:
                snippet = snippet.split("```", 1)[0]
                snippet = " ".join(snippet.strip().split())
                max_len = 40
                if len(snippet) > max_len:
                    snippet = snippet[:max_len - 3].rstrip() + "..."
            else:
                snippet = "(no content)"
            label = f"💬 {session_date} - {language} - {snippet}"
            key = f"session_btn_{idx}_{session['id']}"
            is_current = session["id"] == st.session_state.current_session_id

            if st.button(label, key=key, use_container_width=True,
                         type="primary" if is_current else "secondary",
                         help="Currently active session" if is_current else "Click to switch to this session"):
                load_chat_session(session["id"])
                st.rerun()

    if (st.session_state.chat_history and st.session_state.current_session_id and
            any(s["id"] == st.session_state.current_session_id for s in st.session_state.chat_sessions)):
        if not had_sessions:
            st.markdown("---")
        st.subheader("📄 Export Current Chat")
        _export_chat_history()


def _export_chat_history():
    """Offer PDF download for current session via ExportManager."""
    export_manager = st.session_state.export_manager
    language = get_supported_languages()[st.session_state.selected_language]
    model = st.session_state.selected_model
    chat_history = st.session_state.chat_history
    
    if not chat_history:
        st.warning("No chat history to export!")
        return
    
    try:
        pdf_data = export_manager.export_to_pdf(chat_history, language, model, content_type="chat")
        filename_pdf = export_manager.get_export_filename('pdf')
        
        st.download_button(
            label="📚 Download PDF",
            data=pdf_data,
            file_name=filename_pdf,
            mime="application/pdf",
            use_container_width=True,
            type="secondary"
        )
    except ImportError:
        st.error("📚 PDF export requires reportlab library")
        st.code("pip install reportlab")
    except Exception as e:
        st.error(f"PDF export failed: {str(e)}")
def _render_sidebar_buttons():
    """Render clear all action for chat management."""
    st.markdown("---")
    
    if st.button("🗑️ Clear All", use_container_width=True, type="secondary", help="Clear all chat history"):
        st.session_state.chat_history = []
        st.session_state.chat_sessions = []
        st.session_state.current_exercise = None
        st.session_state.current_explanation = None
        st.session_state.current_session_id = None
        clear_chat_history_file()
        st.rerun()
def handle_chat_input(user_input: str, config: Dict[str, Any]) -> str:
    """Send user input to model pipeline and return assistant reply."""
    try:
        validate_inputs(user_input, st.session_state.selected_language, st.session_state.selected_model)
        
        endpoint = config.get("llm_endpoint", "http://localhost:1234")
        
        return handle_query(
            user_input=user_input,
            language=st.session_state.selected_language,
            model=st.session_state.selected_model,
            history=st.session_state.chat_history[-10:],
            endpoint=endpoint,
            config=config
        )
        
    except (TutorError, LMStudioError) as e:
        return f"Sorry, I encountered an error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return f"An unexpected error occurred: {e}"


def _display_chat_history():
    """Stream messages to chat area preserving roles."""
    for message in st.session_state.chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        with st.chat_message(role):
            if role == "user":
                st.write(content)
            else:
                st.markdown(content)


def _process_user_message(user_input: str):
    """Append user message, generate assistant reply, persist state."""
    st.session_state.is_loading = True
    user_msg = create_chat_message("user", user_input)
    st.session_state.chat_history.append(user_msg)
    
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                config = load_config()
                response = handle_chat_input(user_input, config)
                st.markdown(response)
                
                assistant_msg = create_chat_message("assistant", response)
                st.session_state.chat_history.append(assistant_msg)
                
                save_chat_history()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
            finally:
                st.session_state.is_loading = False


def render_chat_interface():
    """Chat tab UI including history display and input box."""
    language_name = get_supported_languages()[st.session_state.selected_language]
    
    # Create centered container for title and buttons with more space
    col1, col2, col3 = st.columns([0.7, 3, 0.7])
    with col2:
        # Check if there's an active conversation
        has_active_chat = bool(st.session_state.chat_history and st.session_state.current_session_id)
        
        if has_active_chat:
            # Create two separate containers: one for text (left) and one for buttons (right)
            text_col, buttons_col = st.columns([1.8, 1])
            
            with text_col:
                st.markdown(f"<h2 style='margin: 0; text-align: left;'>💬 Chat with your {language_name} tutor</h2>", 
                            unsafe_allow_html=True)
            
            with buttons_col:
                # Create a container for buttons with custom CSS to align them right with small gap
                st.markdown("""
                    <style>
                    .button-container {
                        display: flex;
                        justify-content: flex-end;
                        align-items: center;
                        gap: 8px;
                        margin-top: -5px;
                    }
                    .button-container > div {
                        display: flex;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                # Use columns for the two buttons with minimal space
                btn_col1, btn_col2 = st.columns([1, 1])
                
                with btn_col1:
                    if st.button("🆕 New Chat", help="Start a new conversation", key="new_chat_btn", use_container_width=True):
                        start_new_chat()
                        st.rerun()
                
                with btn_col2:
                    if st.button("🗑️ Clear Current", help="Clear current chat only", key="clear_current_btn", use_container_width=True):
                        # Clear current chat but keep it in sessions if it has messages
                        if st.session_state.chat_history and st.session_state.current_session_id:
                            save_chat_history()
                        
                        st.session_state.chat_history = []
                        st.session_state.current_exercise = None
                        st.session_state.current_explanation = None
                        st.session_state.current_session_id = None
                        save_chat_history()
                        st.rerun()
        else:
            # Simpler layout for no active chat - title left, single button right
            text_col, button_col = st.columns([2.5, 1])
            
            with text_col:
                st.markdown(f"<h2 style='margin: 0; text-align: left;'>💬 Chat with your {language_name} tutor</h2>", 
                            unsafe_allow_html=True)
            
            with button_col:
                # Add the same CSS styling for proper alignment
                st.markdown("""
                    <style>
                    .single-button-container {
                        display: flex;
                        justify-content: flex-end;
                        align-items: center;
                        margin-top: -5px;
                    }
                    .single-button-container > div {
                        display: flex;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                if st.button("🆕 New Chat", help="Start a new conversation", key="new_chat_btn", use_container_width=True):
                    start_new_chat()
                    st.rerun()
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display chat history
    _display_chat_history()
    
    # Chat input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_input = st.chat_input("Ask me anything about programming...")
    
    if user_input:
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        _process_user_message(user_input)


def _generate_explanation(selected_topic: str) -> None:
    """Generate and store explanation text for a concept."""
    st.session_state.is_loading = True
    col1_spinner, col2_spinner, col3_spinner = st.columns([1, 1, 1])
    with col2_spinner:
        with st.spinner("Generating..."):
            try:
                config = load_config()
                endpoint = config.get("llm_endpoint", "http://localhost:1234")
                
                explanation = explain_concept(
                    concept=selected_topic.replace('_', ' '),
                    language=st.session_state.selected_language,
                    model=st.session_state.selected_model,
                    endpoint=endpoint,
                    config=config
                )
                st.session_state.current_explanation = explanation
                
            except (TutorError, LMStudioError) as e:
                st.error(f"Failed to generate explanation: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Unexpected error in concept explainer: {e}")
            finally:
                st.session_state.is_loading = False


def _display_explanation():
    """Show current explanation with export and clear controls if present."""
    if st.session_state.get('current_explanation'):
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Current Explanation:</h3>", unsafe_allow_html=True)
        st.markdown(st.session_state.current_explanation)
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_button = st.button("📄 Export PDF", use_container_width=True, type="secondary")
        
        with col2:
            clear_button = st.button("🗑️ Clear Explanation", use_container_width=True, type="secondary")
        
        if export_button:
            _export_explanation_to_pdf()
        
        if clear_button:
            st.session_state.current_explanation = None
            st.rerun()


def _export_explanation_to_pdf():
    """Export current explanation to PDF using the export manager."""
    if not st.session_state.get('current_explanation'):
        st.warning("No explanation to export!")
        return
    
    try:
        export_manager = st.session_state.export_manager
        language = get_supported_languages()[st.session_state.selected_language]
        model = st.session_state.selected_model
        
        # Create a mock chat history format for the explanation
        explanation_history = [
            {
                "role": "user",
                "content": f"Please explain this concept in {language}",
                "timestamp": datetime.now().isoformat()
            },
            {
                "role": "assistant", 
                "content": st.session_state.current_explanation,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        pdf_data = export_manager.export_to_pdf(explanation_history, language, model, content_type="explanation")
        
        # Create custom filename for explanation export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_pdf = f"coding_tutor_explanation_{timestamp}.pdf"
        
        st.download_button(
            label="📚 Download PDF",
            data=pdf_data,
            file_name=filename_pdf,
            mime="application/pdf",
            use_container_width=True,
            type="secondary"
        )
    except ImportError:
        st.error("📚 PDF export requires reportlab library")
        st.code("pip install reportlab")
    except Exception as e:
        st.error(f"PDF export failed: {str(e)}")


def render_concept_explainer():
    """Concept explainer tab UI."""
    st.markdown("<h2 style='text-align: center;'>📖 Concept Explainer</h2>", unsafe_allow_html=True)
    
    topics = get_topics(st.session_state.selected_language)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        selected_topic = st.selectbox(
            "Choose a concept to learn about:",
            topics,
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select a programming concept for detailed explanation. Responses take 20-30 seconds to generate please don't switch tabs."
        )
        
        explain_button = st.button("📚 Explain Concept", use_container_width=True, type="secondary")
        
        if explain_button and selected_topic:
            st.session_state.active_tab = "📖 Learn Concepts"
            _generate_explanation(selected_topic)
    _display_explanation()


def _generate_exercise(selected_topic: str) -> None:
    """Generate and store practice exercise."""
    st.session_state.is_loading = True
    col1_spinner, col2_spinner, col3_spinner = st.columns([1, 1, 1])
    with col2_spinner:
        with st.spinner("Generating..."):
            try:
                config = load_config()
                endpoint = config.get("llm_endpoint", "http://localhost:1234")
                
                exercise = generate_exercise(
                    topic=selected_topic,
                    language=st.session_state.selected_language,
                    model=st.session_state.selected_model,
                    endpoint=endpoint,
                    config=config
                )
                st.session_state.current_exercise = exercise
                
            except (TutorError, LMStudioError) as e:
                st.error(f"Failed to generate exercise: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Unexpected error in exercise generator: {e}")
            finally:
                st.session_state.is_loading = False


def _evaluate_code(user_code: str) -> None:
    """Evaluate submitted solution against current exercise."""
    st.session_state.is_loading = True
    with st.spinner("Evaluating your code..."):
        try:
            config = load_config()
            endpoint = config.get("llm_endpoint", "http://localhost:1234")
            
            feedback = evaluate_user_code(
                code=user_code,
                task=st.session_state.current_exercise["task_description"],
                language=st.session_state.selected_language,
                model=st.session_state.selected_model,
                endpoint=endpoint,
                config=config
            )
            
            st.markdown("<h3 style='text-align: center;'>Feedback:</h3>", unsafe_allow_html=True)
            st.markdown(feedback)
            
        except (TutorError, LMStudioError) as e:
            st.error(f"Failed to evaluate code: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"Unexpected error in code evaluation: {e}")
        finally:
            st.session_state.is_loading = False


def _display_exercise():
    """Render exercise, code input, and evaluation controls."""
    if not st.session_state.current_exercise:
        return
        
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Current Exercise:</h3>", unsafe_allow_html=True)
    st.markdown(st.session_state.current_exercise["exercise_text"])
    
    st.markdown("<h3 style='text-align: center;'>Your Solution:</h3>", unsafe_allow_html=True)
    language_name = get_supported_languages()[st.session_state.selected_language]
    user_code = st.text_area(
        "Write your code here:",
        height=200,
        placeholder=f"# Write your {language_name} code here...",
        help="Enter your solution to the exercise above"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        submit_button = st.button("✅ Submit Code", use_container_width=True, type="secondary")
    
    with col2:
        clear_button = st.button("🗑️ Clear Exercise", use_container_width=True, type="secondary")
    
    if submit_button and user_code.strip():
        _evaluate_code(user_code)
    elif submit_button:
        st.warning("Please enter some code before submitting.")
    
    if clear_button:
        st.session_state.current_exercise = None
        st.rerun()


def render_exercise_generator():
    """Exercise generator / evaluator tab UI."""
    st.markdown("<h2 style='text-align: center;'>🏋️ Practice Exercises</h2>", unsafe_allow_html=True)
    
    topics = get_topics(st.session_state.selected_language)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        selected_topic = st.selectbox(
            "Choose a topic for practice:",
            topics,
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select a topic to generate a practice exercise. Responses take 20-30 seconds to generate please don't switch tabs."
        )
        
        generate_button = st.button("🎯 Generate Exercise", use_container_width=True, type="secondary")
    
    if generate_button and selected_topic:
        st.session_state.active_tab = "🏋️ Practice"
        _generate_exercise(selected_topic)
    _display_exercise()


def _render_modern_tabs(tab_options, current_tab_key):
    """Render modern clickable tab navigation that's responsive and centered."""
    
    # Create responsive columns that adjust based on sidebar state
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create three columns for the tabs
        tab_col1, tab_col2, tab_col3 = st.columns(3)
        
        tab_mapping = {
            "chat": "💬 Chat",
            "concepts": "📖 Learn Concepts", 
            "practice": "🏋️ Practice"
        }
        
        for i, tab in enumerate(tab_options):
            with [tab_col1, tab_col2, tab_col3][i]:
                # Determine if this tab is active
                is_active = tab["key"] == current_tab_key
                
                # Create the button with dynamic styling
                button_type = "primary" if is_active else "secondary"
                button_key = f"tab_{tab['key']}"
                
                # Create button with icon and label
                button_text = f"{tab['icon']} {tab['label'].split(' ', 1)[1] if ' ' in tab['label'] else tab['label']}"
                
                if st.button(
                    button_text,
                    key=button_key,
                    type=button_type,
                    use_container_width=True,
                    help=f"Switch to {tab['label']}"
                ):
                    st.session_state.active_tab = tab_mapping[tab["key"]]
                    st.rerun()


def _apply_custom_css():
    """Inject modern CSS for tab navigation and responsive design."""
    st.markdown(
        """
        <style>
        /* General button styling - keep normal size for most buttons */
        div[data-testid="stButton"] > button {
            border-radius: 15px !important;
            padding: 15px 10px !important;
            margin: 5px !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            min-height: 70px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
            border: none !important;
            width: calc(100% - 10px) !important;
        }
        
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Primary buttons (active tabs) */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #ff4b4b 0%, #d32f2f 100%) !important;
            color: white !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 25px rgba(255, 75, 75, 0.3) !important;
        }
        
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%) !important;
            transform: translateY(-3px) !important;
        }
        
        /* Secondary buttons (inactive tabs) */
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            color: #495057 !important;
            border: 1px solid #dee2e6 !important;
        }
        
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%) !important;
            color: #212529 !important;
        }
        
        /* Responsive design for smaller screens */
        @media (max-width: 768px) {
            div[data-testid="stButton"] > button {
                padding: 12px 8px !important;
                min-height: 60px !important;
                font-size: 12px !important;
            }
        }
        
        /* Adjust spacing for main content */
        .main .block-container {
            padding-top: 1rem;
        }
        
        /* Reduce sidebar top padding */
        .css-1d391kg {
            padding-top: 1rem !important;
        }
        
        /* Target sidebar title specifically */
        section[data-testid="stSidebar"] .element-container:first-child h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Ensure selectbox and button alignment in concept/practice tabs */
        div[data-testid="stSelectbox"] > div > div {
            width: 100% !important;
        }
        
        /* Force exact alignment between selectbox and buttons */
        div[data-testid="stSelectbox"] {
            margin: 5px !important;
        }
        
        div[data-testid="stSelectbox"] > div {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Dark mode adjustments */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stButton"] > button[kind="secondary"] {
                background: linear-gradient(135deg, #343a40 0%, #495057 100%) !important;
                color: #f8f9fa !important;
                border: 1px solid #6c757d !important;
            }
            
            div[data-testid="stButton"] > button[kind="secondary"]:hover {
                background: linear-gradient(135deg, #495057 0%, #6c757d 100%) !important;
                color: white !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def _render_footer():
    """Footer banner with attribution."""
    st.markdown(
        """
        <div style='
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: var(--background-color);
            text-align: right;
            padding: 5px 15px;
            border-top: 1px solid var(--secondary-background-color);
            z-index: 999;
            font-size: 0.75em;
            opacity: 0.8;
        '>
        Developed by tpravinos | Built with ❤️ using Streamlit and LM Studio
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Entry point orchestrating layout and tabs."""
    initialize_session_state()
    
    # Apply styling immediately before any UI elements
    _apply_custom_css()
    
    config = load_config()
    
    # Render sidebar
    if not render_sidebar(config):
        st.error("Please ensure LM Studio is running with models loaded.")
        st.stop()
    
    # Modern tab navigation using clickable buttons
    tab_options = [
        {"key": "chat", "label": "💬 Chat", "icon": "💬"},
        {"key": "concepts", "label": "📖 Learn Concepts", "icon": "📖"}, 
        {"key": "practice", "label": "🏋️ Practice", "icon": "🏋️"}
    ]
    
    # Map session state to tab keys
    tab_key_mapping = {
        "💬 Chat": "chat",
        "📖 Learn Concepts": "concepts", 
        "🏋️ Practice": "practice"
    }
    
    current_tab_key = tab_key_mapping.get(st.session_state.active_tab, "chat")
    
    # Create modern tab navigation
    _render_modern_tabs(tab_options, current_tab_key)
    
    # Add some spacing
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # Render content based on selected tab
    if st.session_state.active_tab == "💬 Chat":
        render_chat_interface()
    elif st.session_state.active_tab == "📖 Learn Concepts":
        render_concept_explainer()
    elif st.session_state.active_tab == "🏋️ Practice":
        render_exercise_generator()
    
    # Render footer
    _render_footer()


if __name__ == "__main__":
    main()
