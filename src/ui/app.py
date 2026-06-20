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
    page_title="DevTutor AI",
    page_icon="🖥️",
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
    timeout = config.get("generation_settings", {}).get("timeout", 10)
    return test_connection(endpoint=endpoint, timeout=min(timeout, 10))


@st.cache_data(ttl=60)
def get_models(endpoint: str, config: Dict[str, Any]) -> List[str]:
    """Return list of available chat models from LM Studio or fallback."""
    try:
        timeout = config.get("generation_settings", {}).get("timeout", 10)
        return get_available_models(endpoint=endpoint, timeout=timeout)
    except LMStudioError:
        return ["No models available"]


def _sidebar_section(icon: str, title: str) -> None:
    """Render a styled sidebar section heading using native Streamlit markup."""
    st.subheader(f"{icon} {title}")


def _render_sidebar_header(is_connected: bool) -> None:
    """Brand + connection status in one block to avoid Streamlit block gaps."""
    if is_connected:
        status_html = """
        <div class="dt-status connected">
          <span class="dt-dot"></span>
          <span class="dt-status-label">LM Studio connected</span>
          <span class="dt-status-badge">Local · Private</span>
        </div>
        """
    else:
        status_html = """
        <div class="dt-status offline">
          <span class="dt-dot"></span>
          <span class="dt-status-label">LM Studio offline</span>
        </div>
        <p class="dt-status-hint">Start LM Studio and load a model to continue.</p>
        """

    st.markdown(
        f"""
        <div class="dt-sidebar-top">
          <div class="dt-sidebar-brand">
            <div class="dt-sidebar-brand-row">
              <span class="dt-sidebar-brand-icon">🖥️</span>
              <div class="dt-sidebar-brand-text">
                <span class="dt-sidebar-brand-title">DevTutor AI</span>
                <span class="dt-sidebar-brand-tagline">Coding tutor for beginners</span>
              </div>
            </div>
          </div>
          {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: Dict[str, Any]) -> bool:
    """Render sidebar; return False if mandatory resources missing."""
    with st.sidebar:
        endpoint = config.get("llm_endpoint", "http://localhost:1234")
        is_connected = check_lm_studio_connection(endpoint, config)
        _render_sidebar_header(is_connected)

        if not is_connected:
            return False

        if not _render_model_selection(endpoint, config):
            return False

        _render_language_selection()
        _render_chat_history_sidebar()
        _render_sidebar_buttons()

        return True


def _render_model_selection(endpoint: str, config: Dict[str, Any]) -> bool:
    """Render model selector; persist manual change flags; return success."""
    _sidebar_section("🤖", "AI Model")
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
        st.caption("ℹ️ Embedding models are filtered out automatically.")
    
    return True


def _render_language_selection():
    """Render programming language selector."""
    _sidebar_section("💻", "Programming Language")
    languages = get_supported_languages()
    lang_codes = list(languages.keys())

    old_language = st.session_state.selected_language
    st.session_state.selected_language = st.selectbox(
        "Choose a language:",
        lang_codes,
        format_func=lambda x: languages[x],
        index=lang_codes.index(st.session_state.selected_language) if st.session_state.selected_language in lang_codes else 0,
        help="Select the programming language to learn",
    )

    if (old_language != st.session_state.selected_language and
            st.session_state.get("history_loaded", False)):
        st.session_state.language_manually_changed = True
        save_chat_history()


def _render_chat_history_sidebar():
    """Render stored sessions and export control."""
    _sidebar_section("🗂️", "Recent Sessions")

    had_sessions = bool(st.session_state.chat_sessions)

    if had_sessions:
        for idx, session in enumerate(sorted(st.session_state.chat_sessions, key=lambda x: x.get("created_at", ""), reverse=True)[:5]):
            try:
                session_date = datetime.fromisoformat(session["created_at"]).strftime("%b %d, %H:%M")
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
                max_len = 60
                if len(snippet) > max_len:
                    snippet = snippet[:max_len - 3].rstrip() + "..."

            label = f"{session_date} · {language}"
            help_text = snippet or "Empty session"
            key = f"session_btn_{idx}_{session['id']}"
            is_current = session["id"] == st.session_state.current_session_id

            if st.button(label, key=key, use_container_width=True,
                         type="primary" if is_current else "secondary",
                         help=help_text):
                load_chat_session(session["id"])
                st.rerun()
    else:
        st.caption("No saved sessions yet — start chatting to see them here.")

    if (st.session_state.chat_history and st.session_state.current_session_id and
            any(s["id"] == st.session_state.current_session_id for s in st.session_state.chat_sessions)):
        _sidebar_section("📤", "Export Chat")
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
    st.markdown('<div class="dt-sidebar-divider"></div>', unsafe_allow_html=True)

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
    
    # Simplified header with title and buttons in a clean flex row
    has_active_chat = bool(st.session_state.chat_history and st.session_state.current_session_id)
    
    if has_active_chat:
        # Title + two action buttons
        title_col, btn1_col, btn2_col = st.columns([2.5, 1, 1], vertical_alignment="center")
        
        with title_col:
            st.markdown(f"<h3 style='margin: 0; white-space: nowrap; font-size: 18px;'>🖥️ Chat with your {language_name} tutor</h3>", 
                        unsafe_allow_html=True)
        
        with btn1_col:
            if st.button("🆕 New Chat", help="Start a new conversation", key="new_chat_btn", use_container_width=True):
                start_new_chat()
                st.rerun()
        
        with btn2_col:
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
        # Title + single new chat button
        title_col, btn_col = st.columns([3, 1], vertical_alignment="center")
        
        with title_col:
            st.markdown(f"<h3 style='margin: 0; white-space: nowrap; font-size: 18px;'>🖥️ Chat with your {language_name} tutor</h3>", 
                        unsafe_allow_html=True)
        
        with btn_col:
            if st.button("🆕 New Chat", help="Start a new conversation", key="new_chat_btn", use_container_width=True):
                start_new_chat()
                st.rerun()
    
    # Welcome banner on first load
    _render_welcome_banner()
    
    # Display chat history
    _display_chat_history()
    
    # Chat input — st.chat_input already docks full-width at the bottom of the
    # page; wrapping it in a narrow centered column (the old behavior) squeezed
    # it to a third of the screen for no reason.
    user_input = st.chat_input("Ask me anything about programming...")
    
    if user_input:
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        _process_user_message(user_input)


def _generate_explanation(selected_topic: str) -> None:
    """Generate and store explanation text for a concept."""
    st.session_state.is_loading = True
    with st.spinner("Generating explanation..."):
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
            st.session_state.pop("explanation_pdf", None)
            st.session_state.pop("explanation_pdf_filename", None)
            
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
            if st.button("📄 Export PDF", use_container_width=True, type="secondary", key="export_explanation_btn"):
                _prepare_explanation_pdf_download()

        with col2:
            if st.button("🗑️ Clear Explanation", use_container_width=True, type="secondary", key="clear_explanation_btn"):
                st.session_state.current_explanation = None
                st.session_state.pop("explanation_pdf", None)
                st.session_state.pop("explanation_pdf_filename", None)
                st.rerun()

        if st.session_state.get("explanation_pdf"):
            st.download_button(
                label="📚 Download PDF",
                data=st.session_state.explanation_pdf,
                file_name=st.session_state.get("explanation_pdf_filename", "explanation.pdf"),
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="download_explanation_pdf",
            )


def _prepare_explanation_pdf_download():
    """Build explanation PDF bytes and store them for the download button."""
    if not st.session_state.get("current_explanation"):
        st.warning("No explanation to export!")
        return

    try:
        export_manager = st.session_state.export_manager
        language = get_supported_languages()[st.session_state.selected_language]
        model = st.session_state.selected_model

        explanation_history = [
            {
                "role": "user",
                "content": f"Please explain this concept in {language}",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "role": "assistant",
                "content": st.session_state.current_explanation,
                "timestamp": datetime.now().isoformat(),
            },
        ]

        st.session_state.explanation_pdf = export_manager.export_to_pdf(
            explanation_history, language, model, content_type="explanation"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.explanation_pdf_filename = f"coding_tutor_explanation_{timestamp}.pdf"
    except ImportError:
        st.error("📚 PDF export requires reportlab library")
        st.code("pip install reportlab")
    except Exception as e:
        st.error(f"PDF export failed: {str(e)}")


def render_concept_explainer():
    """Concept explainer tab UI."""
    st.markdown("<h2 style='text-align: center;'>📖 Learn Concepts</h2>", unsafe_allow_html=True)
    
    topics = get_topics(st.session_state.selected_language)
    
    col1, col2, col3 = st.columns([1, 3, 1])
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
    with st.spinner("Generating exercise..."):
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
    
    col1, col2, col3 = st.columns([1, 3, 1])
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


def _render_modern_tabs():
    """Render native Streamlit tabs with content inside each tab."""
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📖 Learn Concepts", "🏋️ Practice"])

    with tab1:
        render_chat_interface()

    with tab2:
        render_concept_explainer()

    with tab3:
        render_exercise_generator()


def _apply_custom_css():
    """Inject scoped CSS for DevTutor AI branding and layout.

    Kept intentionally minimal: only rules that Streamlit's native theming
    (see .streamlit/config.toml) can't express. Targets stable class names
    rather than internal data-testid soup where possible, and avoids
    !important except where Streamlit's own inline styles force it.
    """
    st.markdown(
        """
        <style>
        :root {
            --dt-navy: #1A3E5C;
            --dt-navy-mid: #2E5C82;
            --dt-sidebar-border: rgba(255, 255, 255, 0.08);
            --dt-sidebar-muted: rgba(255, 255, 255, 0.55);
            --dt-sidebar-block-gap: 0.5rem;
            --dt-sidebar-control-gap: 0.15rem;
        }

        /* Tighten default page padding */
        .main .block-container {
            padding-top: 0.5rem;
            padding-bottom: 5rem;
            max-width: 1100px;
        }

        /* Reduce space above tab bar */
        .main [data-testid="stTabs"] {
            margin-top: 0;
        }

        /* Buttons: consistent radius + smooth hover */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: var(--dt-navy-mid);
            border-color: var(--dt-navy-mid);
        }

        /* Chat input: pill shape */
        .stChatInput textarea {
            border-radius: 20px;
        }

        /* ── Sidebar shell ── */
        [data-testid="stSidebarHeader"] {
            display: none;
            height: 0;
            min-height: 0;
            padding: 0;
            margin: 0;
            overflow: hidden;
        }
        section[data-testid="stSidebar"] > div:first-child {
            top: 0;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 1.25rem 1.15rem 1.5rem;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: var(--dt-sidebar-block-gap);
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            margin-bottom: 0;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdown"]:has(.dt-sidebar-divider) {
            margin-bottom: 0;
        }

        /* Sidebar section headings (native st.subheader → stHeading) */
        section[data-testid="stSidebar"] [data-testid="stHeading"] {
            margin: 1.15rem 0 0.65rem 0;
            padding: 1rem 0 0 0;
            border-top: 1px solid var(--dt-sidebar-border);
        }
        section[data-testid="stSidebar"] [data-testid="stHeading"] h3 {
            font-size: 0.92rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            color: #e8e8e8;
            line-height: 1.4;
            margin: 0;
            padding: 0;
        }

        /* Sidebar form controls */
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--dt-sidebar-muted);
            margin-bottom: var(--dt-sidebar-control-gap);
        }
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
            border-radius: 8px;
            font-size: 0.88rem;
        }
        section[data-testid="stSidebar"] .stButton > button {
            font-size: 0.84rem;
            padding: 0.5rem 0.75rem;
            min-height: 2.25rem;
            line-height: 1.35;
        }
        section[data-testid="stSidebar"] .stDownloadButton > button {
            font-size: 0.84rem;
            padding: 0.5rem 0.75rem;
            min-height: 2.25rem;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            font-size: 0.75rem;
            opacity: 0.65;
            margin-top: 0.15rem;
        }

        /* ── Sidebar header block (brand + status) ── */
        .dt-sidebar-top {
            margin: 1rem 0 0;
        }
        .dt-sidebar-brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        .dt-sidebar-brand-icon {
            font-size: 2rem;
            line-height: 1;
            flex-shrink: 0;
        }
        .dt-sidebar-brand-text {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 0;
        }
        .dt-sidebar-brand-title {
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: -0.4px;
            line-height: 1.15;
            color: #fafafa;
        }
        .dt-sidebar-brand-tagline {
            font-size: 0.78rem;
            color: var(--dt-sidebar-muted);
            line-height: 1.3;
        }

        /* Connection status pill */
        .dt-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 9px 12px;
            border-radius: 10px;
            font-size: 0.82rem;
            margin: 0;
        }
        .dt-status .dt-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .dt-status-label {
            flex: 1;
            font-weight: 500;
        }
        .dt-status-badge {
            font-size: 0.68rem;
            padding: 2px 8px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.07);
            color: var(--dt-sidebar-muted);
            border: 1px solid rgba(255, 255, 255, 0.1);
            white-space: nowrap;
        }
        .dt-status-hint {
            font-size: 0.78rem;
            color: var(--dt-sidebar-muted);
            margin: 8px 0 0;
            line-height: 1.4;
        }
        .dt-status.connected {
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid rgba(76, 175, 80, 0.28);
        }
        .dt-status.connected .dt-dot { background: #4caf50; box-shadow: 0 0 6px rgba(76,175,80,0.5); }
        .dt-status.connected .dt-status-label { color: #6fcf7a; }
        .dt-status.offline {
            background: rgba(220, 53, 69, 0.1);
            border: 1px solid rgba(220, 53, 69, 0.28);
        }
        .dt-status.offline .dt-dot { background: #dc3545; }
        .dt-status.offline .dt-status-label { color: #f08080; }

        /* Divider before footer actions */
        .dt-sidebar-divider {
            margin: 0.85rem 0 0.35rem;
            border-top: 1px solid var(--dt-sidebar-border);
        }

        /* Footer: fixed to bottom of viewport */
        [data-testid="stMarkdown"]:has(.dt-footer) {
            height: 0;
            margin: 0;
            padding: 0;
            overflow: visible;
        }
        .dt-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 999;
            margin: 0;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            padding: 0.45rem 1.5rem 0.45rem 1rem;
            border-top: 1px solid rgba(150, 150, 150, 0.2);
            background-color: var(--background-color, #0e1117);
            font-size: 0.72rem;
            line-height: 1.3;
            color: rgba(250, 250, 250, 0.55);
            box-sizing: border-box;
        }

        /* Keep chat input above the footer bar */
        [data-testid="stBottomBlockContainer"] {
            padding-bottom: 2rem;
        }

        /* Welcome banner */
        .dt-welcome {
            border: 1px solid rgba(26,62,92,0.4);
            border-left: 3px solid var(--dt-navy);
            border-radius: 8px;
            padding: 20px 24px;
            margin: 8px 0 20px;
            background: rgba(26,62,92,0.08);
        }
        .dt-welcome h3 { margin: 0 0 10px; font-size: 1.3rem; }
        .dt-welcome p { font-size: 0.9rem; opacity: 0.8; line-height: 1.7; margin: 0; }
        .dt-welcome .dt-pills { margin-top: 14px; display: flex; gap: 16px; font-size: 0.8rem; opacity: 0.6; flex-wrap: wrap; }
        </style>
        """,
        unsafe_allow_html=True
    )


def _render_welcome_banner():
    """Show a welcome card on first load (no chat history yet)."""
    if st.session_state.chat_history:
        return
    st.markdown(
        """
        <div class="dt-welcome">
          <h3>👋 Welcome to DevTutor AI</h3>
          <p>
            Your private, local AI coding tutor — no data ever leaves your machine.<br>
            Pick a language in the sidebar, then ask anything below, or explore
            <strong>Learn Concepts</strong> and <strong>Practice</strong> in the tabs above.
          </p>
          <div class="dt-pills">
            <span>💬 Ask questions</span>
            <span>📖 Learn concepts</span>
            <span>🏋️ Practice exercises</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _render_footer():
    """Footer banner fixed to the bottom of the viewport."""
    st.markdown(
        """
        <div class="dt-footer">DevTutor AI · Built by Thomas Pravinos · Powered by Streamlit &amp; LM Studio</div>
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
    
    # Create native tab navigation with content inside each tab
    _render_modern_tabs()
    
    # Render footer
    _render_footer()


if __name__ == "__main__":
    main()
